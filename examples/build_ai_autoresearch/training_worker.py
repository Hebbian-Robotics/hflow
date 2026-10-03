"""Fixed worker: execute a trial snapshot, audit base weights, and save LoRA."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from typing import Protocol as TypingProtocol
from typing import cast

from examples.build_ai_autoresearch.contracts import (
    Protocol,
    WorkerOutcome,
    WorkerStarted,
    checkpoint_digests,
    file_sha256,
    read_record,
    write_record,
)
from examples.build_ai_autoresearch.model_runtime import CpuReferenceRuntime
from examples.build_ai_autoresearch.training_api import TrainingContext
from peft import LoraConfig, PeftModel
from torch import Tensor


class TrainFunction(TypingProtocol):
    def __call__(self, context: TrainingContext) -> PeftModel: ...


def _parameter_digest(parameter: Tensor) -> str:
    return hashlib.sha256(parameter.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run_training(experiment: Path, trial_directory: Path) -> None:
    protocol = read_record(experiment / "protocol.json", Protocol)
    source = trial_directory / "train.py"
    source_digest = file_sha256(source)
    runtime = CpuReferenceRuntime(protocol)
    original_parameters = tuple(runtime.model.parameters())
    original_digests = tuple(_parameter_digest(parameter) for parameter in original_parameters)
    original_parameter_ids = {id(parameter) for parameter in original_parameters}
    started = time.monotonic()
    write_record(trial_directory / "started.tmp", WorkerStarted(started_monotonic=started))
    (trial_directory / "started.tmp").rename(trial_directory / "started.json")
    context = TrainingContext(
        runtime, experiment / "media", started + protocol.budget.training_seconds
    )
    specification = importlib.util.spec_from_file_location("trial_training", source)
    if specification is None or specification.loader is None:
        raise ValueError("cannot load the snapshotted training file")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    function = getattr(module, "train", None)
    if not callable(function):
        raise TypeError("train.py must define train(context) returning a LoRA model")
    # Python module imports are dynamically typed; validate the returned model at this boundary.
    adapted_model = cast(TrainFunction, function)(context)
    if (
        not isinstance(adapted_model, PeftModel)
        or adapted_model.get_base_model() is not context.base_model
    ):
        raise ValueError("training must return LoRA on the supplied fixed base model")
    if not context.losses or len(context.losses) > protocol.budget.max_training_steps:
        raise ValueError("training needs completed optimizer steps within the frozen cap")
    for configuration in adapted_model.peft_config.values():
        if (
            not isinstance(configuration, LoraConfig)
            or configuration.bias != "none"
            or configuration.modules_to_save
        ):
            raise ValueError(
                "only LoRA adapters are allowed; full base-weight changes are not exported"
            )
    current_base_parameters = [
        parameter for name, parameter in adapted_model.named_parameters() if ".lora_" not in name
    ]
    if {id(parameter) for parameter in current_base_parameters} != original_parameter_ids or any(
        _parameter_digest(parameter) != digest
        for parameter, digest in zip(original_parameters, original_digests, strict=True)
    ):
        raise ValueError("training changed the frozen base architecture or weights")
    adapted_model.save_pretrained(str(trial_directory / "adapter"), safe_serialization=True)
    elapsed = time.monotonic() - started
    if elapsed > protocol.budget.training_seconds or file_sha256(source) != source_digest:
        raise ValueError("training exceeded its budget or changed its snapshot")
    write_record(
        trial_directory / "outcome.json",
        WorkerOutcome(
            training_source_sha256=source_digest,
            runtime=runtime.runtime_identity,
            checkpoint_files=checkpoint_digests(trial_directory / "adapter"),
            training_seconds=elapsed,
            training_losses=context.losses,
            trainable_parameters=sum(
                parameter.numel()
                for parameter in adapted_model.parameters()
                if parameter.requires_grad
            ),
        ),
    )


if __name__ == "__main__":
    run_training(Path(sys.argv[1]), Path(sys.argv[2]))
