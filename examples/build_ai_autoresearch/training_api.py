"""Fixed training inputs and compute limits shared by editable training code."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from examples.build_ai_autoresearch.contracts import SampleRecord
from examples.build_ai_autoresearch.model_runtime import CpuReferenceRuntime
from PIL import Image
from torch import Tensor
from transformers import SmolVLMForConditionalGeneration


@dataclass(frozen=True)
class TrainingBatch:
    inputs: dict[str, Tensor]
    labels: Tensor


class TrainingContext:
    def __init__(
        self, runtime: CpuReferenceRuntime, media_directory: Path, deadline: float
    ) -> None:
        if not isinstance(runtime.model, SmolVLMForConditionalGeneration):
            raise ValueError("training must start from the pinned base model")
        self.base_model = runtime.model
        self.samples = runtime.protocol.train
        self.seed = runtime.protocol.budget.seed
        self.max_steps = runtime.protocol.budget.max_training_steps
        self._runtime = runtime
        self._media_directory = media_directory
        self._deadline = deadline
        self._losses: list[float] = []

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self._deadline - time.monotonic())

    @property
    def losses(self) -> tuple[float, ...]:
        return tuple(self._losses)

    def record_loss(self, loss: float) -> None:
        if not math.isfinite(loss) or len(self._losses) >= self.max_steps:
            raise ValueError("loss must be finite and the optimizer-step cap must be respected")
        if self.remaining_seconds <= 0:
            raise TimeoutError("training exceeded the frozen compute budget")
        self._losses.append(loss)

    def batch(
        self, sample: SampleRecord, transform: Callable[[Image.Image], Image.Image] | None = None
    ) -> TrainingBatch:
        if sample not in self.samples:
            raise ValueError("training batches must come from the frozen training subset")
        image = self._runtime._image(sample, self._media_directory)
        if transform is not None:
            image = transform(image)
            if image.mode != "RGB" or image.width * image.height > 32_000_000:
                raise ValueError("training augmentation must return bounded RGB pixels")
        prompt_inputs = self._runtime._inputs(image, self._runtime._render())
        inputs = self._runtime._inputs(image, self._runtime._render(str(int(sample.hand_count))))
        prompt_ids = prompt_inputs["input_ids"]
        if not torch.equal(inputs["input_ids"][:, : prompt_ids.shape[1]], prompt_ids):
            raise ValueError("assistant-loss masking needs an exact tokenized prompt prefix")
        labels = inputs["input_ids"].clone()
        labels[:, : prompt_ids.shape[1]] = -100
        labels[inputs["attention_mask"] == 0] = -100
        if not torch.any(labels != -100):
            raise ValueError("training sample has no assistant target tokens")
        return TrainingBatch(inputs, labels)
