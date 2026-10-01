"""Small, explicit CPU reference runtime; no serving fallback or model judge."""

from __future__ import annotations

import hashlib
import io
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import peft
import torch
import transformers
from examples.build_ai_autoresearch.contracts import (
    Evaluation,
    PredictionRecord,
    Protocol,
    SampleRecord,
    SamplingBalance,
    TrainingRecipe,
    score_predictions,
)
from examples.build_ai_autoresearch.prepare import HandCount
from hflow.build_ai_vlm_checks import BUILD_AI_HAND_VISIBILITY_PROMPT
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image, ImageOps
from torch import Tensor
from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor


@dataclass(frozen=True)
class TrainingOutcome:
    losses: tuple[float, ...]
    elapsed_seconds: float
    trainable_parameters: int


class CpuReferenceRuntime:
    def __init__(self, protocol: Protocol, adapter_directory: Path | None = None) -> None:
        self.protocol = protocol
        torch.set_num_threads(protocol.budget.cpu_threads)
        torch.manual_seed(protocol.budget.seed)
        torch.use_deterministic_algorithms(True)
        self.processor = SmolVLMProcessor.from_pretrained(
            protocol.model_id,
            revision=protocol.model_revision,
            size={"longest_edge": protocol.image_edge},
            max_image_size={"longest_edge": protocol.image_edge},
            do_image_splitting=False,
        )
        if (
            self.processor.image_processor.do_image_splitting
            or self.processor.image_processor.size.longest_edge != protocol.image_edge
            or self.processor.image_processor.max_image_size
            != {"longest_edge": protocol.image_edge}
        ):
            raise ValueError("processor settings differ from the frozen protocol")
        base_model = SmolVLMForConditionalGeneration.from_pretrained(
            protocol.model_id,
            revision=protocol.model_revision,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        self.model: SmolVLMForConditionalGeneration | PeftModel = (
            PeftModel.from_pretrained(base_model, str(adapter_directory))
            if adapter_directory is not None
            else base_model
        )

    @property
    def runtime_identity(self) -> dict[str, str]:
        return {
            "backend": self.protocol.backend,
            "device": "cpu",
            "dtype": "float32",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "reference_reason": self.protocol.budget.reference_reason,
        }

    def _image(self, sample: SampleRecord, media_directory: Path) -> Image.Image:
        path = media_directory / sample.image_filename
        encoded = path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != sample.image_sha256:
            raise ValueError(f"image changed: {sample.sample_id}")
        with Image.open(io.BytesIO(encoded)) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError("image exceeds the fixed preparation pixel budget")
            return ImageOps.exif_transpose(image).convert("RGB")

    def _render(self, answer: str | None = None) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": BUILD_AI_HAND_VISIBILITY_PROMPT},
                ],
            }
        ]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        rendered = self.processor.apply_chat_template(
            messages,  # ty: ignore[invalid-argument-type] -- upstream annotation omits multimodal content lists
            tokenize=False,
            add_generation_prompt=answer is None,
            processor_kwargs={
                "num_frames": self.processor.video_processor.num_frames,
                "fps": self.processor.video_processor.fps,
            },
        )
        if not isinstance(rendered, str):
            raise TypeError("chat template must return a single rendered string")
        return rendered

    def _inputs(self, image: Image.Image, rendered: str) -> dict[str, Tensor]:
        processed = self.processor(
            images=[image], text=rendered, return_tensors="pt", padding=False
        )
        inputs: dict[str, Tensor] = {}
        for name, value in processed.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"processor output {name!r} must be a tensor")
            inputs[name] = value
        return inputs

    def evaluate(self, samples: Sequence[SampleRecord], media_directory: Path) -> Evaluation:
        self.model.eval()
        predictions: list[PredictionRecord] = []
        started = time.perf_counter()
        with torch.inference_mode():
            for sample in samples:
                image = self._image(sample, media_directory)
                inputs = self._inputs(image, self._render())
                generated = self.model.generate(  # ty: ignore[invalid-argument-type, missing-argument] -- upstream self protocol rejects its concrete model
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    pixel_values=inputs["pixel_values"],
                    pixel_attention_mask=inputs["pixel_attention_mask"],
                    do_sample=False,
                    max_new_tokens=self.protocol.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )
                if not isinstance(generated, Tensor):
                    raise TypeError("generation must return token tensors")
                raw_answer = self.processor.tokenizer.decode(
                    generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
                answer = raw_answer.strip()
                prediction = HandCount(int(answer)) if answer in ("0", "1", "2") else None
                predictions.append(
                    PredictionRecord(sample=sample, raw_answer=raw_answer, prediction=prediction)
                )
        return Evaluation(
            predictions=tuple(predictions),
            metrics=score_predictions(predictions),
            per_corpus={
                corpus.value: score_predictions(
                    [
                        prediction
                        for prediction in predictions
                        if corpus in prediction.sample.corpora
                    ]
                )
                for corpus in {corpus for sample in samples for corpus in sample.corpora}
            },
            elapsed_seconds=time.perf_counter() - started,
        )

    def train(
        self,
        recipe: TrainingRecipe,
        samples: Sequence[SampleRecord],
        media_directory: Path,
        output_directory: Path,
    ) -> TrainingOutcome:
        if not isinstance(self.model, SmolVLMForConditionalGeneration):
            raise ValueError("each trial must train from fresh base weights")
        adapted_model = get_peft_model(
            self.model,
            LoraConfig(
                r=recipe.lora_rank,
                lora_alpha=2 * recipe.lora_rank,
                target_modules=r".*text_model.*\.(q_proj|v_proj)$",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        if not isinstance(adapted_model, PeftModel):
            raise TypeError("LoRA training requires a single PEFT model")
        self.model = adapted_model
        self.model.train()
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=recipe.learning_rate, weight_decay=0.0)
        class_counts = Counter(sample.hand_count for sample in samples)
        match recipe.sampling_balance:
            case SamplingBalance.UNIFORM:
                sample_weights = [1.0] * len(samples)
            case SamplingBalance.CLASS_BALANCED:
                sample_weights = [1 / class_counts[sample.hand_count] for sample in samples]
        random_source = random.Random(self.protocol.budget.seed)
        losses: list[float] = []
        started = time.perf_counter()
        for _ in range(self.protocol.budget.training_steps):
            sample = random_source.choices(samples, weights=sample_weights, k=1)[0]
            image = self._image(sample, media_directory)
            prompt_inputs = self._inputs(image, self._render())
            inputs = self._inputs(image, self._render(str(int(sample.hand_count))))
            prompt_ids = prompt_inputs["input_ids"]
            if not torch.equal(inputs["input_ids"][:, : prompt_ids.shape[1]], prompt_ids):
                raise ValueError("assistant-loss masking needs an exact tokenized prompt prefix")
            labels = inputs["input_ids"].clone()
            labels[:, : prompt_ids.shape[1]] = -100
            labels[inputs["attention_mask"] == 0] = -100
            if not torch.any(labels != -100):
                raise ValueError("training sample has no assistant target tokens")
            optimizer.zero_grad(set_to_none=True)
            output = self.model(**inputs, labels=labels, use_cache=False)
            loss = output.loss
            if not isinstance(loss, Tensor) or not torch.isfinite(loss):
                raise ValueError("training loss must be a finite tensor")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.model.save_pretrained(str(output_directory), safe_serialization=True)
        return TrainingOutcome(
            tuple(losses),
            time.perf_counter() - started,
            sum(parameter.numel() for parameter in trainable),
        )

    def export(self, output_directory: Path) -> None:
        model = self.model.merge_and_unload() if isinstance(self.model, PeftModel) else self.model
        model.save_pretrained(str(output_directory), safe_serialization=True)
        self.processor.save_pretrained(str(output_directory))
