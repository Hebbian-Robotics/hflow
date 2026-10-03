"""Agent-editable training code: keep the task, evaluator, and budget fixed."""

from __future__ import annotations

import random

import torch
from examples.build_ai_autoresearch.training_api import TrainingContext
from peft import LoraConfig, PeftModel, get_peft_model
from torch import Tensor


def train(context: TrainingContext) -> PeftModel:
    adapted_model = get_peft_model(
        context.base_model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=r".*text_model.*\.(q_proj|v_proj)$",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if not isinstance(adapted_model, PeftModel):
        raise TypeError("training must return a single LoRA model")
    adapted_model.train()
    trainable_parameters = [
        parameter for parameter in adapted_model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=0.0002, weight_decay=0.0)
    random_source = random.Random(context.seed)
    # Leave time for the fixed worker to audit and save the adapter before its deadline.
    while context.remaining_seconds > 5 and len(context.losses) < context.max_steps:
        sample = random_source.choice(context.samples)
        batch = context.batch(sample)
        optimizer.zero_grad(set_to_none=True)
        output = adapted_model(**batch.inputs, labels=batch.labels, use_cache=False)
        loss = output.loss
        if not isinstance(loss, Tensor) or not torch.isfinite(loss):
            raise ValueError("training loss must be a finite tensor")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        context.record_loss(float(loss.detach()))
    return adapted_model
