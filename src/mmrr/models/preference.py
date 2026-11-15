"""One-token preference loss and HaNoRec policy perturbation."""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from mmrr.models.qwen_scorer import yes_no_margin


class AnswerHead(torch.nn.Module):
    """Project just the last prompt position for one-token SFT or DPO."""

    def __init__(self, head, yes_id, no_id, preference):
        super().__init__()
        self.weight = head.weight
        self.preference = preference
        margin = yes_no_margin(self.weight.detach(), yes_id, no_id)
        self.register_buffer("margin", margin, persistent=False)

    def forward(self, hidden):
        answer_state = hidden[:, -1:, :]
        if self.preference:
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                return F.linear(answer_state.float(), self.margin.unsqueeze(0))
        return F.linear(answer_state, self.weight)


def preference_loss(policy_margin, reference_margin, chosen_yes, beta, hardness=None):
    # The Yes/No softmax denominator cancels within both the policy and reference.
    signed_delta = (policy_margin.float() - reference_margin.float()) * torch.where(chosen_yes, 1, -1)
    scale = beta
    if hardness is not None:
        if hardness.shape != signed_delta.shape or not torch.isfinite(hardness).all() or (hardness <= 0).any():
            raise ValueError("HaRS needs one finite, positive hardness per preference")
        if signed_delta.numel() < 3:
            raise ValueError("HaRS responsiveness requires at least three samples per microbatch")
        gaps = (beta * signed_delta).detach()
        mean = gaps.mean()
        denominator = torch.where(mean.abs() > 1e-8, mean, gaps.abs().mean().clamp_min(1e-8))
        normalized = gaps / denominator
        trimmed = normalized.sort().values[1:-1].mean()
        responsiveness = trimmed.sigmoid() / normalized.mean().sigmoid()
        scale = (beta * responsiveness * hardness).clamp_min(1e-6)
    return -F.logsigmoid(scale * signed_delta).mean()


@contextmanager
def nodo_noise(model, sigma):
    """Keep one noise draw live through forward and checkpoint recomputation."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("NoDO sigma must be finite and non-negative")
    if sigma == 0:
        yield
        return
    handles = []

    def add_noise(weight):
        epsilon = torch.randn_like(weight) * sigma

        def hook(_module, inputs, outputs):
            return outputs + F.linear(inputs[0], epsilon)

        return hook

    try:
        for module in model.modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                for active in module.active_adapters:
                    if active in module.lora_A and active in module.lora_B:
                        for layer in (module.lora_A[active], module.lora_B[active]):
                            handles.append(layer.register_forward_hook(add_noise(layer.weight)))
        if not handles:
            raise ValueError("NoDO requires an active linear LoRA adapter")
        yield
    finally:
        for handle in handles:
            handle.remove()
