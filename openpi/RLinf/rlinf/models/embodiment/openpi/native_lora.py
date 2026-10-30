"""Native PyTorch representation of OpenPI's JAX Gemma LoRA branches."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class JaxLoRALinear(nn.Module):
    """Linear layer that keeps the JAX LoRA branch separate from the base.

    ``standard`` is the usual ``B(A(x))`` layout.  OpenPI's query and output
    projections use an independent low-rank branch per attention head; those
    layouts cannot be represented by one conventional rank-r LoRA module.
    """

    def __init__(
        self,
        linear: nn.Linear,
        *,
        rank: int,
        layout: str,
        num_heads: int = 1,
        head_dim: int = 1,
        scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.layout = layout
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.scaling = float(scaling)

        self.weight = linear.weight
        self.bias = linear.bias
        factory = {"device": self.weight.device, "dtype": self.weight.dtype}
        if layout == "standard":
            self.lora_a = nn.Parameter(torch.zeros(rank, self.in_features, **factory))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank, **factory))
        elif layout == "q_grouped":
            self.lora_a = nn.Parameter(
                torch.zeros(num_heads, self.in_features, rank, **factory)
            )
            self.lora_b = nn.Parameter(
                torch.zeros(num_heads, rank, head_dim, **factory)
            )
        elif layout == "o_grouped":
            self.lora_a = nn.Parameter(
                torch.zeros(num_heads, head_dim, rank, **factory)
            )
            self.lora_b = nn.Parameter(
                torch.zeros(num_heads, rank, self.out_features, **factory)
            )
        else:
            raise ValueError(f"Unsupported JAX LoRA layout: {layout}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = F.linear(inputs, self.weight, self.bias)
        # JAX explicitly casts A/B to the activation dtype before both einsums.
        a = self.lora_a.to(dtype=inputs.dtype)
        b = self.lora_b.to(dtype=inputs.dtype)
        if self.layout == "standard":
            update = F.linear(F.linear(inputs, a), b)
        elif self.layout == "q_grouped":
            low_rank = torch.einsum("...d,hdr->...hr", inputs, a)
            update = torch.einsum("...hr,hrk->...hk", low_rank, b).flatten(-2)
        else:
            grouped = inputs.unflatten(-1, (self.num_heads, self.head_dim))
            low_rank = torch.einsum("...hk,hkr->...hr", grouped, a)
            update = torch.einsum("...hr,hro->...o", low_rank, b)
        return result + update * self.scaling


def _wrap_layer(layer: nn.Module, *, rank: int) -> None:
    attention = layer.self_attn
    num_heads = int(attention.config.num_attention_heads)
    head_dim = int(attention.config.head_dim)
    attention.q_proj = JaxLoRALinear(
        attention.q_proj,
        rank=rank,
        layout="q_grouped",
        num_heads=num_heads,
        head_dim=head_dim,
    )
    attention.k_proj = JaxLoRALinear(attention.k_proj, rank=rank, layout="standard")
    attention.v_proj = JaxLoRALinear(attention.v_proj, rank=rank, layout="standard")
    attention.o_proj = JaxLoRALinear(
        attention.o_proj,
        rank=rank,
        layout="o_grouped",
        num_heads=num_heads,
        head_dim=head_dim,
    )
    layer.mlp.gate_proj = JaxLoRALinear(
        layer.mlp.gate_proj, rank=rank, layout="standard"
    )
    layer.mlp.up_proj = JaxLoRALinear(
        layer.mlp.up_proj, rank=rank, layout="standard"
    )
    layer.mlp.down_proj = JaxLoRALinear(
        layer.mlp.down_proj, rank=rank, layout="standard"
    )


def apply_native_jax_lora(model: nn.Module) -> int:
    """Inject the exact JAX LoRA topology into both Gemma stacks."""
    count = 0
    paligemma_layers = (
        model.paligemma_with_expert.paligemma.model.language_model.layers
    )
    expert_layers = model.paligemma_with_expert.gemma_expert.model.layers
    for layer in paligemma_layers:
        _wrap_layer(layer, rank=16)
        count += 7
    for layer in expert_layers:
        _wrap_layer(layer, rank=32)
        count += 7
    return count


def native_lora_parameter_names(model: nn.Module) -> set[str]:
    return {
        name
        for name, _ in model.named_parameters()
        if name.endswith((".lora_a", ".lora_b"))
    }
