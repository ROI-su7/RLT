import torch
from torch import nn

from rlinf.models.embodiment.openpi.native_lora import JaxLoRALinear


def test_standard_native_lora_matches_two_jax_dots():
    torch.manual_seed(0)
    base = nn.Linear(5, 7, bias=False)
    layer = JaxLoRALinear(base, rank=3, layout="standard")
    layer.lora_a.data.normal_()
    layer.lora_b.data.normal_()
    inputs = torch.randn(2, 4, 5)
    expected = base(inputs) + torch.einsum(
        "...d,rd,or->...o", inputs, layer.lora_a, layer.lora_b
    )
    torch.testing.assert_close(layer(inputs), expected)


def test_grouped_query_native_lora_preserves_head_axis():
    torch.manual_seed(1)
    base = nn.Linear(6, 8, bias=False)
    layer = JaxLoRALinear(
        base, rank=3, layout="q_grouped", num_heads=2, head_dim=4
    )
    layer.lora_a.data.normal_()
    layer.lora_b.data.normal_()
    inputs = torch.randn(2, 5, 6)
    update = torch.einsum(
        "...d,hdr,hrk->...hk", inputs, layer.lora_a, layer.lora_b
    ).flatten(-2)
    torch.testing.assert_close(layer(inputs), base(inputs) + update)


def test_grouped_output_native_lora_sums_heads():
    torch.manual_seed(2)
    base = nn.Linear(8, 6, bias=False)
    layer = JaxLoRALinear(
        base, rank=3, layout="o_grouped", num_heads=2, head_dim=4
    )
    layer.lora_a.data.normal_()
    layer.lora_b.data.normal_()
    inputs = torch.randn(2, 5, 8)
    grouped = inputs.unflatten(-1, (2, 4))
    update = torch.einsum(
        "...hk,hkr,hro->...o", grouped, layer.lora_a, layer.lora_b
    )
    torch.testing.assert_close(layer(inputs), base(inputs) + update)
