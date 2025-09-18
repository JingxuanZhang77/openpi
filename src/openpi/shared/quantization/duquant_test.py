import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

import flax.nnx as nnx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from openpi.shared.quantization.duquant import (  # noqa: E402
    DuQuantConfig,
    DuQuantLinearNNX,
    DuQuantRuntimeMode,
    duquant_pack_linear,
)


def test_duquant_linear_roundtrip():
    rngs = nnx.Rngs(jax.random.key(0))
    linear = nnx.Linear(4, 6, rngs=rngs)
    weight = linear.kernel.value.T  # (out, in)
    bias = linear.bias.value

    cfg = DuQuantConfig(block_size=2, weight_bits=4, act_bits_start=8, act_bits_target=4)
    packed = duquant_pack_linear(weight, bias, cfg)
    duq = DuQuantLinearNNX(packed=packed, cfg=cfg)

    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    out_default = duq(x)
    assert out_default.shape == (2, 6)

    duq.set_runtime_mode(DuQuantRuntimeMode.W4A4)
    out_low_act = duq(x)
    assert out_low_act.shape == (2, 6)
    # Ensure the two modes are close but not identical (activation quantization effect present)
    assert not jnp.allclose(out_default, out_low_act)
