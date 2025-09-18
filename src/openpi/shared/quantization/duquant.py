"""DuQuant integration utilities.

This module implements a deterministic variant of the DuQuant pipeline:

1. Smooth the incoming weights with λ blending.
2. Apply two block-orthogonal rotations (R1 for outputs, R2 for inputs).
3. Apply a zig-zag channel permutation P along the input dimension.
4. Quantize weights (Wq) and activations (Aq) using uniform affine quantization.

The floating-point weights are discarded after packing; at runtime
``DuQuantLinearNNX`` dequantizes on-the-fly while simulating activation
quantization. This keeps the interface simple while still exposing the
``W4A8`` → ``W4A4`` transition required for deployment.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Callable, Optional, Tuple

import logging

import numpy as np

try:  # Optional imports during static analysis; required at runtime.
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - mannequins without JAX/NNX.
    nnx = None  # type: ignore[assignment]
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]


_EPS = 1e-8


def _ensure_jax_available() -> None:
    if jax is None or jnp is None or nnx is None:
        raise ImportError("DuQuant requires JAX and flax.nnx to be installed.")


class DuQuantRuntimeMode(enum.Enum):
    """Runtime activation precision mode."""

    W4A8 = "w4a8"
    W4A4 = "w4a4"


@dataclasses.dataclass(frozen=True)
class DuQuantConfig:
    """Configuration for DuQuant weight & activation quantization."""

    weight_bits: int = 4
    act_bits_start: int = 8
    act_bits_target: int = 4
    # Smoothing and reparameterization
    lambda_smooth: float = 0.0
    block_size: int = 1  # 1 disables rotations by default for stability
    apply_per_channel: bool = True
    symmetric: bool = True
    enable_permute: bool = False
    # Activation quantization controls
    act_quantize: bool = True
    act_per_channel: bool = True
    # Robust scaling using percentiles to mitigate outliers (None -> use max)
    weight_percentile: float | None = None  # e.g., 99.5
    act_percentile: float | None = None     # e.g., 99.5
    # Optional sets to fine-tune layer coverage.
    exclude_layers: tuple[str, ...] = ()
    include_layers: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.weight_bits not in (2, 3, 4, 8):
            raise ValueError("DuQuant currently supports weight bits in {2,3,4,8}.")
        if self.act_bits_start < self.act_bits_target:
            raise ValueError("act_bits_start must be >= act_bits_target")
        if self.block_size not in (1, 2, 4, 8, 16):
            raise ValueError("Supported block sizes are 1/2/4/8/16")
        if not (0.0 <= self.lambda_smooth <= 1.0):
            raise ValueError("lambda_smooth must be in [0, 1]")
        if self.weight_percentile is not None and not (0.0 < self.weight_percentile <= 100.0):
            raise ValueError("weight_percentile must be in (0, 100]")
        if self.act_percentile is not None and not (0.0 < self.act_percentile <= 100.0):
            raise ValueError("act_percentile must be in (0, 100]")


# ---------------------------------------------------------------------------
# Helper transforms (rotations, permutations)
# ---------------------------------------------------------------------------


def _dct_matrix(n: int) -> np.ndarray:
    i = np.arange(n)
    j = np.arange(n)
    mat = np.cos(np.pi / n * (j + 0.5)[:, None] * i[None, :]).astype(np.float32)
    mat *= np.sqrt(2.0 / n)
    mat[0] /= np.sqrt(2.0)
    return mat


def _rotation_matrix(block_size: int) -> np.ndarray:
    if block_size == 1:
        return np.ones((1, 1), dtype=np.float32)
    return _dct_matrix(block_size)


def _apply_block_rotation_rows(weight: jnp.ndarray, block_size: int, matrix: jnp.ndarray) -> jnp.ndarray:
    if block_size == 1:
        return weight
    out_dim = weight.shape[0]
    pad = (-out_dim) % block_size
    if pad:
        pad_block = jnp.zeros((pad, weight.shape[1]), dtype=weight.dtype)
        weight = jnp.concatenate([weight, pad_block], axis=0)
    num_blocks = weight.shape[0] // block_size
    weight = weight.reshape(num_blocks, block_size, weight.shape[1])
    rot = jnp.asarray(matrix)
    rotated = jax.vmap(lambda block: rot @ block)(weight)
    rotated = rotated.reshape(-1, weight.shape[-1])
    rotated = rotated[:out_dim]
    return rotated


def _apply_block_rotation_cols(weight: jnp.ndarray, block_size: int, matrix: jnp.ndarray) -> jnp.ndarray:
    if block_size == 1:
        return weight
    out_dim, in_dim = weight.shape
    pad = (-in_dim) % block_size
    if pad:
        weight = jnp.pad(weight, ((0, 0), (0, pad)))
    num_blocks = weight.shape[1] // block_size
    weight = weight.reshape(out_dim, num_blocks, block_size)
    rot = jnp.asarray(matrix)
    rotated = jnp.einsum("onb,cb->onc", weight, rot.T)
    rotated = rotated.reshape(out_dim, -1)
    return rotated[:, :in_dim]


def _zigzag_permutation(n: int) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    head = np.arange(0, n, 2)
    tail = np.arange(1, n, 2)[::-1]
    return np.concatenate([head, tail]).astype(np.int32)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _calc_scale_symmetric(x: jnp.ndarray, qmax: float, percentile: float | None) -> jnp.ndarray:
    if percentile is None:
        max_val = jnp.max(jnp.abs(x))
    else:
        max_val = jnp.percentile(jnp.abs(x), percentile)
    return jnp.maximum(max_val / qmax, _EPS)


def _affine_quantize_tensor(x: jnp.ndarray, n_bits: int, symmetric: bool, *, percentile: float | None = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    qmax = float(2 ** (n_bits - 1) - 1)
    if symmetric:
        scale = _calc_scale_symmetric(x, qmax, percentile)
        zero = jnp.zeros_like(scale)
        q = jnp.clip(jnp.round(x / scale), -qmax, qmax)
    else:
        qmin, qmax_full = 0.0, float(2**n_bits - 1)
        x_min = jnp.min(x)
        x_max = jnp.max(x)
        scale = jnp.maximum((x_max - x_min) / (qmax_full - qmin), _EPS)
        zero = jnp.round(qmin - x_min / scale)
        q = jnp.clip(jnp.round(x / scale + zero), qmin, qmax_full)
    return q.astype(jnp.float32), scale.astype(jnp.float32), zero.astype(jnp.float32)


def _affine_dequantize_tensor(q: jnp.ndarray, scale: jnp.ndarray, zero: jnp.ndarray, symmetric: bool) -> jnp.ndarray:
    if symmetric:
        return q * scale
    return (q - zero) * scale


def _quantize_weight(weight: jnp.ndarray, cfg: DuQuantConfig) -> dict:
    weight_bits = cfg.weight_bits
    symmetric = cfg.symmetric
    if cfg.apply_per_channel:
        def _q_row(row):
            return _affine_quantize_tensor(row, weight_bits, symmetric, percentile=cfg.weight_percentile)
        q, scale, zero = jax.vmap(_q_row, in_axes=0)(weight)
    else:
        q, scale, zero = _affine_quantize_tensor(weight, weight_bits, symmetric, percentile=cfg.weight_percentile)
    return {"q": q, "scale": scale, "zero": zero}


def _dequantize_weight(packed: dict, cfg: DuQuantConfig) -> jnp.ndarray:
    q = packed["q"]
    scale = packed["scale"]
    zero = packed["zero"]
    if cfg.apply_per_channel:
        return jax.vmap(_affine_dequantize_tensor, in_axes=(0, 0, 0, None))(q, scale, zero, cfg.symmetric)
    return _affine_dequantize_tensor(q, scale, zero, cfg.symmetric)


# ---------------------------------------------------------------------------
# Pack/unpack API
# ---------------------------------------------------------------------------


def duquant_pack_linear(weight: jnp.ndarray, bias: Optional[jnp.ndarray], cfg: DuQuantConfig) -> dict:
    """Pack a weight matrix using DuQuant transformations."""

    _ensure_jax_available()
    cfg.validate()

    w = jnp.asarray(weight, jnp.float32)

    if cfg.lambda_smooth > 0.0:
        mean = jnp.mean(w, axis=-1, keepdims=True)
        w = (1.0 - cfg.lambda_smooth) * w + cfg.lambda_smooth * mean

    rot_mat = jnp.asarray(_rotation_matrix(cfg.block_size))
    w = _apply_block_rotation_rows(w, cfg.block_size, rot_mat)
    w = _apply_block_rotation_cols(w, cfg.block_size, rot_mat)

    perm = None
    if cfg.enable_permute and w.shape[1] > 1:
        perm = _zigzag_permutation(w.shape[1])
        w = w[:, perm]

    packed = _quantize_weight(w, cfg)
    packed.update(
        {
            "orig_shape": tuple(weight.shape),
            "block_size": cfg.block_size,
            "rotation": rot_mat,
            "perm": None if perm is None else jnp.asarray(perm, dtype=jnp.int32),
            "bias": None if bias is None else jnp.asarray(bias, jnp.float32),
        }
    )
    # Optional: caller may enrich with rank; we keep weight-only here.
    return packed


if nnx is not None:  # pragma: no branch - definition guarded by import

    class DuQuantLinearNNX(nnx.Module):  # type: ignore[misc]
        """NNX module that wraps a DuQuant-processed linear layer."""

        def __init__(self, *, packed: dict, cfg: DuQuantConfig):
            super().__init__()
            _ensure_jax_available()
            self.cfg = cfg
            self.weight_q = nnx.Variable(packed["q"])
            self.weight_scale = nnx.Variable(packed["scale"])
            self.weight_zero = nnx.Variable(packed["zero"])
            self.rotation = nnx.Variable(packed["rotation"].astype(np.float32))
            self.block_size = int(packed["block_size"])
            self.perm = None if packed.get("perm") is None else nnx.Variable(packed["perm"])
            self.bias = None if packed.get("bias") is None else nnx.Variable(packed["bias"])
            self.in_features = int(packed["orig_shape"][1])
            self.out_features = int(packed["orig_shape"][0])
            self.act_bits: int = int(cfg.act_bits_start)

        # ------------------------------------------------------------------
        # Runtime controls
        # ------------------------------------------------------------------
        def set_runtime_mode(self, mode: DuQuantRuntimeMode) -> None:
            self.act_bits = int(self.cfg.act_bits_target if mode == DuQuantRuntimeMode.W4A4 else self.cfg.act_bits_start)

        # ------------------------------------------------------------------
        # Internal helpers
        # ------------------------------------------------------------------
        def _dequantize_weight(self) -> jnp.ndarray:
            packed = {"q": self.weight_q.value, "scale": self.weight_scale.value, "zero": self.weight_zero.value}
            return _dequantize_weight(packed, self.cfg)

        def _apply_forward_columns(self, x: jnp.ndarray) -> jnp.ndarray:
            if self.perm is not None:
                x = x[..., self.perm.value]
            if self.block_size > 1:
                rot = self.rotation.value
                x = _apply_block_rotation_cols(x, self.block_size, rot)
            return x

        def _apply_inverse_rows(self, y: jnp.ndarray) -> jnp.ndarray:
            if self.block_size > 1:
                rot_t = jnp.asarray(self.rotation.value.T)
                y = _apply_block_rotation_rows(y, self.block_size, rot_t)
            return y

        # ------------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------------
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            orig_dtype = x.dtype
            x_shape = x.shape
            x = x.reshape(-1, x_shape[-1])
            x = jnp.asarray(x, jnp.float32)
            x = self._apply_forward_columns(x)

            # Optional activation quantization (default on); can be disabled for stability
            if self.cfg.act_quantize:
                if self.cfg.act_per_channel:
                    # per-feature scaling across batch tokens
                    qmax = float(2 ** (int(self.act_bits) - 1) - 1)
                    if self.cfg.act_percentile is None:
                        max_abs = jnp.max(jnp.abs(x), axis=0)
                    else:
                        max_abs = jnp.percentile(jnp.abs(x), self.cfg.act_percentile, axis=0)
                    scale = jnp.maximum(max_abs / qmax, _EPS)
                    x_q = jnp.clip(jnp.round(x / scale), -qmax, qmax)
                    x = (x_q * scale).astype(jnp.float32)
                else:
                    x_q, x_scale, x_zero = _affine_quantize_tensor(
                        x, int(self.act_bits), True, percentile=self.cfg.act_percentile
                    )
                    x = _affine_dequantize_tensor(x_q, x_scale, x_zero, True)

            weight = self._dequantize_weight()
            y = x @ weight.T
            y = self._apply_inverse_rows(y)
            if self.bias is not None:
                y = y + self.bias.value
            y = y.reshape(*x_shape[:-1], self.out_features)
            return y.astype(orig_dtype)

else:  # pragma: no cover - executed only when flax/nnx is unavailable

    class DuQuantLinearNNX:  # type: ignore[misc]
        """Placeholder that raises if flax.nnx is missing."""

        def __init__(self, *_, **__):
            raise ImportError("flax.nnx is required for DuQuantLinearNNX")

        def set_runtime_mode(self, *_args, **_kwargs):
            raise ImportError("flax.nnx is required for DuQuantLinearNNX")



def _get_linear_params(linear_module) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    for name in ("kernel", "weight"):
        kernel = getattr(linear_module, name, None)
        if kernel is not None:
            break
    else:  # pragma: no cover - defensive
        raise AttributeError("Could not find kernel/weight on module")
    bias = getattr(linear_module, "bias", None)

    def _maybe_value(x):
        try:
            return x.value
        except AttributeError:
            return x

    weight = _maybe_value(kernel)
    bias_val = None if bias is None else _maybe_value(bias)
    return jnp.asarray(weight), None if bias_val is None else jnp.asarray(bias_val)


def replace_linear_with_duquant(parent_module, attr_name: str, cfg: DuQuantConfig) -> tuple[bool, dict[str, float]]:
    if not hasattr(parent_module, attr_name):
        return False, {}
    child = getattr(parent_module, attr_name)
    try:
        kernel, bias = _get_linear_params(child)
    except AttributeError:
        return False, {}

    # Convert kernel to (out, in)
    if kernel.ndim != 2:
        return False, {}
    in_features = kernel.shape[0]
    out_features = kernel.shape[1]
    if hasattr(child, "in_features") and hasattr(child, "out_features"):
        in_features = int(getattr(child, "in_features"))
        out_features = int(getattr(child, "out_features"))
    weight = kernel
    if weight.shape == (in_features, out_features):
        weight = weight.T
    packed = duquant_pack_linear(weight, bias, cfg)
    duq = DuQuantLinearNNX(packed=packed, cfg=cfg)
    setattr(parent_module, attr_name, duq)
    fro_err = jnp.linalg.norm(weight - _dequantize_weight(packed, cfg)) / (jnp.linalg.norm(weight) + 1e-12)
    metrics = {"fro_error": float(fro_err), "n_bits": float(cfg.weight_bits)}
    if "rank" in packed:
        try:
            metrics["rank"] = float(packed["rank"])  # type: ignore[arg-type]
        except Exception:
            pass
    return True, metrics


def enable_duquant_for_pi05(
    model,
    cfg: DuQuantConfig,
    *,
    layers: tuple[str, ...] | None = None,
) -> list[str]:
    cfg.validate()
    replaced: list[str] = []
    if layers is not None:
        targets = list(layers)
    else:
        targets = [
            "action_in_proj",
            "action_out_proj",
        ]
        if getattr(model, "pi05", False):
            targets.extend(["time_mlp_in", "time_mlp_out"])
        else:
            targets.extend(["state_proj", "action_time_mlp_in", "action_time_mlp_out"])

    include = set(cfg.include_layers)
    exclude = set(cfg.exclude_layers) - include

    for name in targets:
        if name in exclude:
            continue
        ok, metrics = replace_linear_with_duquant(model, name, cfg)
        if ok:
            replaced.append(name)
            if metrics:
                if "rank" in metrics:
                    logging.info(
                        "DuQuant metrics for %s: fro_error=%.6f rank=%s bits=%s",
                        name,
                        metrics.get("fro_error", float("nan")),
                        metrics.get("rank"),
                        metrics.get("n_bits"),
                    )
                else:
                    logging.info(
                        "DuQuant metrics for %s: fro_error=%.6f bits=%s",
                        name,
                        metrics.get("fro_error", float("nan")),
                        metrics.get("n_bits"),
                    )
    return replaced


def set_duquant_runtime_mode(module, mode: DuQuantRuntimeMode) -> None:
    if nnx is None:
        raise ImportError("flax.nnx is required to traverse modules.")
    for attr in dir(module):
        if attr.startswith("__"):
            continue
        child = getattr(module, attr)
        if isinstance(child, DuQuantLinearNNX):
            child.set_runtime_mode(mode)
        elif isinstance(child, nnx.Module):  # type: ignore[arg-type]
            set_duquant_runtime_mode(child, mode)


def enable_duquant_for_module(
    module, cfg: DuQuantConfig, *, name_filter: Optional[Callable[[tuple[str, ...]], bool]] = None, _path: tuple[str, ...] = ()
) -> list[str]:
    """Recursively replace ``nnx.Linear`` modules by DuQuant wrappers."""

    if nnx is None:
        raise ImportError("flax.nnx is required for DuQuant integration.")

    replaced: list[str] = []
    for attr in dir(module):
        if attr.startswith("__"):
            continue
        child = getattr(module, attr)
        path = _path + (attr,)
        if isinstance(child, nnx.Module):  # type: ignore[arg-type]
            if isinstance(child, nnx.Linear) and (name_filter is None or name_filter(path)):
                ok, metrics = replace_linear_with_duquant(module, attr, cfg)
                if ok:
                    replaced.append(".".join(path))
                    if metrics:
                        if "rank" in metrics:
                            logging.info(
                                "DuQuant metrics for %s: fro_error=%.6f rank=%s bits=%s",
                                ".".join(path),
                                metrics.get("fro_error", float("nan")),
                                metrics.get("rank"),
                                metrics.get("n_bits"),
                            )
                        else:
                            logging.info(
                                "DuQuant metrics for %s: fro_error=%.6f bits=%s",
                                ".".join(path),
                                metrics.get("fro_error", float("nan")),
                                metrics.get("n_bits"),
                            )
            else:
                replaced.extend(enable_duquant_for_module(child, cfg, name_filter=name_filter, _path=path))
    return replaced
