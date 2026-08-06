# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version

import torch

from vllm.platforms import current_platform

_FLAGTREE_VERSION = "0.6.0"
_KDA_HEAD_DIM = 128
_KDA_CHUNK_SIZE = 16


def is_tle_kda_supported(
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> bool:
    """Return whether the H20/SM90 TLE KDA contract can be used."""
    if (
        not (current_platform.is_cuda() or torch.cuda.is_available())
        or not torch.cuda.is_available()
        or head_dim != _KDA_HEAD_DIM
        or dtype != torch.bfloat16
        or lower_bound is None
        or not -5.0 <= lower_bound < 0.0
    ):
        return False

    capability = current_platform.get_device_capability()
    if capability is None:
        # A source checkout has no vLLM `_C` metadata, so platform discovery
        # can remain unspecified even though the CUDA driver is usable.
        capability = torch.cuda.get_device_capability()
    if capability is None:
        return False
    major = capability.major if hasattr(capability, "major") else capability[0]
    return major == 9


def _load_tle_chunk_kda():
    try:
        flagtree_version = version("flagtree")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "KDA TLE requires the FlagTree distribution; standalone Triton "
            "is not a substitute."
        ) from exc

    if flagtree_version != _FLAGTREE_VERSION:
        raise RuntimeError(
            "KDA TLE requires flagtree==0.6.0, "
            f"but found flagtree=={flagtree_version}."
        )

    try:
        importlib.import_module("triton.experimental.tle.language")
    except Exception as exc:
        raise RuntimeError(
            "FlagTree 0.6.0 is installed, but its TLE language extension "
            "cannot be imported."
        ) from exc

    try:
        module = importlib.import_module("flaggems_vllm.ops.FLA.chunk_kda")
    except Exception as exc:
        raise RuntimeError(
            "KDA TLE requires the FlagGems-vllm chunk_kda implementation."
        ) from exc

    if not getattr(module, "HAS_TLE_KDA", False):
        raise RuntimeError(
            "FlagGems-vllm did not enable its TLE KDA path; refusing to "
            "silently run the plain Triton fallback."
        )
    return module.chunk_kda


def tle_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlagTree TLE KDA prefill with the vLLM state/cache contract."""
    chunk_kda = _load_tle_chunk_kda()

    if cu_seqlens.dtype != torch.long:
        cu_seqlens = cu_seqlens.to(dtype=torch.long)

    output, final_state = chunk_kda(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=False,
        safe_gate=True,
        lower_bound=lower_bound,
        state_v_first=True,
        cu_seqlens=cu_seqlens.contiguous(),
        chunk_size=_KDA_CHUNK_SIZE,
        A_log=A_log.contiguous(),
        dt_bias=dt_bias.view(-1, q.shape[-1]).contiguous(),
    )
    assert final_state is not None
    return output, final_state
