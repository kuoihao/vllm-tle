# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.kimi_k3.nvidia.kda import resolve_kda_prefill_backend
from vllm.models.kimi_k3.nvidia.ops.tle_kda import is_tle_kda_supported


@pytest.mark.parametrize(
    ("head_dim", "dtype", "lower_bound"),
    [
        (64, torch.bfloat16, -3.0),
        (128, torch.float16, -3.0),
        (128, torch.bfloat16, None),
        (128, torch.bfloat16, 0.0),
    ],
)
def test_tle_kda_rejects_unsupported_contract(head_dim, dtype, lower_bound):
    assert not is_tle_kda_supported(head_dim, dtype, lower_bound)


def test_tle_kda_backend_is_fail_closed_without_supported_cuda():
    if is_tle_kda_supported(128, torch.bfloat16, -3.0):
        pytest.skip("running on a supported SM90 CUDA device")
    with pytest.raises(RuntimeError, match="KDA TLE requires CUDA SM90"):
        resolve_kda_prefill_backend("tle", 128, torch.bfloat16, -3.0)
