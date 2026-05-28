"""Compile + load the HIP MoE W4A8 kernel via torch.utils.cpp_extension.

Package-aware variant: resolves the .hip source relative to this file so
the loader works when imported from sglang as a package member.

The loader is cached. First call triggers hipcc; subsequent calls hit
the build dir. GEAK may add compile flags here if necessary (e.g. extra
includes), but MUST NOT change the module name or the exported function
name.

The build dir can be overridden with the env var
SGLANG_W4A8_HIP_GEAK_BUILD_DIR (useful in read-only deployments). It
defaults to a hidden dir alongside the source.
"""

import os
from pathlib import Path
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent


def _resolve_build_dir() -> Path:
    override = os.environ.get("SGLANG_W4A8_HIP_GEAK_BUILD_DIR")
    if override:
        p = Path(override).expanduser().resolve()
    else:
        p = _HERE / ".torch_build"
    p.mkdir(parents=True, exist_ok=True)
    return p


_ext = None


def get_extension():
    global _ext
    if _ext is not None:
        return _ext
    build_dir = _resolve_build_dir()
    sources = [str(_HERE / "moe_op_gemm_a8w4_hip.hip")]
    _ext = load(
        name="moe_op_gemm_a8w4_hip",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx950",
            *(os.environ.get("MOE_HIP_EXTRA_CFLAGS", "").split()),
        ],
        with_cuda=True,
        build_directory=str(build_dir),
        verbose=False,
    )
    return _ext
