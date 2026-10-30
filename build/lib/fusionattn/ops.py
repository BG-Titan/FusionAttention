"""ctypes loader + tensor marshalling for the FusionAttention .so (no torch headers).

The .so is built by setup.py with plain nvcc (extern "C" ABI, see
csrc/fusionattn.cu) and searched for in the package directory, or via the
FUSIONATTN_SO environment variable. torch is imported lazily inside the call
paths so this module stays importable on CPU-only boxes.
"""

from __future__ import annotations

import ctypes
import glob
import importlib.machinery
import os
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_cache: dict[str, object] = {}


def _candidate_paths() -> list[str]:
    env = os.environ.get("FUSIONATTN_SO")
    if env:
        return [env]
    suffixes = list(importlib.machinery.EXTENSION_SUFFIXES) + [".so"]
    out: list[str] = []
    for suf in suffixes:
        out += sorted(str(p) for p in _PKG_DIR.glob(f"_fusionattn*{suf}"))
    return out


class _Lib:
    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        lib.fusionattn_abi.restype = ctypes.c_int
        lib.fusionattn_abi.argtypes = []
        lib.fusionattn_has_gpu.restype = ctypes.c_int
        lib.fusionattn_has_gpu.argtypes = []
        i64, i32, f32, ptr = ctypes.c_int64, ctypes.c_int32, ctypes.c_float, ctypes.c_void_p
        lib.fusionattn_cache_update.restype = ctypes.c_int
        lib.fusionattn_cache_update.argtypes = [
            ptr, ptr, ptr, ptr, i64, i64, i64, ptr, i32, i32, i32, i32, i32, ptr,
        ]
        lib.fusionattn_forward.restype = ctypes.c_int
        lib.fusionattn_forward.argtypes = [
            ptr, ptr, ptr, ptr, i64, i64, i64, ptr, i64, ptr, ptr, i32,
            i32, i32, i32, i32, i32, f32, i32, i32, ptr,
        ]


def native_available() -> bool:
    if "lib" in _cache:
        return bool(_cache["lib"])
    lib = None
    if sys.platform == "win32":
        os.environ.setdefault("PATH", "")
    for path in _candidate_paths():
        try:
            cand = ctypes.CDLL(path)
            if cand.fusionattn_abi() == 1:
                lib = _Lib(cand)
                break
        except OSError:
            continue
    _cache["lib"] = lib
    return lib is not None


def _dtype_code(t) -> int:
    import torch

    dt = getattr(t, "dtype", t)  # accept tensor or torch.dtype
    if dt == torch.float16:
        return 0
    if dt == torch.bfloat16:
        return 1
    raise ValueError(f"fusionattn: unsupported dtype {dt} (need float16/bfloat16)")


def _stream_ptr() -> int:
    import torch

    return torch.cuda.current_stream().cuda_stream


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"fusionattn {what} failed: code={rc} (cudaError_t or -1000s arg error)")


def _cache_strides(cache, page_size: int, num_kv_heads: int, fn: str):
    """Return (block, token, head) element strides for a paged KV cache.

    Contract layout is the (num_blocks, page, heads, head_size) view that
    backend.Impl._split_cache hands the kernel. But vLLM's raw cache (v0.14+)
    is DECLARED as (blocks, heads, page, 2*head_size) while its PHYSICAL
    memory is LBNHC = [block][page][head][data] — so the transposed view can
    have EITHER stride order, and stride order alone cannot tell the page
    axis from the head axis (engine K_view has s_page=2048 > s_head=256).
    Detect by SHAPE against the known page_size / num_kv_heads; only when
    page_size == num_kv_heads (sizes ambiguous) fall back to stride order.
    """
    if cache.dim() != 4:
        raise ValueError(f"fusionattn.{fn}: cache must be 4D")
    if cache.stride(3) != 1:
        raise ValueError(f"fusionattn.{fn}: last cache dim must be contiguous")
    sb, s1, s2 = cache.stride(0), cache.stride(1), cache.stride(2)
    d1, d2 = cache.size(1), cache.size(2)
    p1, p2 = d1 == page_size, d2 == page_size
    h1, h2 = d1 == num_kv_heads, d2 == num_kv_heads
    if p1 and not h1:
        st, sh = s1, s2          # (B, page, heads, D) contract layout
    elif p2 and not h2:
        st, sh = s2, s1          # (B, heads, page, D): swap token/head
    elif h2 and not p1:
        st, sh = s2, s1          # dim2 is heads (page==heads ambiguous resolved)
    elif h1 and not p2:
        st, sh = s1, s2
    else:
        # page_size == num_kv_heads (or neither matches): fall back to
        # stride order — token axis assumed the inner (smaller-stride) one.
        st, sh = (s1, s2) if s1 <= s2 else (s2, s1)
    if min(s1, s2) * max(d1, d2) > sb:
        raise ValueError(
            f"fusionattn.{fn}: implausible cache strides shape={tuple(cache.shape)} "
            f"strides={cache.stride()} for page_size={page_size}"
        )
    return sb, st, sh


def cache_update(key, value, k_cache, v_cache, slot_mapping, page_size: int) -> None:
    """Write new-token K/V rows into the paged cache.

    key/value: [num_tokens, num_kv_heads, head_size] (contiguous).
    k_cache/v_cache: [B, page, H, hd] (or [B, H, page, hd]; strides are
    detected) with stride(-1) == 1.
    slot_mapping: int64 [num_tokens], negative = skip.
    """
    lib = _cache.get("lib")
    if lib is None:
        if not native_available():
            raise RuntimeError("fusionattn: native library not loaded")
        lib = _cache["lib"]
    assert k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1, "head dim must be contiguous"
    nkv = key.shape[1]
    sb_k, st_k, sh_k = _cache_strides(k_cache, page_size, nkv, "cache_update")
    sb_v, st_v, sh_v = _cache_strides(v_cache, page_size, nkv, "cache_update")
    rc = lib.lib.fusionattn_cache_update(
        key.data_ptr(), value.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        sb_k, st_k, sh_k,
        slot_mapping.data_ptr(),
        key.shape[0], key.shape[1], key.shape[2], page_size,
        _dtype_code(key), _stream_ptr(),
    )
    _check(rc, "cache_update")


def forward(query, out, k_cache, v_cache, block_table, query_start_loc, seq_lens,
            num_reqs: int, num_tokens: int, num_q_heads: int, num_kv_heads: int,
            head_size: int, page_size: int, scale: float, causal: bool,
             dtype=None) -> None:
    """FA2 paged attention over [num_tokens, nq, hd] query into out (same shape)."""
    lib = _cache.get("lib")
    if lib is None:
        if not native_available():
            raise RuntimeError("fusionattn: native library not loaded")
        lib = _cache["lib"]
    assert k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1, "head dim must be contiguous"
    sb_k, st_k, sh_k = _cache_strides(k_cache, page_size, num_kv_heads, "forward")
    rc = lib.lib.fusionattn_forward(
        query.data_ptr(), out.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
        sb_k, st_k, sh_k,
        block_table.data_ptr(), block_table.stride(0),
        query_start_loc.data_ptr(), seq_lens.data_ptr(), num_reqs,
        num_tokens, num_q_heads, num_kv_heads, head_size, page_size,
        float(scale), 1 if causal else 0,
        _dtype_code(query) if dtype is None else _dtype_code(dtype), _stream_ptr(),
    )
    _check(rc, "forward")
