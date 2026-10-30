# Regression lock: engine LBNHC layout (B,H,page,2hd) -> transposed K/V views
# must be addressed with token stride = page-major, not swapped.
import ctypes, math, os, sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")
pytest.importorskip("numpy")
from fusionattn import ops  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available() or not ops.native_available(),
                    reason="needs CUDA + _fusionattn")
@pytest.mark.parametrize("page,nkv", [(16, 8), (32, 4)])
def test_engine_lbnhc_write_read_roundtrip(page, nkv):
    hd, nblk, nq = 64, 4, 2 * nkv
    hd0 = hd
    # Engine physical memory is [blk][page][head][K|V] but vLLM DECLARES the
    # cache tensor as dims (B, H, page, C) with strides (2hd, page*2hd, 1)
    # (LBNHC) — reproduce that declared/physical mismatch exactly.
    phys = torch.zeros(nblk, page, nkv, 2 * hd, dtype=torch.float16,
                       device="cuda")
    declared = phys.transpose(1, 2)               # (B, nkv, page, 2hd)
    # engine-style split (backend.py _split_cache): transpose(1,2) + last split
    kv = declared.transpose(1, 2)                 # (B, page, nkv, 2hd) s1>s2
    k_cache, v_cache = kv.split(hd, dim=-1)       # (B, page, nkv, hd) views
    assert k_cache.stride(1) > k_cache.stride(2), "test must mirror engine strides"
    T = page + 2
    k = torch.randn(T, nkv, hd).half().cuda()
    v = torch.randn(T, nkv, hd).half().cuda()
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    ops.cache_update(k, v, k_cache, v_cache, slots, page)
    got_k = k_cache.reshape(-1, nkv, hd)[:T]
    got_v = v_cache.reshape(-1, nkv, hd)[:T]
    assert torch.equal(got_k, k), "K written to wrong positions (stride swap?)"
    assert torch.equal(got_v, v), "V written to wrong positions"

    q = torch.randn(T, nq, hd).half().cuda()
    out = torch.empty_like(q)
    bt = torch.arange(nblk, dtype=torch.int32, device="cuda").view(1, nblk)
    qsl = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    seq = torch.tensor([T], dtype=torch.int32, device="cuda")
    ops.forward(q, out, k_cache, v_cache, bt, qsl, seq, 1, T, nq, nkv, hd,
                page, 1.0 / math.sqrt(hd), True)
    # fp32 reference on the SAME written cache (dense)
    kd = k.float().transpose(0, 1)                # (nkv,T,hd)
    vd = v.float().transpose(0, 1)
    scale = 1.0 / math.sqrt(hd)
    ref = torch.zeros(T, nq, hd, dtype=torch.float32, device="cuda")
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device="cuda"))
    for h in range(nq):
        kvh = h * nkv // nq
        s = (q.float()[:, h] @ kd[kvh].T) * scale
        s = s.masked_fill(~mask, float("-inf"))
        ref[:, h] = torch.softmax(s, -1) @ vd[kvh]
    d = (out.float() - ref).abs().max()
    assert float(d) < 2e-2, f"engine-layout forward maxdiff {float(d)}"
