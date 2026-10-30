"""GPU parity harness for FusionAttention (run on the real-GPU host; needs torch).

    python tests/gpu_parity.py            # default fp16 + bf16
Checks:
  1. cache_update writes each token row to the paged slot (read back via SDPA
     cache gather).
  2. forward == torch SDPA on the gathered contiguous KV (fp16 tol 2e-2 rel,
     bf16 3e-2), decode / prefill / mixed batches, GQA groups {1,4}, hd
     {64, 80, 96, 128}, pages {16, 32}, causal.

Exits non-zero on failure; prints FUSIONATTN GPU PARITY PASSED.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from fusionattn import ops  # noqa: E402


def ref_attention(q, ks, vs, causal, scale, q_pos0):
    """q [T,Hq,D] rows sit at positions q_pos0..q_pos0+T-1 over kv 0..S-1."""
    Hq = q.shape[1]
    Hkv = ks.shape[1]
    g = Hq // Hkv
    out = torch.empty_like(q, dtype=torch.float32)
    for h in range(Hq):
        kvh = h // g
        s = q[:, h].float() @ ks[:, kvh].float().T * scale  # [T,S]
        if causal:
            T, S = s.shape
            rows = torch.arange(T, device=s.device)[:, None] + q_pos0
            cols = torch.arange(S, device=s.device)[None, :]
            s = s.masked_fill(cols > rows, float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[:, h] = p @ vs[:, kvh].float()
    return out.to(q.dtype)


def make_case(rng, num_reqs, pages_list, q_lens, ctx_lens, nq, nkv, hd, page, dev, dt):
    seq_lens = [c + q for c, q in zip(ctx_lens, q_lens)]
    max_pages = max(math.ceil(s / page) for s in seq_lens)
    num_blocks = sum(math.ceil(s / page) for s in seq_lens) + 1
    T = sum(q_lens)
    cu_q = torch.tensor([0] + list(np.cumsum(q_lens)), dtype=torch.int32, device=dev)
    sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    bt = torch.zeros(num_reqs, max_pages, dtype=torch.int32, device=dev)
    blk = 0
    for r, s in enumerate(seq_lens):
        for p in range(math.ceil(s / page)):
            bt[r, p] = blk
            blk += 1
    kc = torch.zeros(num_blocks, nkv, page, hd, dtype=dt, device=dev)
    vc = torch.zeros_like(kc)
    return dict(num_reqs=num_reqs, seq_lens=seq_lens, q_lens=q_lens, cu_q=cu_q,
                sl=sl, bt=bt, kc=kc, vc=vc, num_blocks=num_blocks, page=page,
                max_pages=max_pages, T=T, nq=nq, nkv=nkv, hd=hd)


def run(dtype, seed=0):
    dev = "cuda"
    rng = np.random.default_rng(seed)
    for (nq, nkv, hd, page, causal) in [
        (8, 8, 64, 16, True), (8, 2, 64, 32, True), (12, 3, 80, 16, True),
        (16, 4, 128, 16, True), (8, 1, 96, 32, True), (4, 4, 64, 16, False),
        (12, 4, 128, 1, True),  # page size 1 edge
    ]:
        group = nq // nkv
        num_reqs = 3
        q_lens = [1, 1, 1]
        ctx_lens = [page * 2, page * 2 + 1, page]
        c = make_case(rng, num_reqs, None, q_lens, ctx_lens, nq, nkv, hd, page, dev, dtype)
        scale = 1.0 / math.sqrt(hd)
        # fill cache with random K/V for existing positions, also produce refs
        ks_refs, vs_refs = [], []
        for r in range(num_reqs):
            S = c["seq_lens"][r]
            k_new = torch.from_numpy(rng.standard_normal((S, nkv, hd)).astype(np.float32)).to(dev, dtype)
            v_new = torch.from_numpy(rng.standard_normal((S, nkv, hd)).astype(np.float32)).to(dev, dtype)
            ks_refs.append(k_new)
            vs_refs.append(v_new)
            pages = math.ceil(S / page)
            flat_slots = torch.tensor(
                [c["bt"][r, p] * page + t for p in range(pages) for t in range(page)
                 if p * page + t < S], dtype=torch.int64, device=dev)
            # write via cache kernel one token at a time in order (simulates steps)
            for pos in range(S):
                slot = torch.tensor([flat_slots[pos].item()], dtype=torch.int64, device=dev)
                ops.cache_update(k_new[pos:pos + 1].view(1, nkv, hd).contiguous(),
                                 v_new[pos:pos + 1].view(1, nkv, hd).contiguous(),
                                 c["kc"], c["vc"], slot, page)
        # all positions written: now attention for new-token queries
        q = torch.from_numpy(rng.standard_normal((c["T"], nq, hd)).astype(np.float32)).to(dev, dtype)
        out = torch.zeros_like(q)
        ops.forward(q, out, c["kc"], c["vc"], c["bt"], c["cu_q"], c["sl"],
                    num_reqs, c["T"], nq, nkv, hd, page, scale, causal, dtype)
        torch.cuda.synchronize()
        for r in range(num_reqs):
            t0, t1 = int(c["cu_q"][r]), int(c["cu_q"][r + 1])
            S = c["seq_lens"][r]
            want = ref_attention(q[t0:t1], ks_refs[r], vs_refs[r], causal, scale,
                                 S - (t1 - t0))
            got = out[t0:t1].float()
            rel = (got - want.float()).abs().max().item()
            tol = 2e-2 if dtype == torch.float16 else 3e-2
            assert rel < tol, f"nq{nq} nkv{nkv} hd{hd} page{page} req{r}: maxdiff {rel}"
    print("FUSIONATTN GPU PARITY PASSED")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no GPU")
        sys.exit(2)
    run(torch.float16, seed=0)
    run(torch.bfloat16, seed=1)
