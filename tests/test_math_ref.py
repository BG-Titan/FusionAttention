"""CPU math-equivalence test for the FusionAttention kernel algorithm.

Re-implements the exact per-warp algorithm of csrc/fusionattn.cu in numpy
(page walk, causal bounds, online softmax in exp2 domain) and checks it
against a naive float64 attention reference. This guards the ALGORITHM
(kernel execution itself is covered by tests/gpu_parity.py on a real
GPU host).
"""

import math

import numpy as np


def naive_attention(q, ktok, vtok, scale, causal):
    # q (Tq,Hq,hd), ktok/vtok (Hkv,Tk,hd) full dense, GQA by h % Hkv
    Tq, Hq, hd = q.shape
    Hkv, Tk, _ = ktok.shape
    out = np.zeros_like(q)
    for i in range(Tq):
        for h in range(Hq):
            kv = h * Hkv // Hq  # same GQA repeat mapping as the kernel
            s = (ktok[kv] @ q[i, h]) * scale
            if causal:
                # token i sits at KV position (Tk - Tq + i)
                lim = Tk - Tq + i + 1
                s = s[:lim]
                v = vtok[kv, :lim]
            else:
                v = vtok[kv]
            s = s - s.max()
            p = np.exp(s)
            out[i, h] = (p[:, None] * v).sum(0) / p.sum()
    return out


def run_case(nq, nkv, hd, page, lens, q_lens, causal, seed=0):
    rng = np.random.default_rng(seed)
    R = len(lens)
    T = sum(q_lens)
    cu = np.concatenate([[0], np.cumsum(q_lens)])
    pages_per = [math.ceil(L / page) for L in lens]
    nblk = sum(pages_per) + 1
    B = nblk + 2
    kcache = rng.standard_normal((B, page, nkv, hd)).astype(np.float64)
    vcache = rng.standard_normal((B, page, nkv, hd)).astype(np.float64)
    max_pages = max(pages_per)
    block_table = -np.ones((R, max_pages), dtype=np.int64)
    base = 1
    # gather dense per-request tokens from cache for the reference
    k_dense = []  # per req: (nkv, T, hd)
    v_dense = []
    for r in range(R):
        ids = np.arange(base, base + pages_per[r])
        block_table[r, :pages_per[r]] = ids
        base += pages_per[r]
        ks = np.stack([kcache[ids[t // page], t % page] for t in range(lens[r])])
        vs = np.stack([vcache[ids[t // page], t % page] for t in range(lens[r])])
        k_dense.append(np.transpose(ks, (1, 0, 2)))   # (nkv, T, hd)
        v_dense.append(np.transpose(vs, (1, 0, 2)))
    q = rng.standard_normal((T, nq, hd)).astype(np.float64)
    scale = 1.0 / math.sqrt(hd)

    got = emu_kernel_real(q, kcache, vcache, block_table, cu, np.array(lens),
                          scale, causal, page, nkv)
    for r in range(R):
        i0, i1 = cu[r], cu[r + 1]
        want = naive_attention(q[i0:i1], k_dense[r], v_dense[r], scale, causal)
        err = np.abs(got[i0:i1] - want).max()
        denom = np.abs(want).max()
        assert err <= 2e-9 * max(denom, 1.0), (nq, nkv, hd, page, causal, err)


def emu_kernel_real(q, kcache, vcache, block_table, cu, seq_lens, scale,
                    causal, page, nkv):
    """Vectorized clone of the kernel inner loop (same order of ops)."""
    T, nq, hd = q.shape
    out = np.zeros_like(q)
    log2e = 1.4426950408889634
    for tok in range(T):
        r = int(np.searchsorted(cu, tok, side="right") - 1)
        q_start, q_end = cu[r], cu[r + 1]
        seq_len = int(seq_lens[r])
        q_len = q_end - q_start
        qp = seq_len - q_len + (tok - q_start)
        kv_limit = min(qp + 1, seq_len) if causal else seq_len
        for h in range(nq):
            kvh = h * nkv // nq
            qv = q[tok, h]
            m_i = -np.inf
            l_i = 0.0
            acc = np.zeros(hd)
            p = 0
            while p * page < kv_limit:
                blk = int(block_table[r, p])
                t_hi = min(page, kv_limit - p * page)
                for t in range(t_hi):
                    s = (qv @ kcache[blk, t, kvh]) * scale * log2e
                    if s > m_i:
                        alpha = 2.0 ** (m_i - s) if np.isfinite(m_i) else 0.0
                        m_i = s
                        l_i = l_i * alpha + 1.0
                        acc = acc * alpha + vcache[blk, t, kvh]
                    else:
                        p_ = 2.0 ** (s - m_i)
                        l_i += p_
                        acc += p_ * vcache[blk, t, kvh]
                p += 1
            out[tok, h] = acc / l_i if l_i > 0 else 0.0
    return out


def test_decode_batch():
    # 4 sequences decoding 1 token each
    run_case(8, 8, 64, 16, lens=[17, 64, 33, 128], q_lens=[1, 1, 1, 1],
             causal=True, seed=1)


def test_prefill_chunk():
    # one request prefilling 32 tokens on top of a 100-token prefix
    run_case(4, 2, 32, 8, lens=[132], q_lens=[32], causal=True, seed=2)


def test_mixed_lens_page3():
    run_case(6, 3, 16, 3, lens=[5, 19, 27], q_lens=[2, 1, 4],
             causal=True, seed=3)


def test_non_causal():
    run_case(4, 4, 32, 4, lens=[9, 13], q_lens=[3, 5], causal=False, seed=4)


def test_page_boundary_seq():
    run_case(2, 1, 8, 16, lens=[16, 17], q_lens=[1, 1], causal=True, seed=5)


if __name__ == "__main__":
    for fn in sorted(k for k in dict(globals()) if k.startswith("test_")):
        globals()[fn]()
        print(f"{fn}: OK")
