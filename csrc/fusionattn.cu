/* FuseAttn attention kernels: FA2-style paged attention on plain CUDA cores
 * (no Tensor Cores), so it runs on Pascal sm_61 / sm_60 and every newer arch.
 *
 * fp16 / bf16 I/O, fp32 FMA accumulate, warp-per-(token, q-head) work
 * partition, online softmax (running max + sum), exp2f. Paged KV via a
 * block_table row per request; K/V cache layout is fully strided (the
 * per-layer view vLLM hands us may be non-contiguous along B/N/H).
 *
 * ABI is extern "C"; loaded via ctypes from python (no torch headers), so
 * this .so does not care which torch version the host runs.
 *
 * Layouts (element strides, last dim assumed contiguous stride 1):
 *   q/out       : [num_tokens, num_q_heads, head_size] (contiguous)
 *   k/v cache   : logical (B, N, H, hd) with strides (sb, st, sh) + 1
 *   block_table : [num_reqs, max_pages] int32
 *   cu_seqlens_q: [num_reqs+1] int32   seq_lens: [num_reqs] int32
 *   slot_mapping: [num_tokens] int64 (negative = padding, skipped)
 *
 * Causal limit for a query token at q position qp: kv positions [0, qp].
 * qp = seq_len - q_len + (token - cu_q[req])  (new tokens sit after context).
 * head_size must be a multiple of 32 (asserted host-side).
 */
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define FUSIONATTN_ABI_VERSION 1

namespace {

template <typename T> __device__ __forceinline__ float to_f(T v);
template <> __device__ __forceinline__ float to_f<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_f<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

/* which request owns token tok (cu_q ascending, cu_q[0] == 0) */
__device__ __forceinline__ int find_req(const int32_t* __restrict__ cu_q, int num_reqs, int tok) {
    int lo = 0, hi = num_reqs - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (cu_q[mid] <= tok) lo = mid; else hi = mid - 1;
    }
    return lo;
}

/* warp-per-(token, q-head); grid-strides over num_tokens*num_q_heads */
template <typename T, int NREG>
__global__ void fa2_kernel(const T* __restrict__ q, T* __restrict__ out,
                           const T* __restrict__ kcache, const T* __restrict__ vcache,
                           const int64_t sb, const int64_t st, const int64_t sh,
                           const int32_t* __restrict__ block_table, int64_t bt_stride,
                           const int32_t* __restrict__ cu_q,
                           const int32_t* __restrict__ seq_lens, int num_reqs,
                           int num_tokens, int num_q_heads, int num_kv_heads,
                           int head_size, int page_size,
                           float scale_log2e, int causal) {
    const int warp0 = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    const int total = num_tokens * num_q_heads;
    const int group = num_q_heads / num_kv_heads;

    for (int idx = warp0; idx < total; idx += nwarps) {
        const int tok = idx / num_q_heads;
        const int qh = idx - tok * num_q_heads;

        const int r = find_req(cu_q, num_reqs, tok);
        const int q_start = cu_q[r];
        const int q_end = cu_q[r + 1];
        const int seq_len = seq_lens[r];
        const int qp = seq_len - (q_end - q_start) + (tok - q_start);
        int kv_limit = causal ? (qp + 1) : seq_len;
        if (kv_limit > seq_len) kv_limit = seq_len;
        const int kvh = qh / group;

        const T* qrow = q + (int64_t)tok * num_q_heads * head_size + (int64_t)qh * head_size;
        float qreg[NREG], acc[NREG];
#pragma unroll
        for (int i = 0; i < NREG; ++i) {
            const int d = lane + 32 * i;
            qreg[i] = (d < head_size) ? to_f<T>(qrow[d]) : 0.f;
            acc[i] = 0.f;
        }

        float m = -INFINITY, l = 0.f;
        const int pages = (kv_limit + page_size - 1) / page_size;
        const int32_t* bt = block_table + (int64_t)r * bt_stride;
        for (int p = 0; p < pages; ++p) {
            const int64_t blk = bt[p];
            const int base = p * page_size;
            const int cnt = min(page_size, kv_limit - base);
            const T* kp = kcache + blk * sb + (int64_t)kvh * sh;
            const T* vp = vcache + blk * sb + (int64_t)kvh * sh;
            for (int t = 0; t < cnt; ++t) {
                const T* krow = kp + (int64_t)t * st;
                const T* vrow = vp + (int64_t)t * st;
                float dot = 0.f;
#pragma unroll
                for (int i = 0; i < NREG; ++i) {
                    const int d = lane + 32 * i;
                    if (d < head_size) dot += qreg[i] * to_f<T>(krow[d]);
                }
                dot = warp_sum(dot) * scale_log2e;
                const float m_new = fmaxf(m, dot);
                const float alpha = exp2f(m - m_new);
                const float pr = exp2f(dot - m_new);
                l = fmaf(l, alpha, pr);
                m = m_new;
#pragma unroll
                for (int i = 0; i < NREG; ++i) {
                    const int d = lane + 32 * i;
                    acc[i] *= alpha;
                    if (d < head_size) acc[i] = fmaf(pr, to_f<T>(vrow[d]), acc[i]);
                }
            }
        }

        T* orow = out + (int64_t)tok * num_q_heads * head_size + (int64_t)qh * head_size;
        const float inv = (l > 0.f) ? 1.f / l : 0.f;
#pragma unroll
        for (int i = 0; i < NREG; ++i) {
            const int d = lane + 32 * i;
            if (d < head_size) orow[d] = from_f<T>(acc[i] * inv);
        }
    }
}

/* write new-token K,V rows into the paged cache; slot<0 skips */
template <typename T>
__global__ void cache_kernel(const T* __restrict__ key, const T* __restrict__ value,
                             T* __restrict__ kcache, T* __restrict__ vcache,
                             const int64_t sb, const int64_t st, const int64_t sh,
                             const int64_t* __restrict__ slot_mapping,
                             int num_tokens, int num_kv_heads, int head_size,
                             int page_size) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_tokens * num_kv_heads * head_size;
         idx += gridDim.x * blockDim.x) {
        const int d = idx % head_size;
        const int h = (idx / head_size) % num_kv_heads;
        const int tok = idx / (head_size * num_kv_heads);
        const int64_t slot = slot_mapping[tok];
        if (slot < 0) continue;
        const int64_t blk = slot / page_size;
        const int64_t off = slot - blk * page_size;
        const int64_t dst = blk * sb + off * st + (int64_t)h * sh + d;
        const int64_t src = ((int64_t)tok * num_kv_heads + h) * head_size + d;
        kcache[dst] = key[src];
        vcache[dst] = value[src];
    }
}

template <typename T, int NREG>
void launch_fa2(const void* q, void* out, const void* kc, const void* vc,
                int64_t sb, int64_t st, int64_t sh,
                const int32_t* bt, int64_t bts, const int32_t* cu_q,
                const int32_t* sl, int num_reqs, int num_tokens, int nq, int nkv,
                int hd, int page, float scale_log2e, int causal, cudaStream_t s) {
    const int total_units = num_tokens * nq;
    int blocks = (total_units + 7) / 8; /* 8 warps per block */
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    fa2_kernel<T, NREG><<<blocks, 256, 0, s>>>(
        (const T*)q, (T*)out, (const T*)kc, (const T*)vc, sb, st, sh, bt, bts,
        cu_q, sl, num_reqs, num_tokens, nq, nkv, hd, page, scale_log2e, causal);
}

int dispatch_fa2(int dtype, const void* q, void* out, const void* kc, const void* vc,
                 int64_t sb, int64_t st, int64_t sh, const int32_t* bt, int64_t bts,
                 const int32_t* cu_q, const int32_t* sl, int num_reqs, int num_tokens,
                 int nq, int nkv, int hd, int page, float scale_log2e, int causal,
                 cudaStream_t s) {
#define ARGS q, out, kc, vc, sb, st, sh, bt, bts, cu_q, sl, num_reqs, num_tokens, nq, nkv, hd, page, scale_log2e, causal, s
    /* NREG = ceil(hd/32); kernel guards d < head_size, so hd need not be
     * a multiple of 32 (covers Qwen2's 80). hd > 256 rejected. */
    switch ((hd + 31) / 32) {
        case 1: if (dtype == 0) launch_fa2<__half, 1>(ARGS); else launch_fa2<__nv_bfloat16, 1>(ARGS); break;
        case 2: if (dtype == 0) launch_fa2<__half, 2>(ARGS); else launch_fa2<__nv_bfloat16, 2>(ARGS); break;
        case 3: if (dtype == 0) launch_fa2<__half, 3>(ARGS); else launch_fa2<__nv_bfloat16, 3>(ARGS); break;
        case 4: if (dtype == 0) launch_fa2<__half, 4>(ARGS); else launch_fa2<__nv_bfloat16, 4>(ARGS); break;
        case 5: if (dtype == 0) launch_fa2<__half, 5>(ARGS); else launch_fa2<__nv_bfloat16, 5>(ARGS); break;
        case 6: if (dtype == 0) launch_fa2<__half, 6>(ARGS); else launch_fa2<__nv_bfloat16, 6>(ARGS); break;
        case 7: if (dtype == 0) launch_fa2<__half, 7>(ARGS); else launch_fa2<__nv_bfloat16, 7>(ARGS); break;
        case 8: if (dtype == 0) launch_fa2<__half, 8>(ARGS); else launch_fa2<__nv_bfloat16, 8>(ARGS); break;
        default: return -1001; /* unsupported head_size (must be 1..256) */
    }
#undef ARGS
    return (int)cudaGetLastError();
}

} /* namespace */

extern "C" {

int fusionattn_abi(void) { return FUSIONATTN_ABI_VERSION; }

int fusionattn_has_gpu(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return 0;
    return 1;
}

/* dtype: 0 = fp16, 1 = bf16. Returns cudaError_t (0 ok) or negative arg code. */
int fusionattn_cache_update(const void* key, const void* value,
                          void* kcache, void* vcache,
                          int64_t sb, int64_t st, int64_t sh,
                          const int64_t* slot_mapping,
                          int num_tokens, int num_kv_heads, int head_size,
                          int page_size, int dtype, void* stream) {
    if (!key || !value || !kcache || !vcache || !slot_mapping) return -1000;
    if (num_tokens <= 0) return 0;
    if (head_size <= 0 || head_size > 256) return -1001;
    cudaStream_t s = (cudaStream_t)stream;
    const int total = num_tokens * num_kv_heads * head_size;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (dtype == 0)
        cache_kernel<__half><<<blocks, threads, 0, s>>>(
            (const __half*)key, (const __half*)value, (__half*)kcache, (__half*)vcache,
            sb, st, sh, slot_mapping, num_tokens, num_kv_heads, head_size, page_size);
    else
        cache_kernel<__nv_bfloat16><<<blocks, threads, 0, s>>>(
            (const __nv_bfloat16*)key, (const __nv_bfloat16*)value, (__nv_bfloat16*)kcache, (__nv_bfloat16*)vcache,
            sb, st, sh, slot_mapping, num_tokens, num_kv_heads, head_size, page_size);
    return (int)cudaGetLastError();
}

int fusionattn_forward(const void* q, void* out, const void* kcache, const void* vcache,
                     int64_t sb, int64_t st, int64_t sh,
                     const int32_t* block_table, int64_t bt_stride,
                     const int32_t* cu_seqlens_q, const int32_t* seq_lens, int num_reqs,
                     int num_tokens, int num_q_heads, int num_kv_heads,
                     int head_size, int page_size, float scale, int causal,
                     int dtype, void* stream) {
    if (!q || !out || !kcache || !vcache || !block_table || !cu_seqlens_q || !seq_lens)
        return -1000;
    if (num_tokens <= 0) return 0;
    if (num_q_heads <= 0 || num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0)
        return -1002;
    if (head_size <= 0 || head_size > 256) return -1001;  /* NREG dispatch guards d < head_size */
    if (page_size <= 0) return -1003;
    const float scale_log2e = scale * 1.4426950408889634f;
    return dispatch_fa2(dtype, q, out, kcache, vcache, sb, st, sh, block_table,
                        bt_stride, cu_seqlens_q, seq_lens, num_reqs, num_tokens,
                        num_q_heads, num_kv_heads, head_size, page_size,
                        scale_log2e, causal ? 1 : 0, (cudaStream_t)stream);
}

} /* extern "C" */
