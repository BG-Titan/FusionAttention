# FusionAttention 设计文档

> 命名说明：对外组件名已迁移为 FusionAttention/FusionKV（稠密）与 FissionAttention/FissionKV（MoE），旧名 FUSION_ATTN/FUSE_MLA/FusionKV/FusionKV 保留为兼容别名；详见 docs/fusion-fission-naming.md。

目标：让 vLLM v1 在 **2× Pascal 8G (sm_61)** 上拥有真正可用的注意力
后端。vLLM 自带路径全部依赖 Tensor Core（FA2/FlashInfer/TRT-LLM 要
sm_75+）或 Triton（最低 sm_70），sm_61 上无一可用；FLEX 的 math 回退
只在 enforce-eager 下存在且慢到不可用。

## 1. 形态：外置插件 + torch 无关的内核 .so

与 FuseW4/FusionKV 同构：

- `csrc/fusionattn.cu` 纯 CUDA（extern "C" ABI，abi 版本号=1），`nvcc`
  直接编成 `_fusionattn*.so`，**不 include 任何 torch 头**。python 侧
  `fusionattn/ops.py` 用 ctypes 传 `data_ptr()` + `stream.cuda_stream`。
  → 主机上 torch 无论怎么降级/源码构建，内核二进制不变。
- `fusionattn/backend.py` 实现 vLLM v1 后端契约（对齐
  `vllm/v1/attention/backend.py`，以 triton_attn.py 为模板）。
- `fusionattn/plugin.py` 经 `vllm.general_plugins` 入口注册
  `AttentionBackendEnum.register("FUSION_ATTN", …)`，此后
  `--attention-backend FUSION_ATTN` 可解析（general plugins 在
  arg_utils 配置校验之前加载）。

## 2. 内核：CUDA-core FA2（v0 标量版）

- 任务划分：warp-per-(query token, q-head)；grid-stride 覆盖
  `num_tokens × num_q_heads`；block=256 线程（8 warps）。
- head_size：per-lane 寄存器 qreg/acc = `ceil(hd/32)` 槽位（模板
  NREG=1..8），`d < head_size` 谓词 → 支持 80 这类非 32 倍数尺寸
  （Qwen2），上限 256。
- 数据通路：fp16/bf16 进，fp32 FMA 累加（消费级 Pascal fp16 半速
  1/64，Tensor Core 没有 → CUDA-core fp32 是最快正确路径），
  dot 用 warp shuffle 归约，softmax 在 exp2 域在线更新
  （scale·log2e 折叠），输出 cast 回输入 dtype。
- 分页：`block_table[req, p]` → 页基址；K/V 视图
  `(B, N, H, hd)` 三个 int64 元素 stride 原样传入（vLLM 的 per-layer
  视图可能跨层交错，绝不 `.contiguous()`），最后一维 stride 必须为 1。
- causal 边界：`qp = seq_len - q_len + (tok - cu_q[req])`，
  `limit = min(qp+1, seq_len)`；varlen 批（decode+prefill 混排）由
  `cu_seqlens_q` 二分定位请求。
- KV 写入：`fusionattn_cache_update`，slot_mapping int64，负数跳过
  （padding），`blk = slot / page`，`off = slot % page`，三维 stride
  同前。由 `forward_includes_kv_cache_update=False` 路径
  （`do_kv_cache_update`）驱动，与 vLLM 的 unified_kv_cache_update
  自定义算子对齐。

## 3. 正确性三层

1. `tests/test_math_ref.py`（CPU，无 torch）：numpy 复刻内核逐
   warp 算法 vs 朴素 softmax 注意力，覆盖 decode/prefill/混排/
   GQA/page=1/页边界/非因果。
2. `tests/gpu_parity.py`（真卡）：cache_update 逐 token 写入 +
   forward 对 fp16/bf16 × hd{64,80,96,128} × group{1,4} ×
   page{1,16,32} × causal{T,F} 与 fp32 参考对比（容差 2e-2/3e-2）。
3. 主机 e2e：gpu_smoke.sh（FuseW4 权重 + FusionKV KV 分层组合）。

## 4. vLLM 契约细节（本树实测）

- kv_cache 张量：`[num_blocks, num_kv_heads, kernel_block_size,
  2·head_size]`（dtype 已按模型半精度 cast，字节尾布局对后端透明）；
  `transpose(1,2).split(hd,-1)` 得 K/V 视图（照抄 triton 模板）。
- `AttentionCGSupport.NEVER`：必须 `--enforce-eager`。
- `validate_configuration` 追加拒绝 head_size>256 与 per-head
  quant scales；滑窗/ sinks /非因果由基类默认 False 拒绝。
- 已知限制（v0）：alibi / logits_soft_cap / sliding window / fp8
  KV / MLA / encoder 均明确 raise，不做静默错误。

## 5. 性能路线（roadmap）

v0 每 token 每头一个 warp，纯标量：正确性优先。Pascal 带宽 ~
192GB/s（GTX1080 352）而 fp32 FMA 算力有限，长上下文 prefill 是
compute-bound。计划：shared-mem q tile + kv tile（FA2 论文分块），
`__half2` CUDA-core 半精度点积（GP10x 1/64 率仍优于标量 fp32 读
放大），以及 decode 侧 warp-per-page 并行化。测量先行（ncu 不可用
于 WSL 时退 nsys + 手计时）。

## 6. torch 降级调查结论（写死在这里）

`torch 2.9.1+cu126` 官方轮 arch 面 = `sm_50/60/70/75/80/86/90`
SASS、**无 PTX**（`cuobjdump -lptx` 空）。sm_60 SASS 不能在 sm_61
上执行（Volta 前无 minor 兼容）→ 官方轮在 GTX 10 系必挂
`no kernel image`。**torch 必须源码构建**
`TORCH_CUDA_ARCH_LIST="6.1;…"`（见 fusew4/docs/gpu-host-runbook.md
§2）。FusionAttention 因不依赖 torch 头，是这条降级路上唯一零成本环节。
