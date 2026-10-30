# FusionAttention

CUDA-core FA2 paged attention backend for vLLM v1 — runs on Pascal sm_60/61 where
Tensor-Core backends (FA2/FlashInfer/TRT-LLM sm_75+) and Triton (sm_70+) do not.

Build: `python setup.py build_py` (plain nvcc, no torch headers), then install
this directory; plugin auto-registers `--attention-backend FUSION_ATTN` via the
`vllm.general_plugins` entry point. Design + tests: see docs (WIP).
