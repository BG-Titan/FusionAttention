"""FusionAttention: CUDA-core FA2 paged attention for vLLM — runs on Pascal sm_61.

Tensor-core-free attention (fp16/bf16 I/O, fp32 FMA) so vLLM has a real
fused attention path on GPUs where FA2/FlashInfer/TRT-LLM (sm_75+) and even
Triton (sm_70+) cannot run. Loads a plain nvcc-built .so via ctypes, so the
kernel binary is independent of the torch version installed.

Register with vLLM through the ``vllm.general_plugins`` entry point (see
fusionattn.plugin), then ``--attention-backend FUSION_ATTN``.
"""

import logging
import os

FUSIONATTN_BACKEND_NAME = "FUSION_ATTN"
__version__ = "0.1.0"

_level = (os.environ.get("FUSIONATTN_LOG") or "INFO").upper()
logging.getLogger("fusionattn").addHandler(logging.NullHandler())
if _level == "DEBUG":
    import sys

    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("fusionattn %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("fusionattn").addHandler(_h)
    logging.getLogger("fusionattn").setLevel(logging.DEBUG)
