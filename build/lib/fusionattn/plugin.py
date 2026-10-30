"""vLLM general-plugin entry: registers the FusionAttention backend.

The user-facing backend name is FUSION_ATTN; the pre-rename name
FUSION_ATTN stays registered as a compatibility alias.

pyproject entry point:  vllm.general_plugins  fusionattn = "fusionattn.plugin:load"
The vLLM plugin loader calls load() early (before AttentionConfig validation
in arg_utils), so ``--attention-backend FUSION_ATTN`` parses afterwards.

Force-select it explicitly (recommended on Pascal):

    VLLM_PLUGINS=fusionattn vllm serve ... --attention-backend FUSION_ATTN

FUSION_ATTN_AUTO=1 additionally injects FUSION_ATTN at the tail of the CUDA
platform's backend priority list, so auto-selection falls back to it when
no tensor-core backend validates on the device (cap < 7.0 injection only).
"""

from __future__ import annotations

import logging
import os

_REGISTERED = False
_logger = logging.getLogger("fusionattn.plugin")

BACKEND_CLASS_PATH = "fusionattn.backend.FusionAttentionBackend"
BACKEND_NAME = "FUSION_ATTN"


def register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if BACKEND_NAME not in AttentionBackendEnum._member_map_:
        AttentionBackendEnum.register(BACKEND_NAME, BACKEND_CLASS_PATH)
    _REGISTERED = True

    # Auto-selection fallback is unconditional: on GPUs where every
    # built-in backend fails validation, an unqualified request would
    # otherwise hard-fail; FUSION_ATTN only joins the tail (last resort).
    # FUSION_ATTN_AUTO=0 opts out.
    auto = os.environ.get("FUSION_ATTN_AUTO", "1")
    if auto == "1":
        _install_auto_fallback()


def _install_auto_fallback() -> None:
    """Append FUSION_ATTN to the CUDA auto-selection priority list so GPUs
    below sm_70 (where every built-in backend fails validation) still find
    a working backend without an explicit --attention-backend flag."""
    try:
        import vllm.platforms.cuda as cuda_mod
    except Exception:  # pragma: no cover - non-CUDA install
        return
    orig = getattr(cuda_mod, "_get_backend_priorities", None)
    if orig is None or getattr(orig, "_fusionattn_patched", False):
        return

    def patched(*args, **kwargs):
        prios = list(orig(*args, **kwargs))
        try:
            from vllm.v1.attention.backends.registry import AttentionBackendEnum

            if AttentionBackendEnum.FUSION_ATTN not in prios:
                prios.append(AttentionBackendEnum.FUSION_ATTN)
        except Exception:
            pass
        return prios

    patched._fusionattn_patched = True  # type: ignore[attr-defined]
    cuda_mod._get_backend_priorities = patched
    _logger.info("fusionattn: auto-fallback priority injection installed")


def load() -> None:
    register()
