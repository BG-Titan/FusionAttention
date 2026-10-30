"""FusionAttention backend: FA2 paged attention on CUDA cores for vLLM v1.

Backend contract per vllm/v1/attention/backend.py (this tree):
- forward_includes_kv_cache_update = False  -> do_kv_cache_update is called
  by unified_kv_cache_update with the per-layer slot_mapping.
- kv_cache arrives as a per-layer strided view shaped
  [num_blocks, num_kv_heads, kernel_block_size, 2 * head_size]; we split it
  transpose(1,2)-style into K/V logical (B, N, H, hd) views and hand the
  element strides to the kernel (never force .contiguous()).
- cudagraph support: NEVER -> run with --enforce-eager (documented).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from fusionattn import ops

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention import Attention


@dataclass
class FusionAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool = True


class FusionAttentionMetadataBuilder(AttentionMetadataBuilder[FusionAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam: CommonAttentionMetadata = common_attn_metadata
        causal = cam.causal
        if not isinstance(causal, bool):
            # per-row causal tensors come from adaptive/non-causal features
            # this v0 kernel does not implement; all supported configs are
            # uniformly causal.
            if not bool(causal.all()):
                raise NotImplementedError(
                    "fusionattn: per-row causal masks are not supported; "
                    "use a fully-causal configuration."
                )
            causal = True
        return FusionAttentionMetadata(
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            query_start_loc=cam.query_start_loc,
            max_seq_len=cam.max_seq_len,
            seq_lens=cam.seq_lens,
            block_table=cam.block_table_tensor,
            slot_mapping=cam.slot_mapping,
            causal=causal,
        )


class FusionAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "FUSION_ATTN"

    @staticmethod
    def get_impl_cls():
        return FusionAttentionImpl

    @staticmethod
    def get_metadata_cls():
        return FusionAttentionMetadata

    @staticmethod
    def get_builder_cls():
        return FusionAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, **kwargs):
        # classic NHD flash layout: [num_blocks, num_kv_heads, block_size, 2*hd]
        return [num_blocks, num_kv_heads, block_size, head_size * 2]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # kernel handles any head_size <= 256 (per-lane strided, %32 not
        # required); advertise the common ones, empty would mean "any".
        return []

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        # The entire point: CUDA-core math, works on sm_60/61 and newer.
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    forward_includes_kv_cache_update: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    @classmethod
    def validate_configuration(cls, **kwargs) -> list[str]:
        reasons = super().validate_configuration(**kwargs)
        if kwargs.get("head_size", 0) > 256:
            reasons.append("fusionattn supports head_size <= 256")
        if kwargs.get("use_per_head_quant_scales", False):
            reasons.append("fusionattn does not support per-head quant scales")
        return reasons


class FusionAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: torch.Tensor | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type=None,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        if alibi_slopes is not None:
            raise NotImplementedError("fusionattn: alibi is not supported")
        if logits_soft_cap is not None:
            raise NotImplementedError("fusionattn: logits_soft_cap is not supported")
        if sliding_window is not None:
            raise NotImplementedError("fusionattn: sliding window is not supported (v0)")
        if kv_cache_dtype not in ("auto", "fp16", "half", "float16", "bf16", "bfloat16"):
            raise NotImplementedError(f"fusionattn: kv_cache_dtype {kv_cache_dtype}")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype

    # -- helpers ---------------------------------------------------------
    def _split_cache(self, kv_cache, device_block_size):
        """[B, H, N, 2*hd] -> K,V logical [B, N, H, hd] strided views."""
        k_all = kv_cache.transpose(1, 2)  # [B, N, H, 2*hd]
        k, v = torch.split(k_all, self.head_size, dim=-1)
        return k, v

    # -- KV write (unified_kv_cache_update path) --------------------------
    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if slot_mapping is None or slot_mapping.numel() == 0:
            return
        n = slot_mapping.shape[0]
        if n == 0:
            return
        k_view, v_view = self._split_cache(kv_cache, None)
        k4 = key[:n].view(n, self.num_kv_heads, self.head_size)
        v4 = value[:n].view(n, self.num_kv_heads, self.head_size)
        page_size = kv_cache.shape[2]
        ops.cache_update(
            k4.contiguous() if not k4.is_contiguous() else k4,
            v4.contiguous() if not v4.is_contiguous() else v4,
            k_view, v_view, slot_mapping, page_size,
        )

    # -- attention --------------------------------------------------------
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        num_actual_tokens = attn_metadata.num_actual_tokens if attn_metadata else 0
        if attn_metadata is None:
            # profiling pass: no real work
            output.fill_(0)
            return output

        q = query[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, self.head_size
        )
        out = output[:num_actual_tokens].view_as(q)
        k_view, v_view = self._split_cache(kv_cache, None)
        page_size = kv_cache.shape[2]
        # int32 metadata as produced by the core; kernel requires it
        ops.forward(
            q.contiguous() if not q.is_contiguous() else q,
            out.contiguous() if not out.is_contiguous() else out,
            k_view, v_view,
            attn_metadata.block_table,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            attn_metadata.block_table.shape[0],
            num_actual_tokens,
            self.num_heads,
            self.num_kv_heads,
            self.head_size,
            page_size,
            self.scale,
            attn_metadata.causal,
        )
        return output

# Pre-rename compatibility aliases (brand: FusionAttention -> FusionAttention).
FusionAttentionBackend = FusionAttentionBackend
FusionAttentionImpl = FusionAttentionImpl
FusionAttentionMetadata = FusionAttentionMetadata
FusionAttentionMetadataBuilder = FusionAttentionMetadataBuilder
