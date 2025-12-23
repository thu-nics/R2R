import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
# import sglang

# if os.environ["SGLANG_ENABLE_TORCH_COMPILE"] == "1":
#     import logging

#     torch._logging.set_logs(dynamo=logging.ERROR)
#     torch._dynamo.config.suppress_errors = True

from sglang.global_config import global_config
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.attention import flashinfer_backend
from sglang.srt.layers.attention.flashinfer_backend import (
    should_use_tensor_core,
    FlashInferAttnBackend,
    WrapperDispatch,
    FlashInferIndicesUpdaterPrefill,
    FlashInferIndicesUpdaterDecode,
    PrefillMetadata,
    DecodeMetadata,
    fast_decode_plan,
)
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.mem_cache.allocator import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_flashinfer_available, next_power_of_2


from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.model_runner import ModelRunner
if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )
    from flashinfer.cascade import merge_state
    from flashinfer.decode import _get_range_buf, get_seq_lens


@dataclass
class ExtendMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    prefill_wrappers_ragged: List[BatchPrefillWithRaggedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool


def __init__(
    self: FlashInferAttnBackend,
    model_runner: ModelRunner,
    skip_prefill: bool = False,
    kv_indptr_buf: Optional[torch.Tensor] = None,
    kv_last_page_len_buf: Optional[torch.Tensor] = None,
):
    AttentionBackend.__init__(self)

    # Parse constants
    self.decode_use_tensor_cores = should_use_tensor_core(
        kv_cache_dtype=model_runner.kv_cache_dtype,
        num_attention_heads=model_runner.model_config.num_attention_heads // get_attention_tp_size(),
        num_kv_heads=model_runner.model_config.get_num_kv_heads(get_attention_tp_size()),
    )
    self.max_context_len = model_runner.model_config.context_len
    self.skip_prefill = skip_prefill
    self.is_multimodal = model_runner.model_config.is_multimodal

    assert not (model_runner.sliding_window_size is not None and model_runner.model_config.is_encoder_decoder), "Sliding window and cross attention are not supported together"

    if model_runner.sliding_window_size is not None:
        self.num_wrappers = 2
        self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
    elif model_runner.model_config.is_encoder_decoder:
        self.num_wrappers = 2
        self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
    else:
        self.num_wrappers = 1
        self.dispatch_reason = None

    # Qwen2/Qwen3 models require higher flashinfer workspace size
    if (
        "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
        or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
        or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
    ):
        global_config.flashinfer_workspace_size = 512 * 1024 * 1024

    # Allocate buffers
    # global global_workspace_buffer
    if flashinfer_backend.global_workspace_buffer is None:
        # different from flashinfer zero_init_global_workspace_buffer
        flashinfer_backend.global_workspace_buffer = torch.empty(
            global_config.flashinfer_workspace_size,
            dtype=torch.uint8,
            device=model_runner.device,
        )
    self.workspace_buffer = flashinfer_backend.global_workspace_buffer
    max_bs = model_runner.req_to_token_pool.size
    if kv_indptr_buf is None:
        self.kv_indptr = [torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device) for _ in range(self.num_wrappers)]
    else:
        assert self.num_wrappers == 1
        self.kv_indptr = [kv_indptr_buf]

    if kv_last_page_len_buf is None:
        self.kv_last_page_len = torch.ones((max_bs,), dtype=torch.int32, device=model_runner.device)
    else:
        assert self.num_wrappers == 1
        self.kv_last_page_len = kv_last_page_len_buf

    if not self.skip_prefill:
        self.qo_indptr = [torch.zeros((max_bs + 1,), dtype=torch.int32, device=model_runner.device) for _ in range(self.num_wrappers)]

    fmha_backend = "auto"
    if is_sm100_supported():
        fmha_backend = "cutlass"
    self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(self.workspace_buffer, "NHD", backend=fmha_backend)

    # Two wrappers: one for sliding window attention and one for full attention.
    # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
    self.prefill_wrappers_paged = []
    self.prefill_wrappers_verify = []
    self.decode_wrappers = []
    for _ in range(self.num_wrappers):
        if not skip_prefill:
            self.prefill_wrappers_paged.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend="fa2",
                )
            )
            self.prefill_wrappers_verify.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                )
            )
        self.decode_wrappers.append(
            BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                "NHD",
                use_tensor_cores=self.decode_use_tensor_cores,
            )
        )

    # Create indices updater
    if not skip_prefill:
        self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(model_runner, self)  # for verify
        self.indices_updater_extend = FlashInferIndicesUpdaterExtend(model_runner, self)
    self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

    # Other metadata
    self.forward_metadata = None
    self.decode_cuda_graph_metadata = {}
    self.prefill_cuda_graph_metadata = {}  # For verify
    self.draft_extend_cuda_graph_metadata = {}  # For draft extend

def init_forward_metadata_capture_cuda_graph(
    self: FlashInferAttnBackend,
    bs: int,
    num_tokens: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    encoder_lens: Optional[torch.Tensor],
    forward_mode: ForwardMode,
    spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    prefix_lens: Optional[torch.Tensor] = None,
    extend_seq_len: Optional[int] = None,
):
    if forward_mode.is_decode_or_idle():
        decode_wrappers = []
        for i in range(self.num_wrappers):
            decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=self.decode_use_tensor_cores,
                    paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                    paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
                )
            )
        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            seq_lens.cpu(),  # may add a little overhead in capture stage
            seq_lens_sum,
            decode_wrappers=decode_wrappers,
            encoder_lens=encoder_lens,
            spec_info=spec_info,
        )
        self.decode_cuda_graph_metadata[bs] = decode_wrappers
        self.forward_metadata = DecodeMetadata(decode_wrappers)
        for i in range(self.num_wrappers):
            decode_wrappers[i].begin_forward = partial(fast_decode_plan, decode_wrappers[i])
    elif forward_mode.is_target_verify():
        prefill_wrappers = []
        for i in range(self.num_wrappers):
            prefill_wrappers.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    custom_mask_buf=self.cuda_graph_custom_mask,
                    mask_indptr_buf=self.cuda_graph_qk_indptr[i][: bs + 1],
                )
            )
        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_prefill.update(
            req_pool_indices,
            seq_lens,
            seq_lens.cpu(),  # may add a little overhead in capture stage
            seq_lens_sum,
            prefix_lens=None,
            prefill_wrappers=prefill_wrappers,
            use_ragged=False,
            encoder_lens=encoder_lens,
            spec_info=spec_info,
        )
        self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
        self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
    elif forward_mode.is_draft_extend():
        prefill_wrappers = []
        for i in range(self.num_wrappers):
            prefill_wrappers.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    backend="fa2",
                    use_cuda_graph=True,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                )
            )

        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_prefill.update(
            req_pool_indices,
            seq_lens,
            seq_lens.cpu(),  # may add a little overhead in capture stage
            seq_lens_sum,
            prefix_lens=None,
            prefill_wrappers=prefill_wrappers,
            use_ragged=False,
            encoder_lens=encoder_lens,
            spec_info=spec_info,
        )
        self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
        self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
    elif forward_mode.is_extend():
        prefill_wrappers = []
        prefill_wrappers_ragged = []
        for i in range(self.num_wrappers):
            prefill_wrappers.append(
                BatchPrefillWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                    paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                    custom_mask_buf=None,  # self.cuda_graph_custom_mask,
                    mask_indptr_buf=None,  # self.cuda_graph_qk_indptr[i][: bs + 1],
                    backend="fa2",
                )
            )
            prefill_wrappers_ragged.append(
                BatchPrefillWithRaggedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    kv_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                    custom_mask_buf=None,
                    mask_indptr_buf=None,
                )
            )

        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_extend.update(
            req_pool_indices,
            seq_lens,
            seq_lens.cpu(),  # may add a little overhead in capture stage
            seq_lens_sum,
            prefix_lens=prefix_lens,
            prefill_wrappers=prefill_wrappers,
            prefill_wrappers_ragged=prefill_wrappers_ragged,
            use_ragged=True,
            encoder_lens=encoder_lens,
            spec_info=spec_info,
        )

        if bs not in self.prefill_cuda_graph_metadata:
            self.prefill_cuda_graph_metadata[bs] = {}
        self.prefill_cuda_graph_metadata[bs][extend_seq_len] = (prefill_wrappers, prefill_wrappers_ragged)
        self.forward_metadata = ExtendMetadata(
            prefill_wrappers,
            prefill_wrappers_ragged,
            True,
            False,
        )

    else:
        raise ValueError(f"Invalid mode: {forward_mode=}")


def init_forward_metadata_replay_cuda_graph(
    self: FlashInferAttnBackend,
    bs: int,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    encoder_lens: Optional[torch.Tensor],
    forward_mode: ForwardMode,
    spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    seq_lens_cpu: Optional[torch.Tensor],
    prefix_lens: Optional[torch.Tensor] = None,
    extend_seq_len: Optional[int] = None,
):
    if forward_mode.is_decode_or_idle():
        self.indices_updater_decode.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
            seq_lens_sum,
            decode_wrappers=self.decode_cuda_graph_metadata[bs],
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
            spec_info=spec_info,
        )
    elif forward_mode.is_target_verify():
        self.indices_updater_prefill.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
            seq_lens_sum,
            prefix_lens=None,
            prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
            use_ragged=False,
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
            spec_info=spec_info,
        )
    elif forward_mode.is_draft_extend():
        self.indices_updater_prefill.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
            seq_lens_sum,
            prefix_lens=None,
            prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
            use_ragged=False,
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
            spec_info=spec_info,
        )
    elif forward_mode.is_extend():
        self.indices_updater_extend.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
            seq_lens_sum,
            prefix_lens=prefix_lens,
            prefill_wrappers=self.prefill_cuda_graph_metadata[bs][extend_seq_len][0],
            prefill_wrappers_ragged=self.prefill_cuda_graph_metadata[bs][extend_seq_len][1],
            use_ragged=True,
            encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
            spec_info=spec_info,
        )

    else:
        raise ValueError("Invalid forward mode")

def forward_extend(
    self: FlashInferAttnBackend,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
):
    prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[self._get_wrapper_idx(layer)]
    if hasattr(self.forward_metadata, "prefill_wrappers_ragged"):
        prefill_wrapper_ragged = self.forward_metadata.prefill_wrappers_ragged[self._get_wrapper_idx(layer)]
    else:
        prefill_wrapper_ragged = self.prefill_wrapper_ragged

    cache_loc = forward_batch.out_cache_loc if not layer.is_cross_attention else forward_batch.encoder_out_cache_loc

    logits_soft_cap = layer.logit_cap

    q = q.contiguous()
    if not self.forward_metadata.use_ragged:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, layer.k_scale, layer.v_scale)

        o = prefill_wrapper_paged.forward(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            causal=not layer.is_cross_attention,
            sm_scale=layer.scaling,
            window_left=layer.sliding_window_size,
            logits_soft_cap=logits_soft_cap,
            k_scale=layer.k_scale,
            v_scale=layer.v_scale,
        )
    else:
        causal = True
        if layer.attn_type == AttentionType.ENCODER_ONLY:
            save_kv_cache = False
            causal = False

        if self.forward_metadata.extend_no_prefix:
            # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
            # The FlashInfer head_dim limitation itself is tracked here:
            # https://github.com/flashinfer-ai/flashinfer/issues/1048
            o = self.prefill_wrapper_ragged.forward(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=causal,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

        else:
            o1, s1 = prefill_wrapper_ragged.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                k.view(-1, layer.tp_k_head_num, layer.head_dim),
                v.view(-1, layer.tp_v_head_num, layer.head_dim),
                causal=True,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )
            o2, s2 = prefill_wrapper_paged.forward_return_lse(
                q.view(-1, layer.tp_q_head_num, layer.head_dim),
                forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=logits_soft_cap,
            )

            o, _ = merge_state(o1, s1, o2, s2)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, layer.k_scale, layer.v_scale)

    return o.view(-1, layer.tp_q_head_num * layer.head_dim)


class FlashInferIndicesUpdaterExtend:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = model_runner.model_config.num_attention_heads // get_attention_tp_size()
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(get_attention_tp_size())
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
        # if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
        #     self.update = self.update_sliding_window
        # elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
        #     self.update = self.update_cross_attention
        # else:
        assert self.attn_backend.num_wrappers == 1
        self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        prefill_wrappers_ragged: List[BatchPrefillWithRaggedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        prefill_wrappers_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if use_ragged:
            # TODO: remove this device sync, we can use forward_batch.extend_prefix_lens_cpu
            # and forward_batch.extend_seq_lens_cpu
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            prefill_wrappers_ragged[0],
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
        )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        use_sliding_window_kv_pool: bool = False,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]
            custom_mask = None
        else:
            assert isinstance(spec_info, EagleDraftInput) or isinstance(spec_info, EagleVerifyInput)
            kv_indices, kv_indptr, qo_indptr, custom_mask = spec_info.generate_attn_arg_prefill(
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                self.req_to_token,
            )

        # extend part
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
                causal=True,
            )

        if use_sliding_window_kv_pool:
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(kv_indices[:kv_last_index])

        # print("kv_indices", kv_indices)
        # cached part
        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=custom_mask,
            non_blocking=True,
        )
