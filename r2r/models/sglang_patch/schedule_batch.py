from __future__ import annotations

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Store information about requests and batches.

The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.

TODO(lmzheng): ModelWorkerBatch seems a bit redundant and we consider removing it in the future.
"""

import copy
import dataclasses
import logging
import threading
from enum import Enum, auto
from http import HTTPStatus
from itertools import chain
from typing import TYPE_CHECKING, Any, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.global_config import global_config
from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.disaggregation.base import BaseKVSender
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.layers.moe import is_tbo_enabled
from sglang.srt.mem_cache.allocator import (
    BaseTokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.lora_radix_cache import LoRAKey, LoRARadixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
from sglang.srt.metrics.collector import TimeStats
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import flatten_nested_list, support_triton

from sglang.srt.managers.schedule_batch import get_last_loc, MultimodalInputs, write_req_to_token_pool_triton, Req

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5

GLOBAL_SERVER_ARGS_KEYS = [
    "attention_backend",
    "mm_attention_backend",
    "debug_tensor_dump_inject",
    "debug_tensor_dump_output_folder",
    "chunked_prefill_size",
    "device",
    "disable_chunked_prefix_cache",
    "disable_flashinfer_cutlass_moe_fp4_allgather",
    "disable_radix_cache",
    "enable_dp_lm_head",
    "flashinfer_mxfp4_moe_precision",
    "enable_flashinfer_allreduce_fusion",
    "moe_dense_tp_size",
    "ep_dispatch_algorithm",
    "ep_num_redundant_experts",
    "enable_nan_detection",
    "flashinfer_mla_disable_ragged",
    "max_micro_batch_size",
    "disable_shared_experts_fusion",
    "sampling_backend",
    "speculative_accept_threshold_single",
    "speculative_accept_threshold_acc",
    "torchao_config",
    "triton_attention_reduce_in_fp32",
    "num_reserved_decode_tokens",
    "weight_loader_disable_mmap",
    "enable_multimodal",
    "enable_symm_mem",
    "quantization",
    "enable_custom_logit_processor",
    "disaggregation_mode",
]

# Put some global args for easy access
global_server_args_dict = {k: getattr(ServerArgs, k) for k in GLOBAL_SERVER_ARGS_KEYS}

logger = logging.getLogger(__name__)

def __init__(
    self,
    rid: str,
    origin_input_text: str,
    origin_input_ids: List[int],
    sampling_params: SamplingParams,
    return_logprob: bool = False,
    top_logprobs_num: int = 0,
    token_ids_logprob: List[int] = None,
    stream: bool = False,
    origin_input_ids_unpadded: Optional[Tuple[int]] = None,
    lora_id: Optional[str] = None,
    input_embeds: Optional[List[List[float]]] = None,
    token_type_ids: List[int] = None,
    session_id: Optional[str] = None,
    custom_logit_processor: Optional[str] = None,
    return_hidden_states: bool = False,
    eos_token_ids: Optional[Set[int]] = None,
    bootstrap_host: Optional[str] = None,
    bootstrap_port: Optional[int] = None,
    bootstrap_room: Optional[int] = None,
    data_parallel_rank: Optional[int] = None,
    vocab_size: Optional[int] = None,
    status: Optional[str] = "need",
    last_cached_loc: Optional[List[int]] = None,
    last_llm_loc: Optional[int] = None,
):
    # Input and output info
    self.rid = rid
    self.origin_input_text = origin_input_text
    self.origin_input_ids_unpadded = (
        origin_input_ids_unpadded
        if origin_input_ids_unpadded
        else origin_input_ids  # Before image padding
    )
    self.origin_input_ids = origin_input_ids
    # Each decode stage's output ids
    self.output_ids = []
    # fill_ids = origin_input_ids + output_ids. Updated if chunked.
    self.fill_ids = []
    self.session_id = session_id
    self.input_embeds = input_embeds

    # for corss-endoder model
    self.token_type_ids = token_type_ids

    # The length of KV that have been removed in local attention chunked prefill
    self.evicted_seqlen_local = 0

    # Sampling info
    if isinstance(sampling_params.custom_params, dict):
        sampling_params = copy.copy(sampling_params)
        sampling_params.custom_params = sampling_params.custom_params | {
            "__req__": self
        }
    self.sampling_params = sampling_params
    self.custom_logit_processor = custom_logit_processor
    self.return_hidden_states = return_hidden_states
    self.lora_id = lora_id

    # Memory pool info
    self.req_pool_idx: Optional[int] = None

    # Check finish
    self.tokenizer = None
    self.finished_reason = None
    # Whether this request has finished output
    self.finished_output = None
    # If we want to abort the request in the middle of the event loop, set this to true
    # Note: We should never set finished_reason in the middle, the req will get filtered and never respond
    self.to_abort = False
    # This carries the error message for `.to_abort` and will be attached to the finished_reason at the end of the event loop
    self.to_abort_message: str = None
    self.stream = stream
    self.eos_token_ids = eos_token_ids
    self.vocab_size = vocab_size
    self.status = status
    self.last_cached_loc = last_cached_loc
    self.last_llm_loc = last_llm_loc

    # For incremental decoding
    # ----- | --------- read_ids -------|
    # ----- |   surr_ids  |
    # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
    # ----- ^ ----------- ^ ----------- ^
    # ----- 1 ----------- 2 ----------- 3
    # 1: surr_offset
    # 2: read_offset
    # 3: last token
    self.surr_offset = None  # Surrounding offset to defeat the cleanup algorithm
    self.read_offset = None
    self.decoded_text = ""

    # For multimodal inputs
    self.multimodal_inputs: Optional[MultimodalInputs] = None

    # Prefix info
    # The indices to kv cache for the shared prefix.
    self.prefix_indices: torch.Tensor = []
    # Number of tokens to run prefill.
    self.extend_input_len = 0
    # The relative logprob_start_len in an extend batch
    self.extend_logprob_start_len = 0
    self.last_node: Any = None
    self.last_host_node: Any = None
    self.host_hit_length = 0
    # The node to lock until for swa radix tree lock ref
    self.swa_uuid_for_lock: Optional[int] = None

    # Whether or not if it is chunked. It increments whenever
    # it is chunked, and decrement whenever chunked request is
    # processed.
    self.is_chunked = 0

    # For retraction
    self.is_retracted = False

    # Incremental streamining
    self.send_token_offset: int = 0
    self.send_decode_id_offset: int = 0
    # TODO (Byron): send_output_token_logprobs_offset and send_decode_id_offset can be different in disaggregation mode
    # because the decode server does not have the first output token logprobs
    self.send_output_token_logprobs_offset: int = 0

    # Logprobs (arguments)
    self.return_logprob = return_logprob
    # Start index to compute logprob from.
    self.logprob_start_len = 0
    self.top_logprobs_num = top_logprobs_num
    self.token_ids_logprob = token_ids_logprob
    self.temp_scaled_logprobs = False
    self.top_p_normalized_logprobs = False

    # Logprobs (return values)
    # True means the input logprob has been already sent to detokenizer.
    self.input_logprob_sent: bool = False
    self.input_token_logprobs_val: Optional[List[float]] = None
    self.input_token_logprobs_idx: Optional[List[int]] = None
    self.input_top_logprobs_val: Optional[List[float]] = None
    self.input_top_logprobs_idx: Optional[List[int]] = None
    self.input_token_ids_logprobs_val: Optional[List[float]] = None
    self.input_token_ids_logprobs_idx: Optional[List[int]] = None
    # Temporary holder to store input_token_logprobs.
    self.input_token_logprobs: Optional[List[Tuple[int]]] = None
    self.temp_input_top_logprobs_val: Optional[List[torch.Tensor]] = None
    self.temp_input_top_logprobs_idx: Optional[List[int]] = None
    self.temp_input_token_ids_logprobs_val: Optional[List[float]] = None
    self.temp_input_token_ids_logprobs_idx: Optional[List[int]] = None

    if return_logprob:
        # shape: (bs, 1)
        self.output_token_logprobs_val = []
        self.output_token_logprobs_idx = []
        # shape: (bs, k)
        self.output_top_logprobs_val = []
        self.output_top_logprobs_idx = []
        self.output_token_ids_logprobs_val = []
        self.output_token_ids_logprobs_idx = []
    else:
        self.output_token_logprobs_val = self.output_token_logprobs_idx = (
            self.output_top_logprobs_val
        ) = self.output_top_logprobs_idx = self.output_token_ids_logprobs_val = (
            self.output_token_ids_logprobs_idx
        ) = None
    self.hidden_states: List[List[float]] = []
    self.hidden_states_tensor = None  # Note: use tensor instead of list to transfer hidden_states when PD + MTP

    # Embedding (return values)
    self.embedding = None

    # Constrained decoding
    self.grammar: Optional[BaseGrammarObject] = None
    self.grammar_wait_ct = 0

    # The number of cached tokens that were already cached in the KV cache
    self.cached_tokens = 0
    self.already_computed = 0

    # The number of verification forward passes in the speculative decoding.
    # This is used to compute the average acceptance length per request.
    self.spec_verify_ct = 0

    # For metrics
    self.time_stats: TimeStats = TimeStats()
    self.has_log_time_stats: bool = False
    self.queue_time_start = None
    self.queue_time_end = None

    # For disaggregation
    self.bootstrap_host: str = bootstrap_host
    self.bootstrap_port: Optional[int] = bootstrap_port
    self.bootstrap_room: Optional[int] = bootstrap_room
    self.disagg_kv_sender: Optional[BaseKVSender] = None

    # For data parallel rank routing
    self.data_parallel_rank: Optional[int] = data_parallel_rank

    # the start index of the sent kv cache
    # We want to send it chunk by chunk for chunked prefill.
    # After every chunk forward, we do the following:
    # kv_send(req.input_ids[req.start_send_idx:len(req.fill_ids)])
    # start_send_idx = len(req.fill_ids)
    self.start_send_idx: int = 0

    # For overlap schedule, we delay the kv transfer until `process_batch_result_disagg_prefill` rather than `process_prefill_chunk` in non-overlap
    # This is because kv is not ready in `process_prefill_chunk`.
    # We use `tmp_end_idx` to store the end index of the kv cache to send.
    self.tmp_end_idx: int = -1
    self.metadata_buffer_index: int = -1

def init_next_round_input(
    self,
    tree_cache: Optional[BasePrefixCache] = None,
):
    self.fill_ids = self.origin_input_ids + self.output_ids
    if tree_cache is not None:
        if isinstance(tree_cache, LoRARadixCache):
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.host_hit_length,
            ) = tree_cache.match_prefix_with_lora_id(
                key=LoRAKey(
                    lora_id=self.lora_id, token_ids=self.adjust_max_prefix_ids()
                ),
            )
        else:
            (
                self.prefix_indices,
                self.last_node,
                self.last_host_node,
                self.host_hit_length,
            ) = tree_cache.match_prefix(
                key=self.adjust_max_prefix_ids(),
            )
    if self.last_cached_loc is not None:
        self.prefix_indices = torch.tensor(self.last_cached_loc, device=self.device)
    self.extend_input_len = len(self.fill_ids) - len(self.prefix_indices)


def reset_for_retract(self):
    self.prefix_indices = []
    self.last_node = None
    self.swa_uuid_for_lock = None
    self.extend_input_len = 0
    self.is_retracted = True
    self.input_token_logprobs = None
    self.temp_input_top_logprobs_val = None
    self.temp_input_top_logprobs_idx = None
    self.extend_logprob_start_len = 0
    self.is_chunked = 0
    self.req_pool_idx = None
    self.already_computed = 0
    if self.last_cached_loc is not None:
        self.last_cached_loc = []
    self.last_llm_loc = None


def prepare_for_extend(self):
    self.forward_mode = ForwardMode.EXTEND

    # Allocate req slots
    bs = len(self.reqs)

    is_call_llm = False
    for req in self.reqs:
        if req.req_pool_idx is not None:
            is_call_llm = True
            break
    if is_call_llm:
        req_pool_indices = []
        for req in self.reqs:
            if req.req_pool_idx is not None:
                req_pool_indices.append(req.req_pool_idx)
            else:
                req_pool_indices += self.alloc_req_slots(1)
    else:
        req_pool_indices = self.alloc_req_slots(bs)

    # Init tensors
    reqs = self.reqs
    input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
    extend_num_tokens = sum(len(ids) for ids in input_ids)
    seq_lens = [len(r.fill_ids) for r in reqs]
    orig_seq_lens = [max(len(r.fill_ids), len(r.origin_input_ids)) for r in reqs]
    prefix_lens = [len(r.prefix_indices) for r in reqs]
    extend_lens = [r.extend_input_len for r in reqs]

    token_type_ids = [
        r.token_type_ids for r in reqs if r.token_type_ids is not None
    ]

    req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
        self.device, non_blocking=True
    )
    input_ids_tensor = torch.tensor(
        list(chain.from_iterable(input_ids)), dtype=torch.int64
    ).to(self.device, non_blocking=True)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(
        self.device, non_blocking=True
    )
    orig_seq_lens_tensor = torch.tensor(orig_seq_lens, dtype=torch.int32).to(
        self.device, non_blocking=True
    )
    prefix_lens_tensor = torch.tensor(
        prefix_lens, dtype=torch.int64, device=self.device
    )

    token_type_ids_tensor = None
    if len(token_type_ids) > 0:
        token_type_ids_tensor = torch.tensor(
            sum(token_type_ids, []), dtype=torch.int64
        ).to(self.device, non_blocking=True)

    extend_lens_tensor = seq_lens_tensor - prefix_lens_tensor

    # Copy prefix and do some basic check
    input_embeds = []
    extend_input_logprob_token_ids = []
    multimodal_inputs = []

    for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
        req.req_pool_idx = req_pool_indices[i]
        assert seq_len - pre_len == req.extend_input_len

        if pre_len > 0:
            self.req_to_token_pool.write(
                (req.req_pool_idx, slice(0, pre_len)), req.prefix_indices
            )
            if isinstance(self.tree_cache, SWAChunkCache):
                self.tree_cache.evict_swa(
                    req, pre_len, self.model_config.attention_chunk_size
                )

        # If input_embeds are available, store them
        if req.input_embeds is not None:
            # If req.input_embeds is already a list, append its content directly
            input_embeds.extend(req.input_embeds)  # Use extend to avoid nesting

        multimodal_inputs.append(req.multimodal_inputs)

        req.cached_tokens += pre_len - req.already_computed
        req.already_computed = seq_len
        req.is_retracted = False

        # Compute the relative logprob_start_len in an extend batch
        if req.logprob_start_len >= pre_len:
            req.extend_logprob_start_len = min(
                req.logprob_start_len - pre_len,
                req.extend_input_len,
                req.seqlen - 1,
            )
        else:
            req.extend_logprob_start_len = 0

        if self.return_logprob:
            # Find input logprob token ids.
            # First, find a global index within origin_input_ids and slide it by 1
            # to compute input logprobs. It is because you need the next token
            # to compute input logprobs. E.g., (chunk size 2)
            #
            # input_logprobs = [1, 2, 3, 4]
            # fill_ids = [1, 2]
            # extend_input_logprob_token_id = [2, 3]
            #
            # Note that it can also overflow. In this case, we pad it with 0.
            # input_logprobs = [1, 2, 3, 4]
            # fill_ids = [3, 4]
            # extend_input_logprob_token_id = [4, 0]
            global_start_idx, global_end_idx = (
                len(req.prefix_indices),
                len(req.fill_ids),
            )
            # Apply logprob_start_len
            if global_start_idx < req.logprob_start_len:
                global_start_idx = req.logprob_start_len

            logprob_token_ids = req.origin_input_ids[
                global_start_idx + 1 : global_end_idx + 1
            ]
            extend_input_logprob_token_ids.extend(logprob_token_ids)

            # We will need req.extend_input_len - req.extend_logprob_start_len number of
            # tokens, and logprob_token_ids is for input logprob, so pad the rest of them by 0.
            extend_input_logprob_token_ids.extend(
                [0]
                * (
                    req.extend_input_len
                    - req.extend_logprob_start_len
                    - len(logprob_token_ids)
                )
            )

    if self.return_logprob:
        extend_input_logprob_token_ids = torch.tensor(
            extend_input_logprob_token_ids
        )
    else:
        extend_input_logprob_token_ids = None

    # Allocate memory
    if self.token_to_kv_pool_allocator.page_size == 1:
        out_cache_loc = self.alloc_token_slots(extend_num_tokens)
    else:
        last_loc = get_last_loc(
            self.req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_lens_tensor,
        )
        out_cache_loc = self.alloc_paged_token_slots_extend(
            prefix_lens_tensor, seq_lens_tensor, last_loc, extend_num_tokens
        )
    
    tmp_index = 0
    for req in self.reqs:
        if req.last_cached_loc is not None:
            req.last_cached_loc.extend(out_cache_loc[tmp_index : tmp_index + req.extend_input_len].tolist())
        tmp_index += req.extend_input_len

    # Set fields
    self.input_ids = input_ids_tensor
    self.req_pool_indices = req_pool_indices_tensor
    self.seq_lens = seq_lens_tensor
    self.orig_seq_lens = orig_seq_lens_tensor
    self.out_cache_loc = out_cache_loc
    self.input_embeds = (
        torch.tensor(input_embeds).to(self.device, non_blocking=True)
        if input_embeds
        else None
    )
    for mm_input in multimodal_inputs:
        if mm_input is None:
            continue
        for mm_item in mm_input.mm_items:
            pixel_values = getattr(mm_item, "feature", None)
            if isinstance(pixel_values, torch.Tensor):
                mm_item.feature = pixel_values.to(self.device, non_blocking=True)
    self.multimodal_inputs = multimodal_inputs
    self.token_type_ids = token_type_ids_tensor
    self.seq_lens_sum = sum(seq_lens)

    if self.return_logprob:
        self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
        self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

    self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
    self.extend_num_tokens = extend_num_tokens
    self.prefix_lens = prefix_lens
    self.extend_lens = extend_lens
    self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

    # Write to req_to_token_pool
    if support_triton(global_server_args_dict.get("attention_backend")):
        # TODO: some tensors can be reused for ForwardBatchInfo (e.g., extend_lens, cumsum_start)

        write_req_to_token_pool_triton[(bs,)](
            self.req_to_token_pool.req_to_token,
            req_pool_indices_tensor,
            prefix_lens_tensor,
            seq_lens_tensor,
            extend_lens_tensor,
            out_cache_loc,
            self.req_to_token_pool.req_to_token.shape[1],
        )
    else:
        pt = 0
        for i in range(bs):
            self.req_to_token_pool.write(
                (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
                out_cache_loc[pt : pt + extend_lens[i]],
            )
            pt += extend_lens[i]

    if self.model_config.is_encoder_decoder:
        self.prepare_encoder_info_extend(input_ids, seq_lens)

    # Build sampling info
    self.sampling_info = SamplingBatchInfo.from_schedule_batch(
        self,
        self.model_config.vocab_size,
    )

def filter_batch(
    self,
    chunked_req_to_exclude: Optional[Union[Req, List[Req]]] = None,
    keep_indices: Optional[List[int]] = None,
):
    if keep_indices is None:
        if isinstance(chunked_req_to_exclude, Req):
            chunked_req_to_exclude = [chunked_req_to_exclude]
        elif chunked_req_to_exclude is None:
            chunked_req_to_exclude = []
        keep_indices = [
            i
            for i in range(len(self.reqs))
            if not self.reqs[i].finished()
            and self.reqs[i] not in chunked_req_to_exclude and self.reqs[i].status != "finished"
        ]

    if keep_indices is None or len(keep_indices) == 0:
        keep_indices = []

    if len(keep_indices) == len(self.reqs):
        # No need to filter
        return

    keep_indices_device = torch.tensor(keep_indices, dtype=torch.int64).to(
        self.device, non_blocking=True
    )

    if self.model_config.is_encoder_decoder:
        self.encoder_lens = self.encoder_lens[keep_indices_device]
        self.encoder_lens_cpu = [self.encoder_lens_cpu[i] for i in keep_indices]

    self.reqs = [self.reqs[i] for i in keep_indices]
    if self.multimodal_inputs is not None:
        self.multimodal_inputs = [self.multimodal_inputs[i] for i in keep_indices]
    self.req_pool_indices = self.req_pool_indices[keep_indices_device]
    self.seq_lens = self.seq_lens[keep_indices_device]
    self.orig_seq_lens = self.orig_seq_lens[keep_indices_device]
    self.out_cache_loc = None
    self.seq_lens_sum = self.seq_lens.sum().item()
    if self.output_ids is not None:
        self.output_ids = self.output_ids[keep_indices_device]
    self.return_logprob = any(req.return_logprob for req in self.reqs)
    if self.return_logprob:
        self.top_logprobs_nums = [self.top_logprobs_nums[i] for i in keep_indices]
        self.token_ids_logprobs = [self.token_ids_logprobs[i] for i in keep_indices]
    else:
        self.top_logprobs_nums = None
        self.token_ids_logprobs = None

    self.has_stream = any(req.stream for req in self.reqs)
    self.has_grammar = any(req.grammar for req in self.reqs)

    self.sampling_info.filter_batch(keep_indices, keep_indices_device)
    if self.spec_info:
        self.spec_info.filter_batch(keep_indices_device)
