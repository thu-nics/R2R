from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)

from sglang.srt.managers.tp_worker import TpModelWorker
from r2r.models.batch_inference.llm_model_runner import LLMModelRunner

class LLMTpModelWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            is_draft_model=is_draft_worker,
        )

        self.model_runner = LLMModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            nccl_port=nccl_port,
            dp_rank=dp_rank,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_running_requests > 0
        ), "max_queued_requests is zero. We need to be at least 1 to schedule a request."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

        self.hicache_layer_transfer_counter = None
