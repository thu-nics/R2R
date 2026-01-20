from __future__ import annotations

import bisect
import inspect
import logging
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import tqdm
from torch.profiler import ProfilerActivity, profile

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.distributed.parallel_state import graph_capture
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_rank,
    get_attention_tp_size,
    set_dp_buffer_len,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.two_batch_overlap import TboCudaGraphRunnerPlugin
from sglang.srt.utils import (
    empty_context,
    get_available_gpu_memory,
    log_info_on_rank0,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    get_batch_sizes_to_capture,
    set_torch_compile_config,
    model_capture_mode,
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
    freeze_gc,
    patch_model,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
logger = logging.getLogger(__name__)

import r2r.models.sglang_patch.flashinfer_cuda_graph

PREFIX_LEN = 8

class LLMCudaGraphRunner(CudaGraphRunner):

    def __init__(self, model_runner: ModelRunner):
        # Parse args
        self.model_runner = model_runner
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.is_encoder_decoder = model_runner.model_config.is_encoder_decoder
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.enable_two_batch_overlap = model_runner.server_args.enable_two_batch_overlap
        self.speculative_algorithm = model_runner.server_args.speculative_algorithm
        self.enable_profile_cuda_graph = model_runner.server_args.enable_profile_cuda_graph
        self.tp_size = model_runner.server_args.tp_size
        self.dp_size = model_runner.server_args.dp_size
        self.pp_size = model_runner.server_args.pp_size

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Batch sizes to capture
        # self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(model_runner)
        self.capture_bs, self.compile_bs = [1, 2], []
        log_info_on_rank0(logger, f"Capture cuda graph bs {self.capture_bs}")
        self.capture_forward_mode = ForwardMode.EXTEND
        self.capture_hidden_mode = CaptureHiddenMode.NULL
        self.extend_num_tokens_per_bs = list(range(16, 0, -1))
        self.max_extend_num_tokens_per_bs = max(self.extend_num_tokens_per_bs)
        if model_runner.spec_algorithm.is_eagle():
            # if self.model_runner.is_draft_worker:
            #     raise RuntimeError("This should not happen")
            # else:
            #     self.capture_forward_mode = ForwardMode.TARGET_VERIFY
            #     self.extend_num_tokens_per_bs = self.model_runner.server_args.speculative_num_draft_tokens
            raise NotImplementedError("Speculative decoding is not supported in LLMCudaGraphRunner.")

        # If returning hidden states is enabled, set initial capture hidden mode to full to avoid double-capture on startup
        if model_runner.server_args.enable_return_hidden_states:
            self.capture_hidden_mode = CaptureHiddenMode.FULL

        # Attention backend
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_extend_num_tokens_per_bs
        self.model_runner.attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)
        self.seq_len_fill_value = self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()

        # FIXME(lsyin): leave it here for now, I don't know whether it is necessary
        self.encoder_len_fill_value = 0
        self.seq_lens_cpu = torch.full((self.max_bs,), self.seq_len_fill_value, dtype=torch.int32)

        if self.enable_torch_compile:
            set_torch_compile_config()

        if self.model_runner.server_args.enable_lora:
            self.model_runner.lora_manager.init_cuda_graph_batch_info(self.max_bs)

        # Graph inputs
        with torch.device(self.device):
            self.input_ids = torch.zeros(self.max_num_token, dtype=torch.int64)
            self.req_pool_indices = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.seq_lens = torch.zeros((self.max_bs,), dtype=torch.int32)
            self.out_cache_loc = torch.zeros((self.max_num_token,), dtype=self._cache_loc_dtype())
            self.positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            # self.positions = torch.arange(1, self.max_extend_num_tokens_per_bs+1).expand(self.max_bs, self.max_extend_num_tokens_per_bs)
            self.mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
            self.num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
            self.tbo_plugin = TboCudaGraphRunnerPlugin()
            self.extend_prefix_lens = torch.ones((self.max_bs,), dtype=torch.int32) * PREFIX_LEN
            self.extend_start_loc = torch.arange(self.max_bs, dtype=torch.int32)
            self.extend_seq_lens = torch.ones((self.max_bs,), dtype=torch.int32)
            # pipeline parallelism
            if self.pp_size > 1:
                self.pp_proxy_tensors = {
                    "hidden_states": torch.zeros(
                        (self.max_bs, self.model_runner.model_config.hidden_size),
                        dtype=torch.bfloat16,
                    ),
                    "residual": torch.zeros(
                        (self.max_bs, self.model_runner.model_config.hidden_size),
                        dtype=torch.bfloat16,
                    ),
                }

            # Speculative_inference
            if model_runner.spec_algorithm.is_eagle3():
                self.model_runner.model.set_eagle3_layers_to_capture()

            if self.is_encoder_decoder:
                # NOTE: encoder_lens can influence the full_text_row_masked_out_mask tensor when doing mixed batch
                self.encoder_lens = torch.full((self.max_bs,), self.encoder_len_fill_value, dtype=torch.int32)
            else:
                self.encoder_lens = None

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    self.global_num_tokens_gpu = torch.zeros((self.dp_size,), dtype=torch.int32)
                    self.global_num_tokens_for_logprob_gpu = torch.zeros((self.dp_size,), dtype=torch.int32)
                else:
                    assert self.require_attn_tp_gather
                    self.global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    self.global_num_tokens_for_logprob_gpu = torch.zeros((1,), dtype=torch.int32)
            else:
                self.global_num_tokens_gpu = None
                self.global_num_tokens_for_logprob_gpu = None

            self.custom_mask = torch.ones(
                ((self.seq_lens.sum().item() + self.max_num_token) * self.max_extend_num_tokens_per_bs),
                dtype=torch.bool,
                device=self.device,
            )
            self.next_token_logits_buffer = torch.zeros(
                (self.max_num_token, self.model_runner.model_config.vocab_size),
                dtype=torch.float,
                device=self.device,
            )

        # Capture
        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}")


    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = max(forward_batch.global_num_tokens_cpu) // self.extend_num_tokens_per_bs if self.model_runner.spec_algorithm.is_eagle() else max(forward_batch.global_num_tokens_cpu)
        else:
            cuda_graph_bs = forward_batch.batch_size
            cuda_seq_len = forward_batch.extend_seq_lens

        is_bs_supported = cuda_graph_bs in self.graphs # if self.disable_padding else cuda_graph_bs <= self.max_bs
        if is_bs_supported:
            is_seq_len_supported = cuda_seq_len.sum() in self.extend_num_tokens_per_bs # all(seq_len in self.extend_num_tokens_per_bs for seq_len in cuda_seq_len)
            is_bs_supported = is_bs_supported and is_seq_len_supported

        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph

        # NOTE: cuda graph cannot handle mixed batch (encoder_len = 0)
        # If mixed batch cannot be supported, then encoder_lens can be removed in cuda graph
        # because the full_text_row_masked_out_mask tensor will always be ones
        is_encoder_lens_supported = torch.all(forward_batch.encoder_lens > 0) if self.is_encoder_decoder else True

        requested_capture_hidden_mode = max(
            forward_batch.capture_hidden_mode,
            (forward_batch.spec_info.capture_hidden_mode if getattr(forward_batch.spec_info, "capture_hidden_mode", None) is not None else CaptureHiddenMode.NULL),
        )
        capture_hidden_mode_matches = requested_capture_hidden_mode == CaptureHiddenMode.NULL or requested_capture_hidden_mode == self.capture_hidden_mode
        is_tbo_supported = forward_batch.can_run_tbo if self.enable_two_batch_overlap else True

        return is_bs_supported and is_encoder_lens_supported and is_tbo_supported and capture_hidden_mode_matches

    def capture_one_batch_size(self, bs: int, extend_seq_len: int, forward: Callable):
        assert extend_seq_len >= bs
        graph = self._create_device_graph()
        stream = self.stream
        num_tokens = extend_seq_len

        # Graph inputs
        input_ids = self.input_ids[:num_tokens]
        req_pool_indices = self.req_pool_indices[:bs]
        self.seq_lens[:bs].fill_(PREFIX_LEN + 1)
        self.seq_lens[bs-1] = PREFIX_LEN + extend_seq_len - (bs - 1)
        seq_lens = self.seq_lens[:bs]
        out_cache_loc = self.out_cache_loc[:num_tokens]
        positions = self.positions[:num_tokens]

        self.extend_seq_lens[:bs].fill_(1)
        self.extend_seq_lens[bs-1] = extend_seq_len-(bs-1)
        extend_seq_lens = self.extend_seq_lens[:bs]
        extend_prefix_lens = self.extend_prefix_lens[:bs]
        extend_start_loc = self.extend_start_loc[:bs]
        if self.is_encoder_decoder:
            encoder_lens = self.encoder_lens[:bs]
        else:
            encoder_lens = None
        mrope_positions = self.mrope_positions[:, :num_tokens]
        next_token_logits_buffer = self.next_token_logits_buffer[:bs]
        self.num_token_non_padded[...] = num_tokens

        # pipeline parallelism
        if self.pp_size > 1:
            pp_proxy_tensors = PPProxyTensors({k: v[:num_tokens] for k, v in self.pp_proxy_tensors.items()})

        if self.require_mlp_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.dp_size,
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.dp_size
        elif self.require_attn_tp_gather:
            self.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            self.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=input_ids.device,
                )
            )
            global_dp_buffer_len = num_tokens
        else:
            global_dp_buffer_len = None

        spec_info = self.get_spec_info(num_tokens)
        if self.capture_hidden_mode != CaptureHiddenMode.FULL:
            self.capture_hidden_mode = spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL

        if self.model_runner.server_args.enable_lora:
            # It is safe to capture CUDA graph using empty LoRA id, as the LoRA kernels will always be launched whenever
            # `--enable-lora` is set to True (and return immediately if the LoRA id is empty for perf optimization).
            lora_ids = [None] * bs
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=self.capture_forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_num_tokens=bs * extend_seq_len,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            next_token_logits_buffer=next_token_logits_buffer,
            orig_seq_lens=seq_lens,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens.sum().item(),
            encoder_lens=encoder_lens,
            return_logprob=False,
            positions=positions,
            global_num_tokens_gpu=self.global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=self.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=mrope_positions,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=self.capture_hidden_mode,
            num_token_non_padded=self.num_token_non_padded,
            global_forward_mode=self.capture_forward_mode,
            lora_ids=lora_ids,
        )
        self.tbo_plugin.capture_one_batch_size(forward_batch, num_tokens=num_tokens)

        if lora_ids is not None:
            self.model_runner.lora_manager.prepare_lora_batch(forward_batch)

        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            prefix_lens=extend_prefix_lens,
            extend_seq_len=extend_seq_len,
        )

        # Run and capture
        def run_once():
            # Clean intermediate result cache for DP attention
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(global_dp_buffer_len, num_tokens)

            kwargs = {}
            if self.pp_size > 1 and "pp_proxy_tensors" in inspect.signature(forward).parameters:
                kwargs["pp_proxy_tensors"] = PPProxyTensors({k: v.clone() for k, v in pp_proxy_tensors.tensors.items()})

            logits_output_or_pp_proxy_tensors = forward(
                input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        # Set graph pool id globally to be able to use symmetric memory
        set_graph_pool_id(get_global_graph_memory_pool())
        out = self._capture_graph(graph, get_global_graph_memory_pool(), stream, run_once)

        return graph, out

    def capture(self) -> None:
        profile_context = empty_context()
        if self.enable_profile_cuda_graph:
            profile_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            )

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc), graph_capture() as graph_capture_context:
            with profile_context as prof:
                self.stream = graph_capture_context.stream
                avail_mem = get_available_gpu_memory(
                    self.model_runner.device,
                    self.model_runner.gpu_id,
                    empty_cache=False,
                )
                # Reverse the order to enable better memory sharing across cuda graphs.
                capture_range = tqdm.tqdm(list(reversed(self.capture_bs))) if get_tensor_model_parallel_rank() == 0 else reversed(self.capture_bs)
                for i, bs in enumerate(capture_range):
                    if bs not in self.graphs:
                        self.graphs[bs] = {}
                    for j, seq_len in enumerate(self.extend_num_tokens_per_bs):
                        if seq_len < bs:
                            continue
                        if get_tensor_model_parallel_rank() == 0:
                            avail_mem = get_available_gpu_memory(
                                self.model_runner.device,
                                self.model_runner.gpu_id,
                                empty_cache=False,
                            )
                            capture_range.set_description(f"Capturing batches ({bs=} {seq_len=} {avail_mem=:.2f} GB)")

                        with patch_model(
                            self.model_runner.model,
                            bs in self.compile_bs,
                            num_tokens=bs * seq_len,
                            tp_group=self.model_runner.tp_group,
                        ) as forward:
                            (
                                graph,
                                output_buffers,
                            ) = self.capture_one_batch_size(bs, seq_len, forward)
                            self.graphs[bs][seq_len] = graph
                            self.output_buffers[bs] = output_buffers

                        # Save gemlite cache after each capture
                        save_gemlite_cache()

        if self.enable_profile_cuda_graph:
            log_message = (
                "Sorted by CUDA Time:\n"
                + prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10)
                + "\n\nSorted by CPU Time:\n"
                + prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10)
            )
            logger.info(log_message)

    def replay_prepare(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ):
        self.recapture_if_needed(forward_batch)

        raw_bs = forward_batch.batch_size
        raw_seq_len = forward_batch.extend_seq_lens
        raw_num_token = raw_seq_len.sum().item()

        # Pad
        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = max_num_tokens / self.extend_num_tokens_per_bs if self.model_runner.spec_algorithm.is_eagle() else max_num_tokens
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)
        bs = self.capture_bs[index]
        if bs != raw_bs:
            self.seq_lens.fill_(self.seq_len_fill_value)
            self.out_cache_loc.zero_()

        # Common inputs
        self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
        self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:raw_num_token].copy_(forward_batch.out_cache_loc)
        self.positions[:raw_num_token].copy_(forward_batch.positions)
        self.extend_prefix_lens[:raw_bs].copy_(forward_batch.extend_prefix_lens)
        self.extend_seq_lens[:raw_bs].copy_(raw_seq_len)

        seq_lens_cpu = None

        if pp_proxy_tensors:
            for key in self.pp_proxy_tensors.keys():
                dim = pp_proxy_tensors[key].shape[0]
                self.pp_proxy_tensors[key][:dim].copy_(pp_proxy_tensors[key])

        if self.is_encoder_decoder:
            self.encoder_lens[:raw_bs].copy_(forward_batch.encoder_lens)
        if forward_batch.mrope_positions is not None:
            self.mrope_positions[:, :raw_num_token].copy_(forward_batch.mrope_positions)
        if self.require_gathered_buffer:
            self.global_num_tokens_gpu.fill_(bs * self.extend_num_tokens_per_bs)
            self.global_num_tokens_for_logprob_gpu.fill_(bs * self.extend_num_tokens_per_bs)
        if enable_num_token_non_padded(self.model_runner.server_args):
            num_token_non_padded = forward_batch.num_token_non_padded
            if self.require_gathered_buffer:
                tokens_per_rank = bs // self.attn_tp_size * self.extend_num_tokens_per_bs
                num_local_token_non_padded = torch.clamp(
                    num_token_non_padded - tokens_per_rank * self.attn_tp_rank,
                    min=0,
                    max=tokens_per_rank,
                )
                self.num_token_non_padded.copy_(num_local_token_non_padded)
            else:
                self.num_token_non_padded.copy_(num_token_non_padded)
        if self.enable_two_batch_overlap:
            self.tbo_plugin.replay_prepare(
                forward_mode=self.capture_forward_mode,
                bs=bs,
                num_token_non_padded=len(forward_batch.input_ids),
                spec_info=forward_batch.spec_info,
            )
        if forward_batch.forward_mode.is_idle() and forward_batch.spec_info is not None:
            forward_batch.spec_info.custom_mask = self.custom_mask
        # Attention backend
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            forward_batch.seq_lens_sum + (bs - raw_bs) * self.seq_len_fill_value,
            self.encoder_lens[:bs] if self.is_encoder_decoder else None,
            self.capture_forward_mode,
            forward_batch.spec_info,
            seq_lens_cpu=seq_lens_cpu,
            prefix_lens=self.extend_prefix_lens[:bs],
            extend_seq_len=raw_num_token,
        )

        # Store fields
        self.raw_bs = raw_bs
        self.raw_num_token = raw_num_token
        self.bs = bs
        self.seq_len = raw_seq_len

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        else:
            # In speculative decoding, these two fields are still needed.
            self.input_ids[: self.raw_num_token].copy_(forward_batch.input_ids)
            self.positions[: self.raw_num_token].copy_(forward_batch.positions)

        # Replay
        self.graphs[self.bs][self.raw_num_token].replay()

        output = self.output_buffers[self.bs]
        if isinstance(output, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                next_token_logits=output.next_token_logits[: self.raw_num_token],
                hidden_states=(
                    output.hidden_states[: self.raw_num_token]
                    if output.hidden_states is not None
                    else None
                ),
            )
        else:
            assert isinstance(output, PPProxyTensors)
            return PPProxyTensors({k: v[: self.bs] for k, v in output.tensors.items()})

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with self.device_module.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out
