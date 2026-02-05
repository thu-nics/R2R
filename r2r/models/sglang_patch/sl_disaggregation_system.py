import uuid
import torch
import time
import zmq
import pickle
import socket
from tqdm import tqdm
from typing import Optional, Tuple, Union, List, Dict
import multiprocessing as mp
import torch.distributed as dist
from transformers import AutoTokenizer
import threading
import asyncio
import os
from multiprocessing import Value


def find_free_port() -> int:
    """Find and return a free port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


from r2r.models.recorder import GenerationRecord, GenerationRecorder
from r2r.models.sglang_patch.slm_server import SLMServer
from r2r.models.sglang_patch.llm_server import LLMServer
from r2r.utils.config import (
    QUICK_COLOR,
    REFERENCE_COLOR,
    RESET,
)
from r2r.utils.switching import create_switching_strategy
from r2r.utils.token_manager import SGLangTokenManager
from r2r.utils.dataclass import ModelOutputs
from r2r.utils.sampling import sample_token

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ForwardMode, SamplingBatchInfo, write_req_to_token_pool_triton
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.configs.model_config import ModelConfig

import r2r.models.sglang_patch.patch


def get_mem_fraction_statics(
    model_config: Dict,
    overlap_tp_schedule: bool = False,
    quick_sglang_kwargs: Dict = {},
    reference_sglang_kwargs: Dict = {},
    quick_num_gpus: int = 1,
    reference_num_gpus: int = 1
) -> Tuple[float, float]:
    if not overlap_tp_schedule:
        return 0.9, 0.9

    small_server_args = ServerArgs(
        model_path=model_config["quick"]["model_path"],
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=False,
        mem_fraction_static=0.9,
        **quick_sglang_kwargs,
    )
    large_server_args = ServerArgs(
        model_path=model_config["reference"]["model_path"],
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=False,
        mem_fraction_static=0.9,
        **reference_sglang_kwargs,
    )

    large_model_config = ModelConfig.from_server_args(large_server_args)
    small_model_config = ModelConfig.from_server_args(small_server_args)

    small_num_layers = small_model_config.num_hidden_layers
    large_num_layers = large_model_config.num_hidden_layers

    small_cell_size = (
        small_model_config.get_num_kv_heads(quick_num_gpus)
        * small_model_config.head_dim
        * small_num_layers
        * 2
        * torch._utils._element_size(small_model_config.dtype)
    )
    large_cell_size = (
        large_model_config.get_num_kv_heads(reference_num_gpus)
        * large_model_config.head_dim
        * large_num_layers
        * 2
        * torch._utils._element_size(large_model_config.dtype)
    )

    small_bytes = torch.tensor([], dtype=small_model_config.dtype).element_size()
    small_model_mem = small_bytes * float(model_config["quick"]["param"]) # in GB
    large_bytes = torch.tensor([], dtype=large_model_config.dtype).element_size()
    large_model_mem = large_bytes * float(model_config["reference"]["param"]) # in GB
    total_gpu_memory = min(torch.cuda.get_device_properties(i).total_memory for i in range(max(quick_num_gpus,reference_num_gpus))) / (1 << 30) # in GB
    available_gpu_memory = (total_gpu_memory * 0.95 - (small_model_mem / quick_num_gpus + large_model_mem / reference_num_gpus))

    assert available_gpu_memory >= 0, f"Not enough GPU memory for both models: total {total_gpu_memory} GB, used {small_model_mem / quick_num_gpus + large_model_mem / reference_num_gpus} GB"

    total_token_num = int(available_gpu_memory * 0.9 * (1<<30) / (small_cell_size + large_cell_size))

    small_mem_fraction_static = (total_token_num * small_cell_size + small_model_mem / quick_num_gpus * (1<<30)) / (total_gpu_memory * (1<<30))
    large_mem_fraction_static = (total_token_num * large_cell_size + large_model_mem / reference_num_gpus * (1<<30)) / (total_gpu_memory * (1<<30))

    print(f"[SLDisaggregationSystem] overlap_tp_schedule=True, calculated mem_fraction_static: quick model {small_mem_fraction_static:.2f}, reference model {large_mem_fraction_static:.2f}, max num tokens is {total_token_num}")

    return small_mem_fraction_static, large_mem_fraction_static/(1-small_mem_fraction_static)


class SLDisaggregationSystem:
    """Dynamic model selection using SGLang models"""

    def __init__(
        self,
        model_config: Dict,
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        switching_strategy: str = "neural",
        strategy_kwargs: Optional[dict] = None,
        is_record: bool = False,
        quick_sglang_kwargs: Optional[dict] = None,
        reference_sglang_kwargs: Optional[dict] = None,
        is_logits_processor: bool = True,
        overlap_tp_schedule: bool = False,
    ):
        """Initialize the SL Disaggregation System for dynamic model selection.

        Args:
            model_config: Model configuration dict with structure:
                {
                    "quick": {"model_path": str, "param": str},
                    "reference": {"model_path": str, "param": str}
                }
            device: Device to run models on (default: "cuda")
            dtype: Data type for model weights (default: torch.bfloat16)
            switching_strategy: Strategy for routing between models.
                Options: "neural", "always_quick", "always_reference", etc.
            strategy_kwargs: Additional kwargs for the switching strategy.
                For "neural": {"model_path": str, "threshold": float}
            is_record: Whether to record generation details (default: False)
            quick_sglang_kwargs: SGLang kwargs for quick model, e.g.:
                {"dtype": "bfloat16", "tp_size": 1, "enable_return_hidden_states": True}
            reference_sglang_kwargs: SGLang kwargs for reference model, e.g.:
                {"dtype": "bfloat16", "tp_size": 1}
            is_logits_processor: Whether to use logits processor (default: True)
            overlap_tp_schedule: Whether to overlap TP scheduling for memory
                optimization when both models share GPUs (default: False)
        """
        self.device = device
        self.dtype = dtype
        self.model_config = model_config
        self.strategy_kwargs = strategy_kwargs or {}
        self.switching_strategy = switching_strategy
        self.is_record = is_record
        self.rid = 1
        self.rid_lock = threading.Lock()
        self.tokenizer = None
        self.reference_tokenizer = None

        # Merge kwargs
        quick_sglang_kwargs = {**(quick_sglang_kwargs or {})}
        reference_sglang_kwargs = {**(reference_sglang_kwargs or {})}

        # GPU info
        self.quick_num_gpus = quick_sglang_kwargs.get("tp_size", torch.cuda.device_count())
        self.reference_num_gpus = reference_sglang_kwargs.get("tp_size", torch.cuda.device_count())
        #self.world_size = self.num_gpus

        # Create dictionary to store recorders
        self.generation_records = {}

        self.total_prompts_num = 0

        # Currently only support tp_size=1 for quick model
        quick_sglang_kwargs["tp_size"] = self.quick_num_gpus
        reference_sglang_kwargs["tp_size"] = self.reference_num_gpus
        assert self.reference_num_gpus >= 1, f"Using {self.reference_num_gpus} GPUs for SGLang, expected larger than 1."
        print(f"Using {self.reference_num_gpus+self.quick_num_gpus if overlap_tp_schedule is False else max(self.reference_num_gpus, self.quick_num_gpus)} GPUs for SGLang, with {self.quick_num_gpus} for quick.")

        # ZMQ PUB socket (broadcast producer) so all TP ranks receive the same message
        self.zmq_ctx = zmq.Context.instance()
        self.req_sender = self.zmq_ctx.socket(zmq.PUB)
        self.req_port = self.req_sender.bind_to_random_port("tcp://127.0.0.1")

        self._quick_ready_queue = mp.Queue()

        # Compute MASTER_PORT once in the main process and pass to workers
        self.slm_master_port = find_free_port()
        print(f"[SLDisaggregationSystem] SLM MASTER_ADDR: localhost, MASTER_PORT: {self.slm_master_port}")

        small_mem_fraction_static, large_mem_fraction_static = get_mem_fraction_statics(
            model_config=self.model_config,
            overlap_tp_schedule=overlap_tp_schedule,
            quick_sglang_kwargs=quick_sglang_kwargs,
            reference_sglang_kwargs=reference_sglang_kwargs,
            quick_num_gpus=self.quick_num_gpus,
            reference_num_gpus=self.reference_num_gpus,
        )

        """
        cell_size = (
            self.model_config.get_num_kv_heads(get_attention_tp_size())
            * self.model_config.head_dim
            * num_layers
            * 2
            * torch._utils._element_size(self.kv_cache_dtype)
        )
        """
        self.llm_kvcache_size = Value('i', 0)

        # Launch quick model workers with req_port(port that receive Req objects)
        # Inter-server queues (Q2): create before servers so workers can use them
        # Instantiate SLMServer first (it binds its PUB for SLM->LLM)
        self.slm_server = SLMServer(
            model_config=self.model_config,
            quick_sglang_kwargs=quick_sglang_kwargs,
            quick_num_gpus=self.quick_num_gpus,
            req_port=self.req_port,
            ready_queue=self._quick_ready_queue,
            switching_strategy=self.switching_strategy,
            strategy_kwargs=self.strategy_kwargs,
            mem_fraction_static=small_mem_fraction_static,
            llm_kvcache_size=self.llm_kvcache_size,
            master_port=self.slm_master_port,  # Pass master_port to SLMServer
        )

        try:
            got = 0
            tok = None
            while got < self.quick_num_gpus:
                msg, rank, payload = self._quick_ready_queue.get(timeout=120)
                if msg != "READY":
                    raise RuntimeError(f"unexpected ready msg: {msg}")
                if tok is None and payload is not None:
                    tok = payload
                got += 1
            assert tok is not None, "Failed to get tokenizer from quick model scheduler"
            self.tokenizer = tok
        except Exception as e:
            raise RuntimeError("Waiting for SLMServer launching or Failed to get tokenizer from scheduler") from e


        # Launch reference model workers
        self._llm_ready_queue = mp.Queue()
        self.llm_server = LLMServer(
            model_config=self.model_config,
            reference_sglang_kwargs=reference_sglang_kwargs,
            quick_num_gpus=self.quick_num_gpus,
            reference_num_gpus=self.reference_num_gpus,
            reference_master_port=find_free_port(),
            ready_queue=self._llm_ready_queue,
            overlap_tp_schedule=overlap_tp_schedule,
            mem_fraction_static=large_mem_fraction_static,
            llm_kvcache_size=self.llm_kvcache_size,
        )

        try:
            # Wait LLM workers
            got_llm = 0
            ref_tok = None
            while got_llm < self.reference_num_gpus:
                ready_msg = self._llm_ready_queue.get(timeout=300)
                # Compatible with formats: (msg, rank) or (msg, rank, tokenizer)
                if len(ready_msg) == 2:
                    msg, rank = ready_msg
                    tk = None
                else:
                    msg, rank, tk = ready_msg
                if msg != "READY":
                    raise RuntimeError(f"unexpected llm ready msg: {msg}")
                if ref_tok is None and tk is not None:
                    ref_tok = tk
                got_llm += 1
            if ref_tok is None:
                raise RuntimeError("Failed to get tokenizer from LLMServer rank0")
            self.reference_tokenizer = ref_tok
            self._llm_all_ready = True
            print(f"[SLDisaggregationSystem] llm READY={got_llm}")
        except Exception as e:
            raise RuntimeError("Waiting for reference model workers launching failed") from e

        if self.tokenizer is None:
            self.tokenizer = self.reference_tokenizer
        assert self.tokenizer is not None, "Failed to get tokenizer from both quick and reference models"

        # After both servers constructed, start reciprocal SUB threads (controller step):
        # 1) LLM subscribe to SLM's PUB
        llm_recv_port = getattr(self.slm_server, 'llm_recv_port', None)
        if llm_recv_port is not None:
            self.llm_server.start_llm_sub(llm_recv_port)
        else:
            print('[SLDisaggregationSystem] Warning: slm_server.llm_recv_port is None')
        # 2) SLM subscribe to LLM's PUB
        slm_recv_port = getattr(self.llm_server, 'slm_recv_port', None)
        if slm_recv_port is not None:
            self.slm_server.start_slm_sub(slm_recv_port)
        else:
            print('[SLDisaggregationSystem] Warning: llm_server.slm_recv_port is None')

        self.finished_reqs = {}
        self.finished_reqs_rids = []

        # 3) System subscribe to SLM finished-reqs PUB
        self._finished_reqs_queue = mp.Queue()
        self._finished_recv_stop = mp.Event()
        finished_port = getattr(self.slm_server, 'system_receive_port', None)
        if finished_port is None:
            print('[SLDisaggregationSystem] Warning: slm_server.system_receive_port is None; finished reqs will not be received')
            self._finished_sub = None
        else:
            try:
                self._finished_sub = self.zmq_ctx.socket(zmq.SUB)
                self._finished_sub.setsockopt(zmq.LINGER, 0)
                self._finished_sub.setsockopt(zmq.RCVHWM, 100000)
                self._finished_sub.connect(f"tcp://127.0.0.1:{finished_port}")
                self._finished_sub.setsockopt(zmq.SUBSCRIBE, b"")
            except Exception as e:
                print(f"[SLDisaggregationSystem] Failed to start SUB for finished reqs: {e}")
                self._finished_sub = None

            def _finished_recv_loop():
                if self._finished_sub is None:
                    return
                poller = zmq.Poller()
                poller.register(self._finished_sub, zmq.POLLIN)
                while not self._finished_recv_stop.is_set():
                    try:
                        events = dict(poller.poll(timeout=50))
                    except Exception:
                        continue
                    if self._finished_sub in events and events[self._finished_sub] == zmq.POLLIN:
                        while True:
                            try:
                                msg = self._finished_sub.recv_pyobj(flags=zmq.NOBLOCK)
                                # msg may be a dict payload from SLMServer
                                if isinstance(msg, dict) and msg.get("status") == "finished":
                                    rid = msg.get("rid")
                                    if rid not in self.finished_reqs_rids:
                                        self.finished_reqs[rid] = msg
                                        self.finished_reqs_rids.append(rid)
                                else:
                                    # fallback: assume Req-like object
                                    rid = getattr(msg, 'rid', None)
                                    self.finished_reqs[rid] = msg
                                    self.finished_reqs_rids.append(rid)
                            except zmq.Again:
                                break
                            except Exception:
                                break
                            try:
                                self._finished_reqs_queue.put_nowait(msg)
                            except Exception:
                                try:
                                    self._finished_reqs_queue.put(msg)
                                except Exception:
                                    pass
            self._finished_recv_thread = threading.Thread(target=_finished_recv_loop, daemon=True)
            self._finished_recv_thread.start()


        # Warm up the models
        self.warm_up_quick_model()
        #self.warm_up_reference_model()

        """
        # Initialize switching strategy
        self.switching_strategy = create_switching_strategy(
            switching_strategy, **self.strategy_kwargs
        )
        """
    
    def warm_up_reference_model(self):
        pass

    def warm_up_quick_model(self):
        # dummy call to warm up the quick model
        warmup_iter = 5
        input_text = ["Hi"]
        input_ids = [self.tokenizer.encode(text) for text in input_text]
        quick_sampling_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=warmup_iter, stop=[])
        for i, (text, input_id) in enumerate(zip(input_text, input_ids)):
            req = Req(
                rid=str(i),
                origin_input_text=text,
                origin_input_ids=input_id,
                sampling_params=quick_sampling_params,
                return_hidden_states=True,
                status="need",
            )
            req.display_progress = False
            try:
                self.req_sender.send_pyobj(req, flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.01)
                self.req_sender.send_pyobj(req)
        
        time.sleep(5)
    
    def get_rid(self):
        with self.rid_lock:
            rid = self.rid
            self.rid += 1
        return str(rid)

    def generate(
        self,
        input_ids: List[List[int]],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 100,
        record_generation: bool = False,
        print_tokens: bool = False,
        display_progress: bool = False,
    ) -> Union[
        List[str],
        Tuple[List[str], List[GenerationRecorder]]
    ]:
        """
        Generate text using dynamic model selection with SGLang models.

        Args:
            input_ids: A list of lists of token IDs for batch processing
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for nucleus sampling
            top_k: Top-k for sampling
            record_generation: If True, return both generated text and generation records
            print_tokens: Whether to print tokens during generation

        Returns:
            If record_generation is False: list of generated texts
            If record_generation is True: tuple of (list of generated texts, list of GenerationRecorders)
        """
        self.reset_cache_simple()
        #batch_input_ids = input_ids
        #batch_size = len(batch_input_ids)

        # Prepare sampling parameters for SGLang
        reference_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=1,
            stop=[],
        )

        # sglang will revise the output logits in-place if we set temperature > 0.0
        # so we set temperature to 0.0 here and sample in the decode_step
        quick_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop=[],
        )

        num_prompts = len(input_ids)
        rids = []
        for i in range(num_prompts):
            rids.append(self.get_rid())
        for i, input_id in enumerate(input_ids):
            req = Req(
                rid=rids[i],
                origin_input_text=self.tokenizer.decode(input_id),
                origin_input_ids=input_id,
                sampling_params=quick_sampling_params,
                return_hidden_states=True,
                status="need",
            )
            req.display_progress = display_progress
            # Send Req object to worker process via ZMQ
            try:
                self.req_sender.send_pyobj(req, flags=zmq.NOBLOCK)
            except zmq.Again:
                # If send buffer is full, wait and retry (simple backoff)
                time.sleep(0.01)
                self.req_sender.send_pyobj(req)
        # Wait until all rids appear in finished map
        results = []
        while True:
            all_done = all(rid in self.finished_reqs for rid in rids)
            if all_done:
                for rid in rids:
                    obj = self.finished_reqs[rid]
                    if isinstance(obj, dict):
                        # Print nicely and append decoded string
                        #print(f"Request {obj['rid']} finished")
                        #print(f"You: {obj.get('origin_input_text','')}")
                        #print(f"Bot: {self.tokenizer.decode(obj.get('output_ids', []))}")
                        #print("===")
                        results.append(obj)
                    else:
                        #print(f"Request {obj.rid} finished")
                        #print(f"You: {obj.origin_input_text}")
                        #print(f"Bot: {self.tokenizer.decode(obj.output_ids)}")
                        #print("===")
                        results.append(obj)
                break
            time.sleep(0.1)

        return results
    
    async def generate_one_request(
        self,
        input_id: List[int],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 100,
        display_progress: bool = False,
    ) -> Union[Dict, object]:
        """
        Async version of generate for a single request.
        Does not reset cache.
        """
        # Prepare sampling parameters for SGLang (Quick Model)
        # sglang will revise the output logits in-place if we set temperature > 0.0
        # so we set temperature to 0.0 here and sample in the decode_step
        quick_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stop=[],
        )

        rid = self.get_rid()
        
        req = Req(
            rid=rid,
            origin_input_text=self.tokenizer.decode(input_id),
            origin_input_ids=input_id,
            sampling_params=quick_sampling_params,
            return_hidden_states=True,
            status="need",
        )
        req.display_progress = display_progress

        # Send Req object via ZMQ
        # Note: ZMQ send is generally fast enough to be treated as sync, 
        # but we handle EAGAIN with async sleep just in case.
        try:
            self.req_sender.send_pyobj(req, flags=zmq.NOBLOCK)
        except zmq.Again:
            await asyncio.sleep(0.01)
            self.req_sender.send_pyobj(req)

        # Wait asynchronously until the rid appears in finished map
        while True:
            if rid in self.finished_reqs:
                return self.finished_reqs[rid]
            await asyncio.sleep(0.01) # Non-blocking wait

    async def generate_batch_requests(
        self,
        input_ids: List[List[int]],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 100,
        display_progress: bool = False,
    ):
        tasks = []
        
        # 1. Create all tasks
        for i, input_id in enumerate(input_ids):
            # Create task and start execution immediately
            task = asyncio.create_task(
                self.generate_one_request(
                    input_id=input_id,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    display_progress=display_progress
                )
            )
            tasks.append(task)

        results = []

        for i, future in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await future
                results.append(result)
            except Exception as e:
                pass

        return results

    def reset_cache_simple(self):
        """Reset the cache for the quick model"""
        try:
            req = Req(
                rid="RESET_CACHE",
                origin_input_text="",
                origin_input_ids=[],
                sampling_params=SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=1, stop=[]),
                return_hidden_states=False,
                status="RESET_CACHE"
            )
            self.req_sender.send_pyobj(req, flags=zmq.NOBLOCK)
        except zmq.Again:
            time.sleep(0.01)
            self.req_sender.send_pyobj(req)

    def shutdown(self):
        """Shut down the SGLang engines to free resources"""
        # Broadcast shutdown to quick workers (all TP ranks receive it)
        try:
            # ...existing code...
            self.req_sender.send_pyobj(Req(rid="SHUTDOWN", origin_input_text="", origin_input_ids=[], sampling_params=None, status="SHUTDOWN"))
        except Exception:
            pass

        # Call shutdown on servers to clean up their processes and threads
        if hasattr(self, "slm_server") and self.slm_server is not None:
            self.slm_server.shutdown()
        
        if hasattr(self, "llm_server") and self.llm_server is not None:
            self.llm_server.shutdown()

        # Optional: shutdown reference workers if they exist
        if hasattr(self, "reference_model_input_queues"):
            # ...existing code...
            pass

        # Stop finished-req SUB thread and close socket
        try:
            self._finished_recv_stop.set() # Set Event
            # ...existing code...
        except Exception:
            pass
        try:
            if hasattr(self, "req_sender") and self.req_sender is not None:
                self.req_sender.close(linger=0)
        except Exception:
            pass
        
        # Destroy ZMQ context to release sockets
        try:
            if hasattr(self, "zmq_ctx") and self.zmq_ctx is not None:
                self.zmq_ctx.term()
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
