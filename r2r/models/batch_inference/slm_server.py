import uuid
import torch
import time
import zmq
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, Union, List, Dict
import multiprocessing as mp
import torch.distributed as dist
import atexit
import signal
import os
import threading
import queue

from r2r.models.recorder import GenerationRecord, GenerationRecorder
from r2r.utils.config import (
    MODEL_DICT,
    QUICK_COLOR,
    REFERENCE_COLOR,
    RESET,
)
from r2r.utils.switching import create_switching_strategy
from r2r.utils.token_manager import SGLangTokenManager
from r2r.utils.dataclass import ModelOutputs
from r2r.utils.sampling import sample_token
from r2r.models.batch_inference.schedule_req import WaitingReq, SimpleSamplingParams

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ForwardMode, SamplingBatchInfo, write_req_to_token_pool_triton
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.utils import broadcast_pyobj

class SLMServer:
    """SLM Server launched by SGLang"""
    def __init__(
        self,
        quick_sglang_kwargs, 
        quick_num_gpus: int, 
        req_port: int,
        ready_queue: Optional[mp.Queue]=None,
        switching_strategy: str = "neural",
        strategy_kwargs: Dict = {},
        overlap_tp_schedule: bool = False,
    ):
        self.quick_waiting_line = []
        self.is_reset_cache = False
        self.shutdown_loop = False
        self.batch = None
        self.new_reqs=[]
        self.quick_sglang_kwargs = quick_sglang_kwargs
        self.quick_num_gpus = quick_num_gpus
        self.req_port = req_port
        self.ready_queue = ready_queue
        self._seq_counter = 0  # 全局单调递增序列号
        self.switching_strategy = switching_strategy
        self.strategy_kwargs = strategy_kwargs
        # Inter-server queues (outbound to LLM / inbound from LLM)
        self.queue_to_llm = mp.Queue()
        # Dedicated outbound sequence counter for messages sent to LLM
        self._seq_to_llm = 0
        # Per-rank inbound queues (LLM -> SLM); each rank consumes its own queue
        self._inbound_rank_queues = [mp.Queue() for _ in range(self.quick_num_gpus)]

        # ZMQ context (used for PUB/SUB sockets)
        self._ctx = zmq.Context.instance()

        # ================== New: PUB to system (SLM -> System) for finished reqs ==================
        # Create a dedicated mp.Queue to gather finished requests from workers,
        # and a thread to publish them over ZMQ in real time. Bind and expose port early,
        # before sending tokenizer back to controller.
        self._finished_reqs_queue: mp.Queue = mp.Queue()
        try:
            self._pub_finished = self._ctx.socket(zmq.PUB)
            self._pub_finished.setsockopt(zmq.LINGER, 0)
            self._pub_finished.setsockopt(zmq.SNDHWM, 100000)
            self.system_receive_port = self._pub_finished.bind_to_random_port("tcp://127.0.0.1")
        except Exception as e:
            print(f"[SLMServer] Failed to bind PUB to system: {e}")
            self._pub_finished = None
            self.system_receive_port = None
        # Sender thread for finished reqs
        self._send_finished_stop = threading.Event()
        def _send_finished_loop():
            time.sleep(0.05)
            while not self._send_finished_stop.is_set():
                try:
                    item = self._finished_reqs_queue.get(timeout=0.1)
                except Exception:
                    continue
                if item is None:
                    break
                if self._pub_finished is None:
                    continue
                try:
                    self._pub_finished.send_pyobj(item)
                except Exception:
                    pass
        self._send_finished_thread = threading.Thread(target=_send_finished_loop, daemon=True)
        self._send_finished_thread.start()

        # ================== Inter-server PUB (SLM -> LLM) BEFORE workers ==================
        try:
            self._pub_llm = self._ctx.socket(zmq.PUB)
            self._pub_llm.setsockopt(zmq.LINGER, 0)
            self._pub_llm.setsockopt(zmq.SNDHWM, 100000)
            # Bind now (B1) and expose port so controller can start LLM SUB later
            self.llm_recv_port = self._pub_llm.bind_to_random_port("tcp://127.0.0.1")
        except Exception as e:
            print(f"[SLMServer] Failed to bind PUB to LLM (early): {e}")
            self._pub_llm = None
            self.llm_recv_port = None

        # Send thread: pulls objects from queue_to_llm (start early so ready when workers spawn)
        self._send_llm_stop = threading.Event()
        def _send_llm_loop():
            time.sleep(0.05)
            while not self._send_llm_stop.is_set():
                try:
                    item = self.queue_to_llm.get(timeout=0.1)
                except Exception:
                    continue
                if item is None:
                    break
                if self._pub_llm is None:
                    continue
                # Ensure outbound message carries a sequence number for LLM-side alignment
                try:
                    if not hasattr(item, "_seq"):
                        try:
                            setattr(item, "_seq", self._seq_to_llm)
                        except Exception:
                            # Fallback for dict-like payloads
                            if isinstance(item, dict):
                                item["_seq"] = self._seq_to_llm
                    # Increment after assignment
                    self._seq_to_llm += 1
                except Exception:
                    # Best-effort; continue even if _seq cannot be set
                    pass
                try:
                    self._pub_llm.send_pyobj(item)
                except Exception:
                    pass
        self._send_llm_thread = threading.Thread(target=_send_llm_loop, daemon=True)
        self._send_llm_thread.start()

        # Placeholder for later SUB (LLM -> SLM) started by controller
        self._sub_from_llm = None
        self._recv_from_llm_thread = None
        self._recv_from_llm_stop = threading.Event()

        print(f"Loading quick model {MODEL_DICT['quick']['model_name']}...")
        # readiness queue
        self.ready_queue = ready_queue

        # Register atexit & signal handlers for safe shutdown like LLMServer
        atexit.register(self.shutdown)
        def _sig_handler(sig, frame):
            try:
                self.shutdown()
            finally:
                os._exit(0)
        try:
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except Exception:
            pass

        quick_server_args = ServerArgs(
            model_path=MODEL_DICT["quick"]["model_path"], 
            disable_cuda_graph=True, 
            disable_overlap_schedule=True,
            disable_radix_cache=False,
            mem_fraction_static=0.15 if overlap_tp_schedule else 0.9,
            **quick_sglang_kwargs,
        )
        # ==== New: per-rank queues and recv thread (central SUB) ====
        self._rank_queues: List[mp.Queue] = [mp.Queue() for _ in range(quick_num_gpus)]
        self._stop_event = threading.Event()
        self._recv_thread = threading.Thread(
            target=self._sub_recv_loop,
            args=(req_port,),
            daemon=True,
        )
        self._recv_thread.start()

        self.quick_model_procs = []
        for rank in range(quick_num_gpus):
            proc = mp.Process(
                target=self.quick_model_worker,
                args=(
                    rank, 
                    quick_num_gpus, 
                    quick_server_args, 
                    self._rank_queues[rank], 
                    self.ready_queue, 
                    self.switching_strategy, 
                    self.strategy_kwargs,
                    self._inbound_rank_queues[rank],  # per-rank inbound msgs
                    self.queue_to_llm,    # outbound (for potential direct worker usage),
                    self._finished_reqs_queue,  # finished reqs back to system
                ),
            )
            proc.start()
            self.quick_model_procs.append(proc)

    
    def process_new_requests(reqs: List[Req], scheduler: Scheduler):
        if len(reqs) == 0:
            return
        for req in reqs:
            if req.status in ("SHUTDOWN", "RESET_CACHE"):
                continue
            
            scheduler.waiting_queue.append(req)
        

    @staticmethod
    def quick_model_worker(rank, world_size: int, server_args: ServerArgs, rank_queue: mp.Queue, ready_queue: Optional[mp.Queue] = None, switching_strategy: str = "neural", strategy_kwargs: Dict = {}, inbound_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None, finished_queue: Optional[mp.Queue] = None):
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        port_args = PortArgs.init_new(server_args)
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=rank,
            tp_rank=rank,
            dp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
        )

        # Initialize switching strategy
        router = create_switching_strategy(
            switching_strategy, **strategy_kwargs
        )

        # Notify readiness after subscription
        tokenizer = scheduler.tokenizer
        if ready_queue is not None:
            try:
                ready_queue.put(("READY", rank, tokenizer if rank == 0 else None))
            except Exception as e:
                print(f"SLMServer Failed to send tokenizer from rank {rank}: {e}")

        print(f"Quick model worker {rank} started, waiting for requests...")

        pending_reqs: List[Req] = []

        scheduler.batch_not_need = ScheduleBatch.init_new(
            [],
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
            scheduler.tree_cache,
            scheduler.model_config,
            scheduler.enable_overlap,
            scheduler.spec_algorithm,
            scheduler.server_args.enable_custom_logit_processor,
        )
        SLMServer.simple_prepare_for_extend(scheduler.batch_not_need)
        scheduler.batch_not_need.multimodal_inputs = []
        scheduler.batch_not_need.output_ids = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )
        scheduler.batch_not_need.orig_seq_lens = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )
        scheduler.batch_not_need.output_ids = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )

        try:
            idle_loops = 0
            while True:
                if inbound_queue is not None: # Process message from LLM
                    device = scheduler.batch_not_need.device
                    # TODO: use sglang's way to receive message and broadcast to all ranks
                    commit_msgs = SLMServer.fetch_and_align_inbound(
                        inbound_queue=inbound_queue,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                    )
                    if commit_msgs:
                        SLMServer.process_result_from_llm(rank, scheduler, commit_msgs, finished_queue, outbound_queue)

                # TODO: use sglang's way to receive message and broadcast to all ranks
                reqs = SLMServer._drain_rank_queue(scheduler, rank_queue, rank, outbound_queue)
                if isinstance(reqs, int) and reqs == -1:
                    print(f"[quick rank{rank}] SHUTDOWN received (queue), exiting...")
                    break
                if reqs:
                    # 追加到本地缓冲（按 _seq 单调递增，无需排序）
                    pending_reqs.extend(reqs)

                device = scheduler.batch_not_need.device
                len_local = torch.tensor([len(pending_reqs)], device=device, dtype=torch.int64)
                gather_info = [torch.zeros_like(len_local) for _ in range(world_size)]
                dist.all_gather(gather_info, len_local)
                min_len = min(int(t.item()) for t in gather_info)

                # ==== 序列对齐提交阶段 ====
                # 只有在存在待提交请求时才做一次 all_gather
                if min_len > 0:
                    try:
                        local_max_seq = pending_reqs[-1]._seq
                    except AttributeError:
                        # 如果上游未正确打序列号，直接全部提交（回退）
                        commit_list = pending_reqs
                        pending_reqs = []
                    else:
                        # 收集每个 rank 已看到的最大序列
                        t_local = torch.tensor([local_max_seq], device=device, dtype=torch.long)
                        gather_buf = [torch.zeros_like(t_local) for _ in range(world_size)]
                        dist.all_gather(gather_buf, t_local)
                        commit_seq = min(int(t.item()) for t in gather_buf)
                        # 计算可提交前缀（所有 rank 至少已看到 commit_seq）
                        commit_end = 0
                        # pending_reqs 按序号递增
                        while commit_end < len(pending_reqs) and getattr(pending_reqs[commit_end], "_seq", -1) <= commit_seq:
                            commit_end += 1
                        if commit_end > 0:
                            commit_list = pending_reqs[:commit_end]
                            pending_reqs = pending_reqs[commit_end:]
                        else:
                            commit_list = []
                    if commit_list:
                        SLMServer.process_new_requests(commit_list, scheduler)
                

                batch = scheduler.get_next_batch_to_run()
                if batch:
                    idle_loops = 0
                    result = scheduler.run_batch(batch)
                    # Generate with quick model to get hidden states
                    batch, hidden_states, logits, next_token_ids = SLMServer.process_routing_input(batch, result)
                    # Create a ModelOutputs object for switching strategy
                    model_outputs = ModelOutputs(
                        logits=logits,
                        hidden_states=[hidden_states],  # dummy layer dimension
                        token=next_token_ids[:, None],
                    )
                    # Use switching strategy to decide which model to use for each input
                    model_choices = router.route(model_outputs)
                    # TODO: merge router into sglang

                    # Check if reference model is needed for any prompt
                    reference_needed = torch.any(model_choices).item()
                    if reference_needed:
                        req_to_send = []
                        for i, req in enumerate(batch.reqs):
                            if model_choices[i].item() == 1:
                                # TODO: send origin input to LLM to prefill prefix only
                                req.status = "notneed"
                                new_token_ids = []
                                if req.last_llm_loc is None:
                                    req.last_llm_loc = 0
                                    new_token_ids = req.origin_input_ids
                                new_token_ids = new_token_ids + req.output_ids[req.last_llm_loc:]
                                req.last_llm_loc = len(req.output_ids)
                                waiting_req=WaitingReq(
                                    rid=req.rid, 
                                    new_token_ids=new_token_ids, 
                                    sampling_params=SimpleSamplingParams(
                                        temperature = req.sampling_params.temperature, 
                                        top_k = req.sampling_params.top_k, 
                                        top_p = req.sampling_params.top_p, 
                                        max_new_tokens = 1
                                        )
                                    )
                                req_to_send.append(waiting_req)
                        
                        SLMServer.process_batch_results(batch, result, scheduler, finished_queue, outbound_queue, rank)
                        scheduler.check_batch_status(batch)
                        scheduler.last_batch=batch
                        if rank == 0:
                            for waiting_req in req_to_send:
                                outbound_queue.put_nowait(waiting_req)

                    else:
                        SLMServer.process_batch_results(batch, result, scheduler, finished_queue, outbound_queue, rank)
                        scheduler.last_batch=batch
                else:
                    idle_loops += 1
                    #time.sleep(0.003)
        finally:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[rank {rank}] destroy_process_group error: {e}")

    @staticmethod
    def process_result_from_llm(rank: int, scheduler: Scheduler, commit_msgs, finished_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None):
        better_token_ids = {}
        returned_rid_list = []
        for waiting_req in commit_msgs:
            better_token_ids[waiting_req.rid] = waiting_req.new_token_ids[-1]
            returned_rid_list.append(waiting_req.rid)
        keep_indices = []
        not_keep_indices = []
        finished_reqs = []
        if scheduler.batch_not_need is not None:
            if scheduler.last_batch is None:
                return
            else:
                scheduler.last_batch.merge_batch(scheduler.batch_not_need)
            output_ids_list = []
            for i, req in enumerate(scheduler.last_batch.reqs):
                if req.rid in returned_rid_list:
                    if better_token_ids[req.rid] in scheduler.model_config.hf_eos_token_id:
                        scheduler.abort_request(AbortReq(req.rid))
                    req.output_ids.append(better_token_ids[req.rid])
                    req.status = "need"
                    req.check_finished()
                    if req.finished():
                        scheduler.tree_cache.cache_finished_req(req)
                        finished_reqs.append(req)
                    keep_indices.append(i)
                elif req.status == "need":
                    keep_indices.append(i)
                output_ids_list.append(req.output_ids[-1] if req.output_ids else 1)
            
            scheduler.last_batch.output_ids = torch.tensor(output_ids_list, dtype=torch.int64).to(
                scheduler.last_batch.device, non_blocking=True
            )
            
            scheduler.last_batch.filter_batch(keep_indices=keep_indices)
            
            for i, req in enumerate(scheduler.batch_not_need.reqs):
                if req.status == "notneed" and req.rid not in returned_rid_list:
                    not_keep_indices.append(i)
            scheduler.batch_not_need.filter_batch(keep_indices=not_keep_indices)
            if finished_reqs and rank == 0:
                SLMServer.process_finished_requests(finished_reqs, scheduler.tokenizer, finished_queue, outbound_queue)
    
    @staticmethod
    def simple_prepare_for_extend(batch: ScheduleBatch):
        global end_of_cache_loc
        batch.forward_mode = ForwardMode.EXTEND

        # Allocate req slots
        bs = len(batch.reqs)
        req_pool_indices = [req.rid for req in batch.reqs]

        # Init tensors
        reqs = batch.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = [len(r.fill_ids) for r in reqs]
        prefix_lens = [len(r.prefix_indices) for r in reqs]
        extend_lens = [r.extend_input_len for r in reqs]
        req_pool_indices_tensor = torch.tensor(req_pool_indices, dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        input_ids_tensor = torch.tensor(sum(input_ids, []), dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        prefix_lens_tensor = torch.tensor(prefix_lens, dtype=torch.int64).to(
            batch.device, non_blocking=True
        )
        extend_lens_tensor = seq_lens_tensor - prefix_lens_tensor
        for i, (req, seq_len, pre_len) in enumerate(zip(reqs, seq_lens, prefix_lens)):
            req.req_pool_idx = req_pool_indices[i]
            assert seq_len - pre_len == req.extend_input_len
            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False
        # Allocate memory for multiple sequences
        out_cache_locs = []
        for i in range(bs):
            start = end_of_cache_loc
            end_of_cache_loc += extend_lens[i]
            end = start + extend_lens[i]
            out_cache_loc = torch.arange(
                start=start,
                end=end,
                dtype=torch.int64,
                device=batch.device,
            )
            out_cache_locs.append(out_cache_loc)
        
        if len(out_cache_locs) == 0:
            out_cache_loc = None
        else:
            out_cache_loc = torch.cat(out_cache_locs) if len(out_cache_locs) > 1 else out_cache_locs[0]

        # Set fields
        batch.input_ids = input_ids_tensor
        batch.req_pool_indices = req_pool_indices_tensor
        batch.seq_lens = seq_lens_tensor
        batch.out_cache_loc = out_cache_loc
        batch.input_embeds = None
        batch.seq_lens_sum = sum(seq_lens)
        if batch.return_logprob:
            batch.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            batch.token_ids_logprobs = [r.token_ids_logprob for r in reqs]
        batch.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        batch.extend_num_tokens = extend_num_tokens
        batch.prefix_lens = prefix_lens
        batch.extend_lens = extend_lens
        if out_cache_loc is not None:
            write_req_to_token_pool_triton[(bs,)](
                batch.req_to_token_pool.req_to_token,
                req_pool_indices_tensor,
                prefix_lens_tensor,
                seq_lens_tensor,
                extend_lens_tensor,
                out_cache_loc,
                batch.req_to_token_pool.req_to_token.shape[1],
            )
        # Build sampling info
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch,
            batch.model_config.vocab_size,
        )

    @staticmethod
    def fetch_and_align_inbound(inbound_queue: mp.Queue, rank: int, world_size: int, device: torch.device):
        """Drain inbound_queue and return the largest cross-rank aligned prefix by _seq.

        Persist uncommitted tail in a per-process map, keyed by queue id,
        so that unaligned items are not lost across calls.
        """
        # Initialize per-process pending map on the class
        if not hasattr(SLMServer, "_inbound_pending_map"):
            SLMServer._inbound_pending_map = {}  # { id(queue): list }
        pending_map = SLMServer._inbound_pending_map
        qkey = id(inbound_queue)

        # Get or init pending buffer for this queue
        pending = pending_map.get(qkey, [])
        # Drain non-blocking
        while True:
            try:
                item = inbound_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            pending.append(item)
        
        len_local = torch.tensor([len(pending)], device=device, dtype=torch.int64)
        gather_len = [torch.zeros_like(len_local) for _ in range(world_size)]
        dist.all_gather(gather_len, len_local)
        min_len = min(int(t.item()) for t in gather_len)
        if min_len == 0:
            pending_map[qkey] = pending
            return []

        # Determine local max seq; if absent, commit all
        local_max = getattr(pending[-1], "_seq", None)
        if local_max is None:
            out = list(pending)
            pending_map[qkey] = []
            return out

        # Cross-rank alignment by min gathered seq
        t_local = torch.tensor([int(local_max)], device=device, dtype=torch.long)
        gather_buf = [torch.zeros_like(t_local) for _ in range(world_size)]
        dist.all_gather(gather_buf, t_local)
        commit_seq = min(int(t.item()) for t in gather_buf)

        # Find longest aligned prefix
        commit_end = 0
        while commit_end < len(pending) and getattr(pending[commit_end], "_seq", -1) <= commit_seq:
            commit_end += 1
        if commit_end == 0:
            # Keep all in pending for next round
            pending_map[qkey] = pending
            return []

        out = pending[:commit_end]
        # Save uncommitted tail back
        pending_map[qkey] = pending[commit_end:]
        return out

    def process_routing_input(batch: ScheduleBatch, result):
        device = batch.seq_lens.device
        extend_lens = torch.tensor(batch.extend_lens, device=device)
        batch_size = batch.batch_size()
        is_prefill = (result.logits_output.hidden_states.shape[0] != batch_size)

        if is_prefill:
            # For prefill, use cumsum of extend_lens to get correct indices
            hidden_indices = torch.cumsum(extend_lens, dim=0) - 1
        else:
            # For decode, use sequential indices
            hidden_indices = torch.arange(batch_size, device=device)

        # Get hidden states for the relevant positions
        hidden_states = result.logits_output.hidden_states[hidden_indices, :][:, None, :] # batch_size, 1, hidden_size
        logits = result.logits_output.next_token_logits # batch_size, vocab_size
        next_token_ids = result.next_token_ids # batch_size

        return batch, hidden_states, logits[:, None, :], next_token_ids

    # ================= New central SUB thread & queue helpers =================
    def _sub_recv_loop(self, req_port: int):
        """SUB loop in main process: receive Req objects and replicate to rank queues."""
        ctx = zmq.Context.instance()
        sub_socket = ctx.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.LINGER, 0)
        sub_socket.connect(f"tcp://127.0.0.1:{req_port}")
        sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        poller = zmq.Poller()
        poller.register(sub_socket, zmq.POLLIN)
        while not self._stop_event.is_set():
            try:
                events = dict(poller.poll(timeout=20))
            except KeyboardInterrupt:
                break
            if events.get(sub_socket) == zmq.POLLIN:
                while True:
                    try:
                        req = sub_socket.recv_pyobj(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        setattr(req, "_seq", self._seq_counter)
                        self._seq_counter += 1
                    except Exception:
                        pass
                    # 分发到所有 rank 队列
                    for q in self._rank_queues:
                        try:
                            q.put(req)
                        except Exception:
                            pass
                    if getattr(req, "status", "") == "SHUTDOWN":
                        self._stop_event.set()
                        break
        try:
            sub_socket.close(0)
        except Exception:
            pass

    @staticmethod
    def _drain_rank_queue(scheduler: Scheduler, rank_queue: mp.Queue, rank: Optional[int]=None, outbound_queue: Optional[mp.Queue]=None):
        out = []
        while True:
            try:
                item = rank_queue.get_nowait()
            except queue.Empty:
                break
            status = getattr(item, "status", "")
            if status == "SHUTDOWN":
                if rank == 0:
                    outbound_queue.put_nowait(WaitingReq(status="SHUTDOWN",))
                return -1
            elif status == "RESET_CACHE":
                ok = scheduler.flush_cache()
                print(f"[quick rank{scheduler.gpu_id}] Cache reset: {ok}")
                if rank == 0:
                    outbound_queue.put_nowait(WaitingReq(rid=-1, new_token_ids=[], status="RESET_CACHE",))
                continue
            item.eos_token_ids = scheduler.model_config.hf_eos_token_id
            item.vocab_size = scheduler.model_config.vocab_size
            out.append(item)
        return out

    # NOTE: original recv_requests removed in favor of central queue distribution.
    
    def process_batch_results(batch: ScheduleBatch, result, scheduler: Scheduler, finished_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None, rank: int = 0):
        batch.output_ids = result.next_token_ids
        finished_reqs = []

        for req, next_token_id in zip(batch.reqs, result.next_token_ids):
            if req.status == "notneed":
                continue
            if next_token_id in scheduler.model_config.hf_eos_token_id:
                scheduler.abort_request(AbortReq(req.rid))
            req.output_ids.append(next_token_id.item())
            req.check_finished()
            if req.finished():
                scheduler.tree_cache.cache_finished_req(req)
                finished_reqs.append(req)

        if len(finished_reqs) > 0 and rank == 0:
            SLMServer.process_finished_requests(finished_reqs, scheduler.tokenizer, finished_queue, outbound_queue)


    def process_finished_requests(finished_reqs: List[Req], tokenizer, finished_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None):
        """Process finished requests, e.g., logging or updating status."""
        for req in finished_reqs:
            #print(f"Request {req.rid} finished")
            #print(f"You: {req.origin_input_text}")
            #print(f"Bot: {tokenizer.decode(req.output_ids)}")
            #print("===")
            # Enqueue to system if queue is provided (send a lightweight serializable payload)
            outbound_queue.put_nowait(WaitingReq(
                rid=req.rid, 
                new_token_ids=[], 
                sampling_params=SimpleSamplingParams(),
                status="finished",
            ))
            if finished_queue is not None:
                payload = {
                    "rid": getattr(req, "rid", None),
                    "origin_input_text": getattr(req, "origin_input_text", None),
                    "origin_input_ids": list(getattr(req, "origin_input_ids", [])),
                    "output_ids": list(getattr(req, "output_ids", [])),
                    "status": "finished",
                }
                try:
                    finished_queue.put_nowait(payload)
                except Exception:
                    try:
                        finished_queue.put(payload)
                    except Exception:
                        pass

    def shutdown(self):
        """Gracefully terminate quick model workers (mirrors LLMServer)."""
        # stop recv thread
        try:
            if hasattr(self, "_stop_event"):
                self._stop_event.set()
            if hasattr(self, "_recv_thread") and self._recv_thread.is_alive():
                self._recv_thread.join(timeout=2)
        except Exception:
            pass
        # stop finished-reqs send thread
        try:
            if hasattr(self, "_send_finished_stop"):
                self._send_finished_stop.set()
            if hasattr(self, "_finished_reqs_queue"):
                try:
                    self._finished_reqs_queue.put_nowait(None)
                except Exception:
                    pass
            if hasattr(self, "_send_finished_thread") and self._send_finished_thread.is_alive():
                self._send_finished_thread.join(timeout=2)
        except Exception:
            pass
        # stop inter-server send thread
        try:
            if hasattr(self, "_send_llm_stop"):
                self._send_llm_stop.set()
            if hasattr(self, "queue_to_llm"):
                try:
                    self.queue_to_llm.put_nowait(None)
                except Exception:
                    pass
            if hasattr(self, "_send_llm_thread") and self._send_llm_thread.is_alive():
                self._send_llm_thread.join(timeout=2)
        except Exception:
            pass
        # stop inter-server recv thread
        try:
            if hasattr(self, "_recv_from_llm_stop"):
                self._recv_from_llm_stop.set()
            if hasattr(self, "_recv_from_llm_thread") and self._recv_from_llm_thread and self._recv_from_llm_thread.is_alive():
                self._recv_from_llm_thread.join(timeout=2)
        except Exception:
            pass
        # close sockets
        try:
            if hasattr(self, "_pub_llm") and self._pub_llm is not None:
                self._pub_llm.close(0)
        except Exception:
            pass
        try:
            if hasattr(self, "_sub_from_llm") and self._sub_from_llm is not None:
                self._sub_from_llm.close(0)
        except Exception:
            pass
        try:
            if hasattr(self, "_pub_finished") and self._pub_finished is not None:
                self._pub_finished.close(0)
        except Exception:
            pass
        if hasattr(self, "quick_model_procs"):
            for p in self.quick_model_procs:
                try:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            try:
                                p.kill()
                            except Exception:
                                pass
                except Exception:
                    pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    # ===== Controller-triggered: start SUB to receive from LLM (LLM -> SLM) =====
    def start_slm_sub(self, port: int):
        if port is None:
            print('[SLMServer] start_slm_sub called with None port')
            return
        if self._sub_from_llm is not None:
            return  # already started
        ctx = zmq.Context.instance()
        try:
            self._sub_from_llm = ctx.socket(zmq.SUB)
            self._sub_from_llm.setsockopt(zmq.LINGER, 0)
            self._sub_from_llm.connect(f"tcp://127.0.0.1:{port}")
            self._sub_from_llm.setsockopt(zmq.SUBSCRIBE, b"")
        except Exception as e:
            print(f"[SLMServer] Failed to connect SUB from LLM: {e}")
            self._sub_from_llm = None
            return
        def _recv_loop():
            poller = zmq.Poller()
            poller.register(self._sub_from_llm, zmq.POLLIN)
            while not self._recv_from_llm_stop.is_set():
                try:
                    events = dict(poller.poll(timeout=50))
                except Exception:
                    continue
                if self._sub_from_llm in events and events[self._sub_from_llm] == zmq.POLLIN:
                    while True:
                        try:
                            msg = self._sub_from_llm.recv_pyobj(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        except Exception:
                            break
                        # Replicate to every rank's inbound queue to preserve identical ordering
                        for i, q in enumerate(self._inbound_rank_queues):
                            try:
                                q.put_nowait(msg)
                            except Exception as e:
                                print(f"[SLMServer rank {i}] Failed to enqueue inbound msg to rank queue. Exception:{e}")
                                pass
        self._recv_from_llm_thread = threading.Thread(target=_recv_loop, daemon=True)
        self._recv_from_llm_thread.start()
        print(f"[SLMServer] SUB from LLM started on port {port}, loaded successfully")

    # Helper for user to enqueue outbound messages (optional direct use)
    def enqueue_to_llm(self, obj):
        try:
            self.queue_to_llm.put_nowait(obj)
        except Exception:
            try:
                self.queue_to_llm.put(obj)
            except Exception:
                pass
        