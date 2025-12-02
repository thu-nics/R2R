import uuid
import torch
import time
import zmq
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, Union, List
import atexit
import os
import signal
import multiprocessing as mp
import socket
import torch.distributed as dist
import threading
import queue
from transformers import AutoTokenizer

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

class LLMServer:
    """LLM Server launched by SGLang"""
    def __init__(
        self,
        reference_sglang_kwargs, 
        quick_num_gpus: int,
        reference_num_gpus: int,
        reference_master_port: int | None = None,
        ready_queue: Optional[mp.Queue] = None,
        overlap_tp_schedule: bool = False,
        # New queues for inter-server communication
    ):
        self.is_reset_cache = False
        self.shutdown_loop = False
        self.batch = None
        self.new_reqs=[]
        self.reference_sglang_kwargs = reference_sglang_kwargs
        self.quick_num_gpus = quick_num_gpus if overlap_tp_schedule is False else 0
        self.reference_num_gpus = reference_num_gpus
        # Queue for signaling worker readiness to outside controller (e.g., SLDisaggregationSystem)
        self.ready_queue = ready_queue
        self._seq_to_slm = 0
        # Inter-server queues (outbound to SLM / inbound from SLM)
        self.queue_to_slm = mp.Queue()
        self._inbound_rank_queues = [mp.Queue() for _ in range(self.reference_num_gpus)]

        # ================ Inter-server PUB (LLM -> SLM) BEFORE workers =====================
        # (Notify PUB for finished reqs remains separate above; this PUB is for generic inter-server msgs)
        try:
            self._pub_slm = zmq.Context.instance().socket(zmq.PUB)
            self._pub_slm.setsockopt(zmq.LINGER, 0)
            self._pub_slm.setsockopt(zmq.SNDHWM, 100000)
            self.slm_recv_port = self._pub_slm.bind_to_random_port("tcp://127.0.0.1")
        except Exception as e:
            print(f"[LLMServer] Failed to bind PUB to SLM (early): {e}")
            self._pub_slm = None
            self.slm_recv_port = None

        self._send_slm_stop = threading.Event()
        def _send_slm_loop():
            time.sleep(0.05)
            while not self._send_slm_stop.is_set():
                try:
                    item = self.queue_to_slm.get(timeout=0.1)
                except Exception:
                    continue
                if item is None:
                    break
                if self._pub_slm is None:
                    continue
                try:
                    if not hasattr(item, "_seq"):
                        try:
                            setattr(item, "_seq", self._seq_to_slm)
                        except Exception:
                            # Fallback for dict-like payloads
                            if isinstance(item, dict):
                                item["_seq"] = self._seq_to_slm
                    # Increment after assignment
                    self._seq_to_slm += 1
                except Exception:
                    # Best-effort; continue even if _seq cannot be set
                    pass
                try:
                    self._pub_slm.send_pyobj(item)
                except Exception:
                    pass
        self._send_slm_thread = threading.Thread(target=_send_slm_loop, daemon=True)
        self._send_slm_thread.start()

        self._sub_from_slm = None
        self._recv_from_slm_stop = threading.Event()
        self._recv_from_slm_thread = None

        # Pick a dedicated master port for reference model's process group to avoid clashing with quick model (default 29500)
        if reference_master_port is None:
            # find a free port
            def _find_free_port():
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("127.0.0.1", 0))
                    return s.getsockname()[1]
            reference_master_port = _find_free_port()
        self.reference_master_port = reference_master_port

        # Ensure worker cleanup on interpreter exit and on signals
        atexit.register(self.shutdown)
        def _sig_handler(sig, frame):
            try:
                self.shutdown()
            finally:
                # Hard-exit to avoid hanging CUDA/NCCL threads
                os._exit(0)
        try:
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except Exception:
            # Some environments may disallow setting signals (e.g., Jupyter)
            pass

        print(f"Loading reference model {MODEL_DICT['reference']['model_name']}...")

        reference_server_args = ServerArgs(
            model_path=MODEL_DICT["reference"]["model_path"], 
            disable_cuda_graph=True, 
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            mem_fraction_static=0.8 if overlap_tp_schedule else 0.9,
            **reference_sglang_kwargs,
        )
        # Rank queues kept for future forwarded request injection (e.g., from SLM via inbound queue processing)
        self._stop_event = threading.Event()
        self._recv_thread = None  # Deprecated broadcast SUB thread removed.
        self.reference_model_procs = []
        for rank in range(reference_num_gpus):
            proc = mp.Process(
                target=self.reference_model_worker,
                args=(
                    rank, 
                    self.quick_num_gpus, 
                    self.reference_num_gpus, 
                    reference_server_args, 
                    self.reference_master_port, 
                    self.ready_queue, 
                    self._inbound_rank_queues[rank], 
                    self.queue_to_slm),
            )
            # Mark as daemon so that workers die when parent exits unexpectedly
            proc.daemon = True
            proc.start()
            self.reference_model_procs.append(proc)

    
    def process_new_requests(reqs: List[Req], scheduler: Scheduler):
        if len(reqs) == 0:
            return
        for req in reqs:
            if req.status in ("SHUTDOWN", "RESET_CACHE"):
                continue
            scheduler.waiting_queue.append(req)

    @staticmethod
    def reference_model_worker(rank, quick_num_gpus: int, world_size: int, server_args: ServerArgs, master_port: int = 29500, ready_queue: Optional[mp.Queue] = None, inbound_queue: Optional[mp.Queue] = None, outbound_queue: Optional[mp.Queue] = None):
        # Use a dedicated tcp init_method to avoid port collision with quick model's default 29500 store
        init_method = f"tcp://127.0.0.1:{master_port}"
        dist.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=world_size)
        torch.cuda.set_device(rank + quick_num_gpus)

        port_args = PortArgs.init_new(server_args)
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=rank+quick_num_gpus,
            tp_rank=rank,
            dp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
        )
        tokenizer = scheduler.tokenizer
        # Signal readiness
        if ready_queue is not None:
            try:
                # 仅 rank0 发送 tokenizer，其它 rank 发送 None 占位，方便主控端一次性获取
                ready_queue.put(("READY", rank, tokenizer if rank == 0 else None))
            except Exception as e:
                print(f"[rank {rank}] failed to put READY: {e}")
        
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
        LLMServer.simple_prepare_for_extend(scheduler.batch_not_need)
        scheduler.batch_not_need.multimodal_inputs = []
        scheduler.batch_not_need.output_ids = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )
        scheduler.batch_not_need.orig_seq_lens = torch.tensor([], dtype=torch.int64).to(
            scheduler.batch_not_need.device, non_blocking=True
        )

        print(f"Reference model worker {rank} started, waiting for requests...")
        
        try:
            idle_loops = 0
            while True:
                if inbound_queue is not None: # Process message from SLM
                    device = scheduler.batch_not_need.device
                    commit_msgs = LLMServer.fetch_and_align_inbound(
                        inbound_queue=inbound_queue,
                        rank=rank,
                        world_size=world_size,
                        device=device,
                    )
                    is_shutdown = [msg for msg in commit_msgs if getattr(msg, "status", "") == "SHUTDOWN"]
                    is_reset_cache = [msg for msg in commit_msgs if getattr(msg, "status", "") == "RESET_CACHE"]
                    commit_msgs = [msg for msg in commit_msgs if getattr(msg, "status", "") not in ("SHUTDOWN", "RESET_CACHE")]
                    if is_shutdown:
                        print(f"[reference rank{rank}] SHUTDOWN received (queue), exiting...")
                        break
                    elif is_reset_cache:
                        ok = scheduler.flush_cache()
                        print(f"[reference rank{rank}] Cache reset (queue): {ok}")

                    if commit_msgs:
                        LLMServer.process_result_from_slm(scheduler, commit_msgs)
                # For LLM, there is no need to process new reqs from rank_queue
                batch = scheduler.get_next_batch_to_run()
                if batch:
                    idle_loops = 0
                    result = scheduler.run_batch(batch)
                    LLMServer.process_batch_results(rank, batch, result, scheduler, outbound_queue)
                    scheduler.last_batch = batch
                else:
                    idle_loops += 1
                try:
                    if os.getppid() == 1:
                        print(f"[rank {rank}] parent process disappeared, exiting worker.")
                        break
                except Exception:
                    pass
        except BaseException as e:
            # Any unexpected error -> exit loop to avoid orphaned NCCL workers
            print(f"[rank {rank}] reference worker fatal error: {e}. Exiting loop.")
        finally:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as e:
                print(f"[rank {rank}] destroy_process_group/close socket error: {e}")

    @staticmethod
    def process_result_from_slm(scheduler: Scheduler, commit_msgs):
        new_token_ids = {}
        returned_rid_list = []
        finished_rid_list = torch.zeros(1000)
        if scheduler.batch_not_need is not None:
            req_already_prefilled = [req.rid for req in scheduler.batch_not_need.reqs]
        else:
            req_already_prefilled = []
        for waiting_req in commit_msgs:
            new_token_ids[waiting_req.rid] = waiting_req.new_token_ids
            returned_rid_list.append(waiting_req.rid)
            if waiting_req.status == "finished":
                finished_rid_list[waiting_req.rid] = 1
                continue
            if waiting_req.rid not in req_already_prefilled:
                origin_input_ids = waiting_req.new_token_ids
                origin_input_text = scheduler.tokenizer.decode(origin_input_ids)
                new_req = Req(
                    rid=waiting_req.rid,
                    origin_input_text=origin_input_text,
                    origin_input_ids=origin_input_ids,
                    sampling_params=waiting_req.sampling_params.derive_sampling_params(),
                    eos_token_ids=scheduler.model_config.hf_eos_token_id,
                    return_hidden_states=False,
                    vocab_size=scheduler.model_config.vocab_size,
                    status="need",
                    last_cached_loc=[],
                )
                if not hasattr(new_req, 'device'):
                    new_req.device = scheduler.batch_not_need.device
                scheduler.waiting_queue.append(new_req)

        if scheduler.batch_not_need is not None:
            not_keep_indices = []
            for i, req in enumerate(scheduler.batch_not_need.reqs):
                if finished_rid_list[req.rid] == 1:
                    scheduler.tree_cache.cache_finished_req(req)
                    continue
                if req.rid in returned_rid_list:
                    origin_input_ids = req.origin_input_ids+new_token_ids[req.rid]
                    origin_input_text = scheduler.tokenizer.decode(origin_input_ids)
                    new_req = Req(
                        rid=req.rid,
                        origin_input_text=origin_input_text,
                        origin_input_ids=origin_input_ids,
                        sampling_params=req.sampling_params,
                        return_hidden_states=False,
                        status="need",
                        last_cached_loc=req.last_cached_loc,
                    )
                    if not hasattr(new_req, 'device'):
                        new_req.device = scheduler.batch_not_need.device
                    scheduler.waiting_queue.append(new_req)
                else:
                    not_keep_indices.append(i)
            scheduler.batch_not_need.filter_batch(keep_indices=not_keep_indices)
        

    @staticmethod
    def fetch_and_align_inbound(inbound_queue: mp.Queue, rank: int, world_size: int, device: torch.device):
        """与 SLM 一致：按 _seq 做跨 rank 对齐，并将未对齐部分缓存在进程内 map 中。"""
        if not hasattr(LLMServer, "_inbound_pending_map"):
            LLMServer._inbound_pending_map = {}  # { id(queue): list }
        pending_map = LLMServer._inbound_pending_map
        qkey = id(inbound_queue)

        pending = pending_map.get(qkey, [])

        while True:
            try:
                item = inbound_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            pending.append(item)

        if not pending:
            return []

        local_max = getattr(pending[-1], "_seq", None)
        if local_max is None:
            out = list(pending)
            pending_map[qkey] = []
            return out

        t_local = torch.tensor([int(local_max)], device=device, dtype=torch.long)
        gather_buf = [torch.zeros_like(t_local) for _ in range(world_size)]
        dist.all_gather(gather_buf, t_local)
        commit_seq = min(int(t.item()) for t in gather_buf)

        commit_end = 0
        while commit_end < len(pending) and getattr(pending[commit_end], "_seq", -1) <= commit_seq:
            commit_end += 1

        if commit_end == 0:
            pending_map[qkey] = pending
            return []

        out = pending[:commit_end]
        pending_map[qkey] = pending[commit_end:]
        return out

    def shutdown(self):
        """Terminate reference model worker processes and close notify PUB.
        This is a best-effort cleanup for broken NCCL/TCPStore scenarios.
        """
        # stop recv thread
        try:
            if hasattr(self, "_stop_event"):
                self._stop_event.set()
            if hasattr(self, "_recv_thread") and self._recv_thread.is_alive():
                self._recv_thread.join(timeout=2)
        except Exception:
            pass
        # stop forwarder thread
        try:
            if hasattr(self, "_notify_queue"):
                try:
                    self._notify_queue.put_nowait(None)
                except Exception:
                    pass
        except Exception:
            pass
        # terminate worker processes
        if hasattr(self, "reference_model_procs"):
            for p in self.reference_model_procs:
                try:
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=5)
                        if p.is_alive():
                            # Force kill if it still refuses to exit
                            try:
                                p.kill()
                            except Exception:
                                pass
                except Exception:
                    pass
        # close PUB socket
        try:
            if hasattr(self, "notify_pub"):
                self.notify_pub.setsockopt(zmq.LINGER, 0)
                self.notify_pub.close(0)
        except Exception:
            pass
        # stop inter-server send thread
        try:
            if hasattr(self, "_send_slm_stop"):
                self._send_slm_stop.set()
            if hasattr(self, "queue_to_slm"):
                try:
                    self.queue_to_slm.put_nowait(None)
                except Exception:
                    pass
            if hasattr(self, "_send_slm_thread") and self._send_slm_thread.is_alive():
                self._send_slm_thread.join(timeout=2)
        except Exception:
            pass
        # stop inter-server recv thread
        try:
            if hasattr(self, "_recv_from_slm_stop"):
                self._recv_from_slm_stop.set()
            if hasattr(self, "_recv_from_slm_thread") and self._recv_from_slm_thread and self._recv_from_slm_thread.is_alive():
                self._recv_from_slm_thread.join(timeout=2)
        except Exception:
            pass
        # close inter-server sockets
        try:
            if hasattr(self, "_pub_slm") and self._pub_slm is not None:
                self._pub_slm.close(0)
        except Exception:
            pass
        try:
            if hasattr(self, "_sub_from_slm") and self._sub_from_slm is not None:
                self._sub_from_slm.close(0)
        except Exception:
            pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
    # NOTE: original recv_requests removed; using central SUB thread.

    # Removed _sub_recv_loop: LLM no longer directly SUBscribes to broadcast req_port.

    # Controller-triggered: start SUB to receive from SLM (SLM -> LLM)
    def start_llm_sub(self, port: int):
        if port is None:
            print('[LLMServer] start_llm_sub called with None port')
            return
        if self._sub_from_slm is not None:
            return
        ctx = zmq.Context.instance()
        try:
            self._sub_from_slm = ctx.socket(zmq.SUB)
            self._sub_from_slm.setsockopt(zmq.LINGER, 0)
            self._sub_from_slm.connect(f"tcp://127.0.0.1:{port}")
            self._sub_from_slm.setsockopt(zmq.SUBSCRIBE, b"")
        except Exception as e:
            print(f"[LLMServer] Failed to connect SUB from SLM: {e}")
            self._sub_from_slm = None
            return
        def _recv_loop():
            poller = zmq.Poller()
            poller.register(self._sub_from_slm, zmq.POLLIN)
            while not self._recv_from_slm_stop.is_set():
                try:
                    events = dict(poller.poll(timeout=50))
                except Exception:
                    continue
                if self._sub_from_slm in events and events[self._sub_from_slm] == zmq.POLLIN:
                    while True:
                        try:
                            msg = self._sub_from_slm.recv_pyobj(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        except Exception:
                            break
                        # Replicate to every rank's inbound queue to preserve identical ordering
                        for q in self._inbound_rank_queues:
                            try:
                                q.put_nowait(msg)
                            except Exception:
                                pass
        self._recv_from_slm_thread = threading.Thread(target=_recv_loop, daemon=True)
        self._recv_from_slm_thread.start()
        print(f"[LLMServer] SUB from SLM started on port {port}, loaded successfully")

    def enqueue_to_slm(self, obj):
        try:
            self.queue_to_slm.put_nowait(obj)
        except Exception:
            try:
                self.queue_to_slm.put(obj)
            except Exception:
                pass

    @staticmethod
    def _drain_rank_queue(scheduler: Scheduler, rank_queue: mp.Queue, rank: Optional[int] = None):
        out = []
        while True:
            try:
                item = rank_queue.get_nowait()
            except queue.Empty:
                break
            status = getattr(item, "status", "")
            if status == "SHUTDOWN":
                return -1
            elif status == "RESET_CACHE":
                ok = scheduler.flush_cache()
                print(f"[reference rank{scheduler.tp_rank}] Cache reset: {ok}")
                continue
            item.eos_token_id = scheduler.model_config.hf_eos_token_id
            item.vocab_size = scheduler.model_config.vocab_size
            out.append(item)
        return out
    
    def process_batch_results(rank: int, batch: ScheduleBatch, result, scheduler: Scheduler, outbound_queue: Optional[mp.Queue] = None):
        batch.output_ids = result.next_token_ids
        finished_reqs = []
        next_token_ids = result.next_token_ids.tolist()

        for req, next_token_id in zip(batch.reqs, next_token_ids):
            req.status = "notneed"
            waiting_req = WaitingReq(rid=req.rid,new_token_ids=[next_token_id],sampling_params=None)
            if rank == 0:
                try:
                    outbound_queue.put_nowait(waiting_req)
                except Exception:
                    # Fallback to blocking put if needed
                    try:
                        outbound_queue.put(waiting_req)
                    except Exception:
                        pass
    
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