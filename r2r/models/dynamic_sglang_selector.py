import uuid
import torch
from tqdm import tqdm
from typing import Optional, Tuple, Union, List
import multiprocessing as mp
import torch.distributed as dist

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

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch, ForwardMode, SamplingBatchInfo, write_req_to_token_pool_triton
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import PortArgs, ServerArgs


class DynamicSimpleSGLangSelector:
    """Dynamic model selection using SGLang models"""

    def __init__(
        self,
        device: Optional[str] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        switching_strategy: str = "neural",
        strategy_kwargs: Optional[dict] = None,
        is_record: bool = False,
        sglang_kwargs: Optional[dict] = None,
        is_logits_processor: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.strategy_kwargs = strategy_kwargs or {}
        self.switching_strategy_name = switching_strategy
        self.is_record = is_record
        
        # Combine default with provided kwargs
        quick_sglang_kwargs = {**(sglang_kwargs or {})}
        reference_sglang_kwargs = {**(sglang_kwargs or {})}
        
        self.num_gpus = reference_sglang_kwargs.get("tp_size", torch.cuda.device_count())
        self.world_size = self.num_gpus

        # Create dictionary to store recorders
        self.generation_records = {}

        # Currently only support tp_size=1 for quick model
        quick_sglang_kwargs["tp_size"] = 1 
        reference_sglang_kwargs["tp_size"] = self.world_size
        assert self.num_gpus >= 2, f"Using {self.num_gpus} GPUs for SGLang, expected larger than 2."
        print(f"Using {self.num_gpus} GPUs for SGLang, with {self.world_size} for reference and 1 for quick.")

        # Initialize SGLang models
        print(f"Loading quick model {MODEL_DICT['quick']['model_name']}...")

        self.quick_server_args = ServerArgs(
            model_path=MODEL_DICT["quick"]["model_path"], 
            disable_cuda_graph=False, 
            disable_overlap_schedule=True,
            disable_radix_cache=False,
            mem_fraction_static=MODEL_DICT["quick"]["mem_fraction_static"],
            **quick_sglang_kwargs,
        )
        quick_port_args = PortArgs.init_new(self.quick_server_args)
        self.quick_scheduler = Scheduler(
            server_args=self.quick_server_args,
            port_args=quick_port_args,
            gpu_id=1,
            tp_rank=0,
            dp_rank=0,
        )
        # Load tokenizer
        self.tokenizer = self.quick_scheduler.tokenizer
        # # warm up the quick model
        self.warm_up_quick_model()

        print(f"Loading reference model {MODEL_DICT['reference']['model_name']}...")
        self.reference_server_args = ServerArgs(
            model_path=MODEL_DICT["reference"]["model_path"],
            disable_cuda_graph=True, 
            disable_overlap_schedule=True,
            disable_radix_cache=False,
            mem_fraction_static=MODEL_DICT["reference"]["mem_fraction_static"],
            **reference_sglang_kwargs,
        )

        self.reference_model_input_queues = [mp.Queue() for _ in range(self.world_size)]
        self.reference_model_ack_queues = [mp.Queue() for _ in range(self.world_size)]
        self.reference_model_output_queue = mp.Queue()

        self.reference_model_procs = []
        for rank in range(self.world_size):
            proc = mp.Process(
                target=self.reference_model_worker,
                args=(rank, self.world_size, self.reference_server_args, self.reference_model_input_queues, self.reference_model_output_queue, self.reference_model_ack_queues[rank]),
            )
            proc.start()
            self.reference_model_procs.append(proc)

        # Initialize prefix indices list for reference model
        self.reference_prefix_indices_list = []

        # warm up the reference model
        self.warm_up_reference_model()

        # Initialize switching strategy
        self.switching_strategy = create_switching_strategy(
            switching_strategy, **self.strategy_kwargs
        )
    
    def warm_up_reference_model(self):
        # dummy call to warm up the reference model
        if not self.reference_prefix_indices_list:
            self.reference_prefix_indices_list.append([])
        
        test_input = [self.tokenizer.encode("Hi")]
        
        self.extend_step(
            input_ids=test_input,
            input_indices=[0],  
            sampling_params=SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_new_tokens=1, stop=[])
        )

    def warm_up_quick_model(self):
        # dummy call to warm up the quick model
        warmup_iter = 5
        req = Req(
            rid=str(uuid.uuid4()),
            origin_input_text="Hi",
            origin_input_ids=self.quick_scheduler.tokenizer.encode("Hi"),
            sampling_params=SamplingParams(
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                max_new_tokens=warmup_iter,
                stop=[]
            ),
            eos_token_ids=self.quick_scheduler.model_config.hf_eos_token_id,
            return_hidden_states=True
        )
        req.sampling_params.normalize(None)
        self.quick_scheduler.waiting_queue.append(req)
        for _ in range(warmup_iter):
            batch = self.quick_scheduler.get_next_batch_to_run()
            if batch is None:
                break
            result = self.quick_scheduler.run_batch(batch)
            next_token_ids = result.next_token_ids
            self.quick_scheduler.last_batch = batch
        for req in batch.reqs:
            self.quick_scheduler.abort_request(req)
            req.check_finished()
            if req.finished():
                self.quick_scheduler.tree_cache.cache_finished_req(req)
        self.quick_scheduler.last_batch = batch

    @staticmethod
    def reference_model_worker(rank, world_size: int, server_args: ServerArgs, input_queues: List[mp.Queue], output_queue: mp.Queue, ack_queue: mp.Queue):

        # initialize the process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

        global end_of_cache_loc
        end_of_cache_loc = 0

        input_queue = input_queues[rank]
        port_args = PortArgs.init_new(server_args)
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=rank,
            tp_rank=rank,
            dp_rank=0,
        )

        while True:
            reqs: Union[List[Req], int] = input_queue.get()
            if isinstance(reqs, int):
                # terminate the process
                break
            elif isinstance(reqs, str):
                if reqs == "RESET_CACHE":
                    # reset the cache
                    end_of_cache_loc = 0
                    ack_queue.put(end_of_cache_loc)
                    continue
            else:
                new_batch = ScheduleBatch.init_new(
                    reqs,
                    scheduler.req_to_token_pool,
                    scheduler.token_to_kv_pool_allocator,
                    scheduler.tree_cache,
                    scheduler.model_config,
                    scheduler.enable_overlap,
                    scheduler.spec_algorithm,
                    scheduler.server_args.enable_custom_logit_processor,
                )
                DynamicSimpleSGLangSelector.simple_prepare_for_extend(new_batch)
                batch = new_batch.get_model_worker_batch()
                _, next_token_ids = scheduler.tp_worker.forward_batch_generation(batch)
                next_token_ids_list = next_token_ids.tolist()

                if rank == 0:
                    output_queue.put(next_token_ids_list)

    def init_model_switching_strategy(self):
        """Initialize or reinitialize the model switching strategy with stored parameters"""
        self.switching_strategy = create_switching_strategy(
            self.switching_strategy_name, **self.strategy_kwargs
        )

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

    def extend_step(self, input_ids: List[List[int]], input_indices: List[int], sampling_params: SamplingParams) -> List[int]:
        """
        Extend the input ids using the reference model
        """
        subset_batch_size = len(input_ids)
        input_texts = self.tokenizer.batch_decode(input_ids)
        reqs = []
        for i, (input_text, input_id) in enumerate(zip(input_texts, input_ids)):
            req = Req(
                rid=input_indices[i],
                origin_input_text=input_text,
                origin_input_ids=input_id,
                sampling_params=sampling_params,
                eos_token_ids=self.quick_scheduler.model_config.hf_eos_token_id, # noqa
                return_hidden_states=False
            )
            req.sampling_params.normalize(None) # disable str-based stop token
            req.prefix_indices = self.reference_prefix_indices_list[input_indices[i]]
            req.fill_ids = input_id
            req.extend_input_len = len(input_id) - len(self.reference_prefix_indices_list[input_indices[i]])
            reqs.append(req)

        for q in self.reference_model_input_queues:
            q.put_nowait(reqs)
        next_token_ids = self.reference_model_output_queue.get()

        # Update prefix indices for each prompt
        for i in range(subset_batch_size):
            self.reference_prefix_indices_list[input_indices[i]]=list(range(len(input_ids[i])))

        return next_token_ids

    def decode_step(self, scheduler: Scheduler, temperature: float = 0.0, top_p: float = 1.0, top_k: int = -1):
        """
        Decode one step using the quick model
        
        Args:
            scheduler: The scheduler to use
            
        Returns:
            batch: The batch to use
            hidden_states: The hidden states from the quick model, shape (batch_size, seq_len, hidden_size)
            logits: The logits from the quick model, shape (batch_size, 1, vocab_size)
            next_token_ids: The next token ids from the quick model, shape (batch_size)
        """
        batch = scheduler.get_next_batch_to_run()
        result = scheduler.run_batch(batch)

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
        next_token_ids = sample_token(logits, temperature=temperature, top_p=top_p, top_k=top_k)

        return batch, hidden_states, logits[:, None, :], next_token_ids

    def update_output_ids(self, batch: ScheduleBatch, scheduler: Scheduler, next_token_ids: List[int]):
        """Update the output ids for the batch"""
        batch.output_ids = next_token_ids

        for req, next_token_id in zip(batch.reqs, next_token_ids):
            if next_token_id in self.quick_scheduler.model_config.hf_eos_token_id:
                scheduler.abort_request(req)
            req.output_ids.append(next_token_id.item())
            req.check_finished()
            if req.finished():
                scheduler.tree_cache.cache_finished_req(req)

        scheduler.last_batch = batch

    def generate(
        self,
        input_ids: List[List[int]],
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 100,
        record_generation: bool = False,
        print_tokens: bool = False,
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
        batch_input_ids = input_ids
        batch_size = len(batch_input_ids)
        self.reference_prefix_indices_list = [[] for _ in range(batch_size)]

        # Setup recorders if recording is enabled
        recorders = (
            [GenerationRecorder() for _ in range(batch_size)]
            if record_generation
            else None
        )

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

        # Use uid to generate unique ids for each request
        rids = [str(uuid.uuid4()) for _ in range(batch_size)]
        for i, input_id in enumerate(batch_input_ids):
            req = Req(
                rid=rids[i],
                origin_input_text=self.tokenizer.decode(input_id),
                origin_input_ids=input_id,
                sampling_params=quick_sampling_params,
                eos_token_ids=self.quick_scheduler.model_config.hf_eos_token_id,
                return_hidden_states=True
            )
            self.quick_scheduler.waiting_queue.append(req)

        # Initialize token manager with tokenized inputs
        token_manager = SGLangTokenManager(
            batch_input_ids, self.tokenizer, max_new_tokens=max_new_tokens
        )

        # Generate tokens one by one until all prompts reach EOS or max limit
        position = 0

        if not print_tokens:
            # Create tqdm progress bar for token generation
            pbar = tqdm(total=max_new_tokens, desc="Dynamic SGLang: Generating tokens", leave=True)
        while not token_manager.is_generation_complete() and position < max_new_tokens:
            if not print_tokens:
                pbar.update(1)

            active_count = token_manager.get_active_count()

            if active_count < 1:
                break

            # Generate with quick model to get hidden states
            batch, hidden_states, logits, next_token_ids = self.decode_step(self.quick_scheduler, temperature=temperature, top_p=top_p, top_k=top_k)

            # Create a ModelOutputs object for switching strategy
            model_outputs = ModelOutputs(
                logits=logits,
                hidden_states=[hidden_states],  # dummy layer dimension
                token=next_token_ids[:, None],
            )

            # Use switching strategy to decide which model to use for each input
            model_choices = self.switching_strategy.route(model_outputs)

            # Check if reference model is needed for any prompt
            reference_needed = torch.any(model_choices).item()

            if reference_needed:
                # Get indices of inputs that need reference model as a list
                reference_indices = torch.where(model_choices == 1)[0].tolist()
                active_to_original = token_manager.get_active_index()
                reference_original_indices = [active_to_original[i] for i in reference_indices]
                reference_input_ids = token_manager.fetch_active_input_ids(reference_indices)

                # Generate with reference model for inputs that need it
                reference_outputs = self.extend_step(
                    input_ids=reference_input_ids,
                    input_indices=reference_original_indices,
                    sampling_params=reference_sampling_params,
                )
                for i, reference_output_token in enumerate(reference_outputs):
                    next_token_ids[reference_indices[i]] = reference_output_token

                # Combine outputs based on model choices
                # Record if needed
                if record_generation and recorders:
                    for i in range(active_count):
                        if model_choices[i].item() == 1:  # Use reference model
                            # update next token ids
                            source_model = "reference"
                            param_size = float(MODEL_DICT["reference"]["param"])
                        else:  # Use quick model
                            source_model = "quick"
                            param_size = float(MODEL_DICT["quick"]["param"])

                        token = next_token_ids[i].item()
                        token_str = self.tokenizer.decode(token)

                        # Add record
                        active_indicies = token_manager.get_active_index()
                        seq_idx = active_indicies[i]
                        recorders[seq_idx].add_record(
                            GenerationRecord(
                                token_id=token,
                                token_str=token_str,
                                source_model=source_model,
                                position=position,
                                batch_id=seq_idx,
                                param_size=param_size,
                            )
                        )

                        # Print tokens if requested
                        if (
                            print_tokens and seq_idx == 0
                        ):  # Only print for the first batch
                            color = (
                                REFERENCE_COLOR
                                if source_model == "reference"
                                else QUICK_COLOR
                            )
                            print(f"{color}{token_str}{RESET}", end="", flush=True)

            else:
                # Use quick model for all outputs
                # Record if needed
                if record_generation and recorders:
                    for i in range(active_count):
                        token = next_token_ids[i].item()
                        token_str = self.tokenizer.decode(token)

                        # Find original batch index
                        seq_idx = token_manager.get_active_index()[i]

                        # Add record
                        recorders[seq_idx].add_record(
                            GenerationRecord(
                                token_id=token,
                                token_str=token_str,
                                source_model="quick",
                                position=position,
                                batch_id=seq_idx,
                                param_size=float(MODEL_DICT["quick"]["param"]),
                            )
                        )

                        # Print tokens if requested
                        if (
                            print_tokens and seq_idx == 0
                        ):  # Only print for the first batch
                            print(f"{QUICK_COLOR}{token_str}{RESET}", end="", flush=True)

            # Update token manager with final outputs
            self.update_output_ids(batch, self.quick_scheduler, next_token_ids)
            token_manager.update_sequences_direct([token_id.item() for token_id in next_token_ids])
            position += 1

        # Get final outputs from token manager
        final_results = token_manager.get_final_outputs()

        # Prepare return values
        generated_texts = []
        for result in final_results:
            # Combine prompt and output
            output_text = self.tokenizer.decode(result["output_ids"])
            # generated_text = result["prompt"] + output_text
            generated_text = output_text
            generated_texts.append(generated_text)

        if record_generation:
            return generated_texts, recorders
        return generated_texts, None

    def reset_cache_simple(self):
        """Reset the cache for the quick model"""
        for q in self.reference_model_input_queues:
            q.put_nowait("RESET_CACHE")
        # Wait for acknowledgment from the reference model
        for q in self.reference_model_ack_queues:
            ack = q.get()    
            # print(f"cache location reset to {ack}")

    def shutdown(self):
        """Shut down the SGLang engines to free resources"""
        for q in self.reference_model_input_queues:
            q.put_nowait(-1)  # Termination signal

    def __del__(self):
        if hasattr(self, "reference_model_procs"):
            for proc in self.reference_model_procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
