from dataclasses import dataclass
import uuid
import torch
import multiprocessing as mp
from typing import List, Optional, Union
import time

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.bench_serving import get_tokenizer
from r2r.utils.token_manager import SGLangTokenManager

@dataclass
class SGLangWrapperSignal:
    CONTINUE = 1
    CLEAN_UP = 0
    TERMINATE = -1

class SGLangWrapper:
    """Wrapper for SGLang model that handles initialization, warmup, and inference"""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        mem_fraction_static: float = 0.9,
        tp_size: int = 1,
        gpu_id: int = 0,
        disable_cuda_graph: bool = True,
        disable_overlap_schedule: bool = True,
        **sglang_kwargs
    ):
        self.model_path = model_path
        self.device = device
        self.gpu_id = gpu_id
        
        # Create server args for the model
        self.server_args = ServerArgs(
            model_path=model_path,
            disable_cuda_graph=disable_cuda_graph,
            disable_overlap_schedule=disable_overlap_schedule,
            mem_fraction_static=mem_fraction_static,
            tp_size=tp_size,
            **sglang_kwargs
        )
        
        # Create dedicated communication queues
        self.decode_input_queue = mp.Queue()
        self.decode_output_queue = mp.Queue()
        self.update_input_queue = mp.Queue()
        self.update_output_queue = mp.Queue()
        
        # Start model worker in a separate process
        self.model_proc = mp.Process(
            target=self.model_worker, 
            args=(
                self.server_args, 
                self.decode_input_queue, 
                self.decode_output_queue,
                self.update_input_queue,
                self.update_output_queue,
                self.gpu_id
            )
        )
        self.model_proc.start()
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(self.server_args.model_path)
        
        # Warm up the model
        if not disable_cuda_graph:
            self.warm_up()

    @staticmethod
    def model_worker(
        server_args: ServerArgs, 
        decode_input_queue: mp.Queue, 
        decode_output_queue: mp.Queue,
        update_input_queue: mp.Queue,
        update_output_queue: mp.Queue,
        gpu_id: int
    ):
        """Worker function that runs the scheduler in a separate process"""
        port_args = PortArgs.init_new(server_args)
        scheduler = Scheduler(
            server_args=server_args,
            port_args=port_args,
            gpu_id=gpu_id,
            tp_rank=0,
            dp_rank=None,
        )

        # Store the last batch for updates
        last_batch = None

        while True:
            # Get decode request (blocking)
            reqs_or_signal: Union[List[Req], int] = decode_input_queue.get()
            
            # Check for termination signal
            if isinstance(reqs_or_signal, int) and reqs_or_signal == SGLangWrapperSignal.TERMINATE:
                SGLangWrapper._clean_up(scheduler)
                break
                
            # Check for clean up signal
            if isinstance(reqs_or_signal, int) and reqs_or_signal == SGLangWrapperSignal.CLEAN_UP:
                SGLangWrapper._clean_up(scheduler)
                continue

            # Process decode step
            if isinstance(reqs_or_signal, list):
                # Case 1: New generation with a list of requests
                for req in reqs_or_signal:
                    scheduler.waiting_queue.append(req)
                
                batch = scheduler.get_next_batch_to_run()
            elif isinstance(reqs_or_signal, int) and reqs_or_signal == SGLangWrapperSignal.CONTINUE:
                # Case 2: Continue generation with current scheduler state
                # This is similar to DynamicSimpleSGLangSelector.decode_step
                batch = scheduler.get_next_batch_to_run()
            else:
                # Invalid input
                decode_output_queue.put({"error": f"Invalid input: {reqs_or_signal}"})
                continue
            
            # Check if batch is None (no batch to run)
            if batch is None:
                decode_output_queue.put({"error": "No batch available to run"})
                continue
            
            # Run the batch
            result = scheduler.run_batch(batch)
            
            # Extract results
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
            hidden_states = result.logits_output.hidden_states[hidden_indices, :][:, None, :].cpu()
            logits = result.logits_output.next_token_logits[:, None, :].cpu()
            next_token_ids = result.next_token_ids.cpu()
            
            # Store the batch for later updates
            last_batch = batch
            
            # Send results back through decode output queue
            decode_output_queue.put({
                "hidden_states": hidden_states,
                "logits": logits,
                "next_token_ids": next_token_ids
            })
            
            # Get update request (blocking)
            next_token_ids: torch.Tensor = update_input_queue.get()
            last_batch.output_ids = next_token_ids.to(device)
            
            # Process update
            if last_batch is not None:
                # Update output IDs
                for req, next_token_id in zip(last_batch.reqs, next_token_ids):
                    next_token_id = next_token_id.item()
                    if next_token_id in scheduler.model_config.hf_eos_token_id:
                        scheduler.abort_request(req)
                    req.output_ids.append(next_token_id)
                    req.check_finished()
                    if req.finished():
                        scheduler.tree_cache.cache_finished_req(req)

                scheduler.last_batch = last_batch
            else:
                print("No batch to update")
            
            # Confirm update completion
            update_output_queue.put({"status": "updated"})

    def warm_up(self):
        """Warm up the model with dummy inputs"""
        warmup_iter = 5
        input_text = "Hi"
        input_ids = self.tokenizer.encode(input_text)
        
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_new_tokens=1,
            stop=[]
        )
        
        for _ in range(warmup_iter):
            req = Req(
                rid=str(uuid.uuid4()),
                origin_input_text=input_text,
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
                return_hidden_states=True
            )
            req.sampling_params.normalize(None)
            
            # Send decode request
            self.decode_step([req])
            
            # Get next token ID from result
            result = self.decode_output_queue.get()
            next_token_ids = result["next_token_ids"]
            
            # Update output IDs
            self.update_step(next_token_ids)
            
            # Wait for update confirmation
            self.update_output_queue.get()

    def text_to_prompt(self, texts: List[str]) -> List[str]:
        """Convert a list of texts to a list of prompts"""
        prompts = []
        for text in texts:
            messages = [
                {"role": "user", "content": text},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompts.append(prompt)
        return prompts

    def create_req(
        self, 
        rid: str, 
        input_ids: List[int], 
        max_new_tokens: int = 1,
        temperature: float = 0.0, 
        top_p: Optional[float] = None, 
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> Req:
        """Create a SGLang request object
        
        Args:
            rid: Request ID
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for nucleus sampling (optional)
            top_k: Top-k for sampling (optional)
            repetition_penalty: Penalty for token repetition (optional)
            
        Returns:
            SGLang request object
        """
        input_text = self.tokenizer.decode(input_ids)
        
        # Create sampling params with only provided values
        sampling_params_kwargs = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": []
        }
        
        # Only add optional parameters if they are provided
        if top_p is not None:
            sampling_params_kwargs["top_p"] = top_p
        if top_k is not None:
            sampling_params_kwargs["top_k"] = top_k
        if repetition_penalty is not None:
            sampling_params_kwargs["repetition_penalty"] = repetition_penalty
            
        sampling_params = SamplingParams(**sampling_params_kwargs)
        
        req = Req(
            rid=rid,
            origin_input_text=input_text,
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
            # eos_token_ids=self.tokenizer.eos_token_id,
            return_hidden_states=True
        )
        req.sampling_params.normalize(None)
        
        return req

    def decode_step(self, reqs_or_signal: Union[List[Req], int]):
        """Decode one step using the model
        
        Args:
            reqs_or_signal: 
                - List[Req]: List of SGLang request objects to start new generation
                - 1: Continue generation with current scheduler state 
                - -1: Terminate the worker
        """
        self.decode_input_queue.put(reqs_or_signal)

    def update_step(self, next_token_ids: torch.Tensor):
        """Update the output ids with the next tokens
        
        Args:
            next_token_ids: The next token ids to add as a torch.Tensor
        """
        self.update_input_queue.put(next_token_ids)

    def generate(
        self,
        input_ids: List[List[int]],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> List[str]:
        """Generate text using the SGLang model
        
        Args:
            input_ids: A list of lists of token IDs for batch processing
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p probability threshold for nucleus sampling (optional)
            top_k: Top-k for sampling (optional)
            repetition_penalty: Penalty for token repetition (optional)
            
        Returns:
            List of generated texts
        """
        # Initialize token manager
        token_manager = SGLangTokenManager(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens
        )
        
        position = 0
        
        # First step: Initialize with request objects
        active_input_ids = token_manager.get_active_input_ids()
        
        # Create initial requests
        reqs = []
        for i, ids in enumerate(active_input_ids):
            req = self.create_req(
                rid=str(uuid.uuid4()),
                input_ids=ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            reqs.append(req)
        
        # Initial decode step with requests
        self.decode_step(reqs)
        
        # Generate tokens until all sequences are complete or max tokens reached
        while not token_manager.is_generation_complete() and position < max_new_tokens:
            # For subsequent steps, use continue generation signal (1) 
            if position > 0:
                self.decode_step(1)  # Signal to continue generation
                
            # Get results
            result = self.decode_output_queue.get()
            
            # Handle potential errors
            if "error" in result:
                print(f"Error in decode step: {result.get('error', '')}")
                break
                
            next_token_ids = result["next_token_ids"]
            
            # Update model state
            self.update_step(next_token_ids)
            self.update_output_queue.get()  # Wait for update confirmation
            
            # Update token manager with generated tokens
            # Convert to list for the token manager
            token_manager.update_sequences_direct(next_token_ids.tolist())
            
            # Increment position
            position += 1
        
        # Get final outputs from token manager
        final_outputs = token_manager.get_final_outputs()
        
        # Return generated texts
        generated_texts = []
        for output in final_outputs:
            generated_text = self.tokenizer.decode(output["output_ids"])
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def clean_up(self):
        """Clean up all unfinished requests"""
        self.decode_input_queue.put(SGLangWrapperSignal.CLEAN_UP)

    @staticmethod
    def _clean_up(scheduler):
        """Clean up all unfinished requests"""
        batch = scheduler.get_next_batch_to_run()

        if batch is None:
            print("No batch to clean up")
            return
        
        print(f"Cleaning up {len(batch.reqs)} requests")
        # TODO: check if this is necessary
        result = scheduler.run_batch(batch)        
        for req in batch.reqs:
            scheduler.abort_request(req)
            req.check_finished()
            if req.finished():
                scheduler.tree_cache.cache_finished_req(req)

        scheduler.last_batch = batch
        

    def shutdown(self):
        """Shutdown the model and free resources"""
        # Send termination signal to the decode input queue
        self.decode_input_queue.put(SGLangWrapperSignal.TERMINATE)
        self.model_proc.join()
        
    def __del__(self):
        """Clean up resources when the object is deleted"""
        self.shutdown()
