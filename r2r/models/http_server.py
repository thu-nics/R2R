import os
os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "0"
os.environ["SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK"] = "1"

import argparse
import json
import logging
import uvicorn
import time
import uuid
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import multiprocessing as mp

from sglang.srt.managers.io_struct import GenerateReqInput as SGLangGenerateReqInput

from r2r.models.sglang_patch.sl_disaggregation_system import SLDisaggregationSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system: Optional[SLDisaggregationSystem] = None
server_args = None 

# Define request model
class GenerateReqInput(SGLangGenerateReqInput):
    # text: Optional[str] = None
    # input_ids: Optional[List[int]] = None
    # max_new_tokens: int = 2048
    # temperature: float = 0.0
    # top_p: float = 1.0
    # top_k: int = 100
    display_progress: bool = False


# OpenAI Chat Completion API Models
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    n: Optional[int] = 1


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    print("Initializing SLDisaggregationSystem inside lifespan...")

    if server_args:
        # Load config from path (file or folder)
        config_path = server_args.config_path
        with open(config_path, "r") as f:
            model_config = json.load(f)
        router_config = model_config.get("router", {})
        router_path = router_config.get("router_path")

        quick_sglang_kwargs = {
            "dtype": "bfloat16",
            "tp_size": server_args.tp_size_quick,
            "enable_return_hidden_states": True
        }
        reference_sglang_kwargs = {
            "dtype": "bfloat16",
            "tp_size": server_args.tp_size_ref
        }

        # Determine switching strategy first
        switching_strategy = router_config.get("switching_strategy")
        if switching_strategy is None:
            switching_strategy = "neural"
        print(f"Using switching strategy: {switching_strategy}")

        strategy_kwargs = {"model_path": router_path}

        # Threshold loading logic
        if switching_strategy == "neural":
            # Priority: config file's router.threshold > command line arg
            threshold = router_config.get("threshold")
            if threshold is None and server_args.threshold is not None:
                threshold = server_args.threshold
            
            if threshold is not None:
                strategy_kwargs["threshold"] = threshold
                print(f"Using neural threshold: {threshold}")
        else:
            # For non-neural strategies, use specific thresholds from config
            if "aleatoric_threshold" in router_config:
                strategy_kwargs["aleatoric_threshold"] = router_config["aleatoric_threshold"]
                print(f"Using aleatoric threshold from config: {router_config['aleatoric_threshold']}")
            
            if "entropy_threshold" in router_config:
                strategy_kwargs["entropy_threshold"] = router_config["entropy_threshold"]
                print(f"Using entropy threshold from config: {router_config['entropy_threshold']}")

        try:
            system = SLDisaggregationSystem(
                model_config=model_config,
                device="cuda",
                dtype="bfloat16",
                switching_strategy=switching_strategy,
                strategy_kwargs=strategy_kwargs,
                quick_sglang_kwargs=quick_sglang_kwargs,
                reference_sglang_kwargs=reference_sglang_kwargs,
                overlap_tp_schedule=server_args.overlap_tp_schedule
            )
            print("System initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize system: {e}")
            raise e

    yield

    print("Shutting down system...")
    if system:
        # system.shutdown()
        pass

app = FastAPI(lifespan=lifespan)

@app.post("/generate")
async def generate_request(obj: GenerateReqInput):

    global system
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        input_ids = obj.input_ids
        if input_ids is None:
            if obj.text is None:
                raise HTTPException(status_code=400, detail="Either text or input_ids must be provided")
            input_ids = system.tokenizer.encode(obj.text)

        
        result = await system.generate_one_request(
            input_id=input_ids,
            max_new_tokens=obj.sampling_params.get('max_new_tokens', 128),
            temperature=obj.sampling_params.get('temperature', 1.0),
            top_p=obj.sampling_params.get('top_p', 1.0),
            top_k=obj.sampling_params.get('top_k', -1),
            display_progress=obj.display_progress
        )
        
        
        output_ids = result.output_ids if hasattr(result, 'output_ids') else result.get('output_ids', [])
        output_text = system.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return {
            "text": output_text,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "llm_percentage": result.get('llm_percentage', None) if isinstance(result, dict) else None
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    
    global system
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")
    
    try:
        # Convert messages to text using chat template
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Apply chat template if available
        if hasattr(system.tokenizer, 'apply_chat_template'):
            input_ids = system.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=True
            )
        else:
            # Fallback: concatenate messages
            text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            input_ids = system.tokenizer.encode(text)
        
        prompt_tokens = len(input_ids)
        
        # Generate response
        result = await system.generate_one_request(
            input_id=input_ids,
            max_new_tokens=request.max_tokens or 2048,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            top_k=-1,
            display_progress=False
        )
        
        # Extract output
        output_ids = result.output_ids if hasattr(result, 'output_ids') else result.get('output_ids', [])
        output_text = system.tokenizer.decode(output_ids, skip_special_tokens=True)
        completion_tokens = len(output_ids)
        
        # Build OpenAI-compatible response
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model or "default",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text),
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    global system
    if system is None:
        raise HTTPException(status_code=503, detail="System initializing")
    return {"status": "healthy"}
