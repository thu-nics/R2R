import argparse
import os
import asyncio
import uvicorn
import logging
import time
import uuid
import multiprocessing as mp
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sglang as sgl
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine and tokenizer
engine = None
tokenizer = None
server_args = None


# Request/Response Models
class GenerateReqInput(BaseModel):
    input_ids: Optional[List[int]] = None
    text: Optional[str] = None
    sampling_params: Dict[str, Any] = {}
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
    global engine, tokenizer, server_args
    if server_args:
        logger.info(f"Initializing SGLang Engine with model: {server_args.model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                server_args.model_path, trust_remote_code=True
            )

            engine = sgl.Engine(
                model_path=server_args.model_path,
                dtype="bfloat16",
                tp_size=server_args.tp_size,
                dp_size=server_args.dp_size,
                trust_remote_code=True,
                mem_fraction_static=server_args.mem_fraction_static,
            )
            # ---------------------------

            logger.info("SGLang Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            raise e
    yield
    if engine:
        engine.shutdown()


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate_request(obj: GenerateReqInput):
    global engine, tokenizer
    if engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        sampling_params = obj.sampling_params
        input_ids = obj.input_ids
        prompt = obj.text

        if input_ids is None and prompt is None:
            raise HTTPException(
                status_code=400, detail="Either text or input_ids must be provided"
            )

        if input_ids is not None:
            prompt = tokenizer.decode(input_ids)
            prompts = [prompt]
        else:
            prompts = [prompt]
            input_ids = tokenizer.encode(prompt)

        # Use engine.async_generate for proper async handling with uvicorn's event loop
        result_list = await engine.async_generate(prompts, sampling_params=sampling_params)

        result = result_list[0]

        if isinstance(result, dict):
            output_text = result["text"]
        else:
            output_text = getattr(result, "text", str(result))

        output_ids = tokenizer.encode(output_text, add_special_tokens=False)

        return {
            "text": output_text,
            "input_ids": input_ids,
            "output_ids": output_ids,
            "llm_percentage": 100.0,
            "model_agreement_percentage": 0,
            "quick_source_agreement_percentage": 0,
            "total_params_billions": 0,
            "avg_params_billions": 0,
        }

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""

    global engine, tokenizer
    if engine is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported yet")

    try:
        # Convert messages to text using chat template
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Apply chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        else:
            # Fallback: concatenate messages
            prompt = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages]
            )
            input_ids = tokenizer.encode(prompt)

        prompt_tokens = len(input_ids)

        # Build sampling params
        sampling_params = {
            "max_new_tokens": request.max_tokens or 2048,
            "temperature": request.temperature or 1.0,
            "top_p": request.top_p or 1.0,
        }
        if request.stop:
            sampling_params["stop"] = request.stop

        # Use engine.async_generate for proper async handling
        result_list = await engine.async_generate([prompt], sampling_params=sampling_params)

        result = result_list[0]
        output_text = result["text"]

        # Get completion tokens
        output_ids = tokenizer.encode(output_text, add_special_tokens=False)
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
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

        return response

    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    global engine
    if engine is None:
        raise HTTPException(status_code=503, detail="System initializing")
    return {"status": "healthy"}


def run_server(args):
    global server_args
    server_args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Base Model SGLang server")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", type=int, default=30000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallelism size"
    )
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallelism size")
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="Static memory fraction for KV cache",
    )

    mp.set_start_method("spawn", force=True)

    args = parser.parse_args()
    run_server(args)
