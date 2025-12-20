import argparse
import logging
import uvicorn
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import multiprocessing as mp

from r2r.models.batch_inference.sl_disaggregation_system import SLDisaggregationSystem
from r2r.utils.config import MODEL_DICT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

system: Optional[SLDisaggregationSystem] = None
server_args = None 

# Define request model
class GenerateReqInput(BaseModel):
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 100
    display_progress: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global system
    print("Initializing SLDisaggregationSystem inside lifespan...")
    
    if server_args:
        quick_sglang_kwargs = {
            "dtype": "bfloat16",
            "tp_size": server_args.tp_size_quick
        }
        reference_sglang_kwargs = {
            "dtype": "bfloat16",
            "tp_size": server_args.tp_size_ref
        }
        
        strategy_kwargs = {}
        if server_args.router_model_path:
            strategy_kwargs['model_path'] = server_args.router_model_path
        if server_args.router_threshold:
            strategy_kwargs['threshold'] = server_args.router_threshold

        try:
            system = SLDisaggregationSystem(
                device="cuda",
                dtype="bfloat16",
                switching_strategy="neural",
                strategy_kwargs=strategy_kwargs,
                quick_sglang_kwargs=quick_sglang_kwargs,
                reference_sglang_kwargs=reference_sglang_kwargs,
                overlap_tp_schedule=False
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
            max_new_tokens=obj.max_new_tokens,
            temperature=obj.temperature,
            top_p=obj.top_p,
            top_k=obj.top_k,
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

@app.get("/health")
async def health():
    global system
    if system is None:
        raise HTTPException(status_code=503, detail="System initializing")
    return {"status": "healthy"}

def run_server(args):
    global server_args
    server_args = args 
    
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=30000, help="Port to bind the server to")
    parser.add_argument("--tp-size-quick", type=int, default=1)
    parser.add_argument("--tp-size-ref", type=int, default=1)
    parser.add_argument("--overlap-tp-schedule", type=bool, default=False)
    parser.add_argument("--router-model-path", type=str, default="resource/default_router.pt")
    parser.add_argument("--router-threshold", type=float, default=None)

    mp.set_start_method("spawn", force=True)

    args = parser.parse_args()
    run_server(args)