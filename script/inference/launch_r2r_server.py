import argparse
import uvicorn
import multiprocessing as mp

from r2r.models.http_server import app

def run_server(args):
    global server_args
    server_args = args 
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-model-path", type=str, required=True)

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=30000, help="Port to bind the server to")
    parser.add_argument("--tp-size-quick", type=int, default=1)
    parser.add_argument("--tp-size-ref", type=int, default=1)
    parser.add_argument("--overlap-tp-schedule", type=bool, default=False)
    parser.add_argument("--router-threshold", type=float, default=None)

    mp.set_start_method("spawn", force=True)

    args = parser.parse_args()
    run_server(args)
