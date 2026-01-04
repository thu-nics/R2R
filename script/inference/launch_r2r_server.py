import argparse
import uvicorn
import multiprocessing as mp

from r2r.models import http_server
from r2r.models.http_server import app


def run_server(args):
    # Pass args to http_server module
    http_server.server_args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch R2R server with model configurations from a folder"
    )
    parser.add_argument(
        "--config-folder",
        type=str,
        required=True,
        help="Path to config folder containing model_configs.json and router.pt"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port to bind the server to"
    )
    parser.add_argument("--tp-size-quick", type=int, default=1)
    parser.add_argument("--tp-size-ref", type=int, default=1)
    parser.add_argument("--overlap-tp-schedule", action="store_true", default=False)
    parser.add_argument("--router-threshold", type=float, default=None)

    mp.set_start_method("spawn", force=True)

    args = parser.parse_args()
    run_server(args)
