import argparse
import os
import uvicorn
import multiprocessing as mp
import yaml

from r2r.models import http_server
from r2r.models.http_server import app


def run_server(args):
    # Pass args to http_server module
    http_server.server_args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch R2R server with model configurations"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config.yaml file or folder containing config.yaml"
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
    parser.add_argument('--neural-threshold', type=float, default=0.5,
                        help='Threshold for the neural switching strategy (default: 0.5).')

    mp.set_start_method("spawn", force=True)

    args = parser.parse_args()
    
    # Normalize config_path: convert folder to file path if needed
    config_path = args.config_path
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.yaml")
    args.config_path = config_path
    
    run_server(args)
