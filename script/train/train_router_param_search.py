import os
import sys
import json
import argparse
import wandb
from typing import Dict, Any

from script.train.train_router import main


def define_sweep_config() -> Dict[str, Any]:
    """Define the wandb sweep configuration."""
    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'final_positive_rate',
            'goal': 'minimize'  # We want to minimize positive rate while maintaining recall
        },
        'parameters': {
            # Model architecture parameters
            # For 512 dim, 0.5M params per layer
            'model.init_args.hidden_dims': {
                'values': [
                    [256, 256, 256],
                    [256, 256, 256, 256],
                    [256, 256, 256, 256, 256],
                    [256, 256, 256, 256, 256, 256],
                    [512, 512, 512],
                    [512, 512, 512, 512],
                    [512, 512, 512, 512, 512],
                    [512, 512, 512, 512, 512, 512],
                    [1024, 1024, 1024],
                    [1024, 1024, 1024, 1024],
                    [1024, 1024, 1024, 1024, 1024],
                    [1024, 1024, 1024, 1024, 1024, 1024],
                    [2048, 2048, 2048],
                    [2048, 2048, 2048, 2048],
                    [2048, 2048, 2048, 2048, 2048],
                    [2048, 2048, 2048, 2048, 2048, 2048]
                    
                ]
            },
            'model.init_args.dropout_rate': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 0.3
            },
            'model.init_args.expansion_factor': {
                'values': [1, 2, 3, 4]
            },
            'model.init_args.normalize_input': {
                'values': [False, True]
            },

            # Training parameters
            'training.params.batch_size': {
                'values': [128, 256, 512, 1024, 2048, 4096, 8192]
            },
            'training.params.device': {
                'values': ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5']
            },
            'training.params.checkpoint_dir': {
                'values': ['output/checkpoints/sweep']
            },
            'training.optimizer.lr': {
                'distribution': 'log_uniform_values',
                'min': 1e-7,
                'max': 1e-4
            },
            'training.optimizer.weight_decay': {
                'distribution': 'log_uniform_values',
                'min': 1e-5,
                'max': 1e-3
            },
            
            # Loss parameters
            'training.loss.type': {
                'values': [
                    'BCEWithLogitsLoss'
                ]
            },
            # 'training.loss.recall_factor': {
            #     'distribution': 'uniform',
            #     'min': 1.0,
            #     'max': 1.0
            # },
            # Alpha and gamma parameters (will only be used when FocalLoss is selected)
            # 'training.loss.beta': {
            #     'distribution': 'uniform',
            #     'min': 0.1,
            #     'max': 2.0
            # }
        }
    }
    
    return sweep_config


def sweep_train(config=None):
    """Training function for wandb sweep."""
    # Initialize a new wandb run
    with wandb.init(config=config, reinit=True):  # Add reinit=True to ensure clean initialization
        # Get the configuration for this run
        sweep_config = wandb.config
        
        # Load the base configuration
        with open(args.config, 'r') as f:
            base_config = json.load(f)
        
        # Update the base configuration with the sweep parameters
        # Model parameters
        model_params = [
            'model.init_args.hidden_dims',
            'model.init_args.dropout_rate',
            'model.init_args.expansion_factor',
            'model.init_args.normalize_input',
            'model.init_args.use_position_embedding'
        ]
        
        for param in model_params:
            if param in sweep_config:
                parts = param.split('.')
                base_config[parts[0]][parts[1]][parts[2]] = sweep_config[param]
        
        # Training parameters
        if 'training.params.batch_size' in sweep_config:
            base_config['training']['params']['batch_size'] = sweep_config['training.params.batch_size']
            
        # Override device with specific GPU if provided via command line
        gpu_id = None
        if args.gpu:
            base_config['training']['params']['device'] = args.gpu
            # Extract GPU ID for checkpoint path
            if 'cuda:' in args.gpu:
                gpu_id = args.gpu.split(':')[1]
        elif 'training.params.device' in sweep_config:
            base_config['training']['params']['device'] = sweep_config['training.params.device']
            # Extract GPU ID for checkpoint path
            if 'cuda:' in sweep_config['training.params.device']:
                gpu_id = sweep_config['training.params.device'].split(':')[1]
            
        if 'training.optimizer.lr' in sweep_config:
            base_config['training']['optimizer']['lr'] = sweep_config['training.optimizer.lr']
        if 'training.optimizer.weight_decay' in sweep_config:
            base_config['training']['optimizer']['weight_decay'] = sweep_config['training.optimizer.weight_decay']
        
        # Set checkpoint directory with GPU ID suffix when using multi-GPU mode
        if 'training.params.checkpoint_dir' in sweep_config:
            checkpoint_dir = sweep_config['training.params.checkpoint_dir']
            if gpu_id is not None and args.multi_gpu:
                checkpoint_dir = f"{checkpoint_dir}_gpu{gpu_id}"
            base_config['training']['params']['checkpoint_dir'] = checkpoint_dir
        
        # Loss parameters
        if 'training.loss.type' in sweep_config:
            loss_type = sweep_config['training.loss.type']
            base_config['training']['loss']['type'] = loss_type
            
            # Only add alpha and gamma if FocalLoss is selected
            if loss_type == 'FocalLoss':
                base_config['training']['loss']['alpha'] = sweep_config.get('training.loss.alpha', 0.5)
                base_config['training']['loss']['gamma'] = sweep_config.get('training.loss.gamma', 2.0)
            elif loss_type == 'DPOLoss':
                base_config['training']['loss']['beta'] = sweep_config.get('training.loss.beta', 1.0)
            else:
                # Default to BCEWithLogitsLoss
                pass
                
        if 'training.loss.recall_factor' in sweep_config:
            base_config['training']['loss']['recall_factor'] = sweep_config['training.loss.recall_factor']
        
        # Fix min_recall to 0.9
        base_config['optimizing']['min_recall'] = 0.95
        
        # Run the main training function
        main(base_config, use_wandb=True)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run a wandb sweep for router training")
    parser.add_argument("--config", type=str, default="config/router_config_sweep.json", 
                        help="Path to the base configuration file")
    parser.add_argument("--count", type=int, default=10, 
                        help="Number of runs to execute")
    parser.add_argument("--gpu", type=str, default=None, 
                        help="Specific GPU to use for this agent (e.g., 'cuda:0')")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Connect to an existing sweep ID instead of creating a new one")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Run multiple sweep agents, one on each available GPU")
    parser.add_argument("--project", type=str, default="FlexThink",
                        help="W&B project to use for the sweep")
    parser.add_argument("--entity", type=str, default=None,
                        help="W&B entity (username or team) to use for the sweep")
    args = parser.parse_args()
    
    # If connecting to existing sweep, use that ID, otherwise create a new sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Using existing sweep ID: {sweep_id}")
        print(f"Will attempt to connect to project: {args.project}" + (f", entity: {args.entity}" if args.entity else ""))
    else:
        # Define and initialize the sweep
        sweep_config = define_sweep_config()
        
        # If using a specific GPU, remove device from sweep parameters
        if args.gpu:
            if 'training.params.device' in sweep_config['parameters']:
                del sweep_config['parameters']['training.params.device']
        
        print(f"Creating new sweep in project: {args.project}" + (f", entity: {args.entity}" if args.entity else ""))
        
        # Create the sweep with explicit project and entity
        try:
            sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
            print(f"Created new sweep with ID: {sweep_id}")
        except Exception as e:
            print(f"Error creating sweep: {str(e)}")
            sys.exit(1)
    
    if args.multi_gpu:
        # Launch multiple agents on different GPUs
        import subprocess
        import torch
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Launching {num_gpus} agents on available GPUs")
        
        # Launch agents
        processes = []
        runs_per_agent = max(1, args.count // num_gpus)
        
        for i in range(num_gpus):
            gpu = f"cuda:{i}"
            cmd = [
                sys.executable, sys.argv[0], 
                "--count", str(runs_per_agent), 
                "--gpu", gpu,
                "--sweep-id", sweep_id,
                "--config", args.config,
                "--project", args.project,
                "--multi-gpu"  # Add multi-gpu flag to child processes
            ]
            
            # Add entity if specified
            if args.entity:
                cmd.extend(["--entity", args.entity])
                
            print(f"Starting agent on GPU {gpu}")
            
            # Create a new environment for the subprocess without wandb variables
            # to prevent service conflicts
            env = os.environ.copy()
            for var in list(env.keys()):
                if var.startswith('WANDB_'):
                    del env[var]
            
            # Set specific wandb environment variables for the subprocess
            env['WANDB_START_METHOD'] = 'thread'
            env['WANDB_AGENT_DISABLE_FLAPPING'] = 'true'
            
            # Keep the API key for authentication
            if 'WANDB_API_KEY' in os.environ:
                env['WANDB_API_KEY'] = os.environ['WANDB_API_KEY']
                
            # Explicitly set project and entity in environment
            env['WANDB_PROJECT'] = args.project
            if args.entity:
                env['WANDB_ENTITY'] = args.entity
            
            processes.append(subprocess.Popen(cmd, env=env))
        
        # Wait for all processes to complete with proper error handling
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Terminating all agents...")
            for p in processes:
                try:
                    p.terminate()
                    p.wait(timeout=5)  # Give it some time to terminate gracefully
                except:
                    # Force kill if termination doesn't work
                    try:
                        p.kill()
                    except:
                        pass  # Process might already be gone
            raise  # Re-raise the KeyboardInterrupt
    else:
        # Run a single agent
        if args.gpu:
            print(f"Running sweep agent on GPU {args.gpu}")
        else:
            print("Running sweep agent with default GPU configuration")
        
        # Run the agent with explicit project and entity
        try:
            wandb.agent(sweep_id, function=sweep_train, count=args.count, 
                       project=args.project, entity=args.entity)
        except wandb.errors.errors.UsageError as e:
            if "could not find sweep" in str(e):
                print(f"\nERROR: Sweep {sweep_id} not found in project {args.project}")
                print("This could be because:")
                print("1. The sweep ID is incorrect")
                print("2. The sweep exists in a different project or entity")
                print("3. You don't have permission to access this sweep")
                print("\nTo fix this, ensure you're using the correct:")
                print("- Sweep ID (check the URL in your wandb dashboard)")
                print("- Project name (use --project)")
                print("- Entity name (use --entity if needed)")
                print("\nFull error:", str(e))
            else:
                print(f"Wandb usage error: {str(e)}")
        except Exception as e:
            print(f"Error running wandb agent: {str(e)}")