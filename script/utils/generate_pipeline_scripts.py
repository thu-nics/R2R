#!/usr/bin/env python3
"""
Data Generation Pipeline Script Generator

This script generates all necessary commands for the R2R data generation pipeline.
It prompts for required variables and outputs a comprehensive markdown file with all commands.

The pipeline consists of:
1. Step 0: LLM Response 
2. Step 1: SLM Prefill
3. Step 2: LLM Continuation (choice between local or API version)
4. Step 3: Verify
5. Step 4: Construct Training Dataset

Available model configs:
- R1: DeepSeek-R1-Distill-Qwen (1.5B small, 32B large)
- Qwen3_0.6B: Qwen3-0.6B (small) and Qwen3-8B (large)
- Qwen3_1.7B: Qwen3-1.4B (small) and Qwen3-8B (large) 
- Qwen3_4B: Qwen3-4B (small) and Qwen3-8B (large)
- Qwen3_0.6B_32B: Qwen3-0.6B (small) and Qwen3-32B (large)
- Qwen3_30B-A3B: Qwen3-0.6B (small) and Qwen3-30B-A3B (large)
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

class PipelineScriptGenerator:
    def __init__(self):
        self.config_mapping = {
            "R1": "model_configs_r1.json",
            "Qwen3_0.6B": "model_configs_qwen3_0_6B.json", 
            "Qwen3_1.7B": "model_configs_qwen3_1_4B.json",
            "Qwen3_4B": "model_configs_qwen3_4B.json",
            "Qwen3_0.6B_32B": "model_configs_qwen3_0_6B_32B.json",
            "Qwen3_30B-A3B": "model_configs_qwen3_30B-A3B.json"
        }
        
    def load_model_config(self, config_choice: str) -> Dict[str, Any]:
        """Load the model configuration from the JSON file."""
        config_file = self.config_mapping[config_choice]
        config_path = Path("script/utils/configs") / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def prompt_for_variables(self) -> Dict[str, Any]:
        """Prompt user for all required variables."""
        print("=== R2R Data Generation Pipeline Script Generator ===\n")
        
        # Model config choice
        print("Available model configs:")
        for i, (key, desc) in enumerate([
            ("R1", "DeepSeek-R1-Distill-Qwen (1.5B small, 32B large)"),
            ("Qwen3_0.6B", "Qwen3-0.6B (small) and Qwen3-8B (large)"),
            ("Qwen3_1.7B", "Qwen3-1.4B (small) and Qwen3-8B (large)"),
            ("Qwen3_4B", "Qwen3-4B (small) and Qwen3-8B (large)"),
            ("Qwen3_0.6B_32B", "Qwen3-0.6B (small) and Qwen3-32B (large)"),
            ("Qwen3_30B-A3B", "Qwen3-0.6B (small) and Qwen3-30B-A3B (large)")
        ], 1):
            print(f"{i}. {key}: {desc}")
        
        while True:
            choice = input("\nSelect model config (1-6): ").strip()
            if choice in ["1", "2", "3", "4", "5","6"]:
                config_choices = ["R1", "Qwen3_0.6B", "Qwen3_1.7B", "Qwen3_4B", "Qwen3_0.6B_32B", "Qwen3_30B-A3B"]
                model_config = config_choices[int(choice) - 1]
                break
            print("Invalid choice. Please select 1-6.")
        
        # Temperature and top_p
        while True:
            try:
                temperature = float(input("Enter temperature (e.g., 0.6): ").strip())
                if 0.0 <= temperature <= 2.0:
                    break
                print("Temperature should be between 0.0 and 2.0")
            except ValueError:
                print("Please enter a valid number for temperature.")
        
        while True:
            try:
                top_p = float(input("Enter top_p (e.g., 0.95): ").strip())
                if 0.0 <= top_p <= 1.0:
                    break
                print("top_p should be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number for top_p.")
        
        # Output directory
        output_dir = input("Enter output directory (e.g., output/my_experiment): ").strip()
        if not output_dir:
            output_dir = "output/data_generation"
        
        # Step 2 script choice
        print("\nStep 2 script options:")
        print("1. script/data_labeling/step_2_llm_continuation.py (local inference)")
        print("2. script/data_labeling_api/step_2_llm_continuation.py (API inference)")
        
        while True:
            step2_choice = input("Select Step 2 script (1-2): ").strip()
            if step2_choice in ["1", "2"]:
                step2_script = "script/data_labeling/step_2_llm_continuation.py" if step2_choice == "1" else "script/data_labeling_api/step_2_llm_continuation.py"
                break
            print("Invalid choice. Please select 1 or 2.")
        
        # Dataset choice
        print("\nDataset options:")
        print("1. Use HuggingFace dataset (nics-efc/R2R_query)")
        print("2. Use local dataset path")
        
        while True:
            dataset_choice = input("Select dataset option (1-2): ").strip()
            if dataset_choice in ["1", "2"]:
                if dataset_choice == "1":
                    dataset_path = "nics-efc/R2R_query"
                    use_hf_dataset = True
                else:
                    dataset_path = input("Enter local dataset path: ").strip()
                    use_hf_dataset = False
                break
            print("Invalid choice. Please select 1 or 2.")
        
        # Split choice
        print("\nDataset split options:")
        print("1. Generate scripts for both train and validation splits")
        print("2. Generate scripts for train split only")
        print("3. Generate scripts for validation split only")
        
        while True:
            split_choice = input("Select split option (1-3): ").strip()
            if split_choice in ["1", "2", "3"]:
                if split_choice == "1":
                    generate_splits = ["train", "validation"]
                elif split_choice == "2":
                    generate_splits = ["train"]
                else:
                    generate_splits = ["validation"]
                break
            print("Invalid choice. Please select 1-3.")
        
        # Config replacement choice
        print(f"\nModel config replacement:")
        print(f"Do you want to replace r2r/utils/model_configs.json with the {model_config} recipe?")
        print("1. Yes - Replace the config file")
        print("2. No - Keep the current config file")
        
        while True:
            replace_choice = input("Select option (1-2): ").strip()
            if replace_choice in ["1", "2"]:
                replace_config = replace_choice == "1"
                break
            print("Invalid choice. Please select 1 or 2.")
        
        return {
            "model_config": model_config,
            "temperature": temperature,
            "top_p": top_p,
            "output_dir": output_dir,
            "step2_script": step2_script,
            "dataset_path": dataset_path,
            "use_hf_dataset": use_hf_dataset,
            "generate_splits": generate_splits,
            "replace_config": replace_config
        }
    
    def generate_markdown(self, variables: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate the complete markdown with all pipeline commands."""
        
        # Extract model paths
        quick_model = config["quick"]["model_path"]
        reference_model = config["reference"]["model_path"]
        verify_model = config["verify"]["model_path"]
        
        output_dir = variables["output_dir"]
        dataset_path = variables["dataset_path"]
        use_hf_dataset = variables["use_hf_dataset"]
        temperature = variables["temperature"]
        top_p = variables["top_p"]
        step2_script = variables["step2_script"]
        generate_splits = variables["generate_splits"]
        
        replace_config = variables["replace_config"]
        
        markdown = f"""# R2R Data Generation Pipeline Scripts

## Configuration Summary
- **Model Config**: {variables["model_config"]}
- **Small Model**: {quick_model}
- **Large Model**: {reference_model}
- **Verify Model**: {verify_model}
- **Temperature**: {temperature}
- **Top P**: {top_p}
- **Output Directory**: {output_dir}
- **Dataset**: {dataset_path}
- **Splits**: {', '.join(generate_splits)}
- **Step 2 Script**: {step2_script}
- **Replace Config**: {"Yes" if replace_config else "No"}

## Pipeline Commands

"""

        # Add config replacement step if needed
        if replace_config:
            config_file = self.config_mapping[variables["model_config"]]
            markdown += f"""### Step -1: Replace Model Configuration
Replace the current model config with the {variables["model_config"]} recipe:

**Note**: This step updates the model configuration used by the R2R pipeline. The script will automatically create a backup of your current config.

```bash
# Create backup (optional - script does this automatically)
cp r2r/utils/model_configs.json r2r/utils/model_configs.json.backup

# Replace with selected recipe
cp script/utils/configs/{config_file} r2r/utils/model_configs.json
```

This ensures that:
- The pipeline uses the correct model paths for {variables["model_config"]}
- All model parameters (memory fractions, tensor parallelism) are properly configured
- Special tokens and model-specific settings are correctly set

"""
        # Generate commands for each split
        for split in generate_splits:
            # Dataset flags for this split
            dataset_flags = f"--dataset_path {dataset_path}"
            if use_hf_dataset:
                dataset_flags += " --use_hf_dataset"
            dataset_flags += f" --split {split}"
            
            split_title = split.capitalize()
            markdown += f"""
## {split_title} Split Commands

### Step 0: LLM Response Generation ({split_title})
Generate responses using the large language model:
```bash
python script/data_labeling/step_0_llm_response.py \\
    --model_path {reference_model} \\
    {dataset_flags} \\
    --output_dir {output_dir}/query_dataset_{split}/LLM_response \\
    --tp_size {config["reference"].get("tp_size", 2)} \\
    --temperature {temperature} \\
    --top_p {top_p} \\
    --batch_size 32
```

### Step 1: SLM Prefill Analysis ({split_title})
Use the small language model to find non-identical responses:
```bash
python script/data_labeling/step_1_slm_prefill.py \\
    --dataset_path {output_dir}/query_dataset_{split}/LLM_response/dataset_finished \\
    --test_model_list {quick_model} \\
    --output_path {output_dir}/query_dataset_{split}/LLM_response/SLM_prefill \\
    --temperature {temperature} \\
    --top_p {top_p}
```

### Step 2: LLM Continuation ({split_title})
Continue generation from mismatch points:

**Note**: This step always uses temperature=0 and top_p=1, regardless of the input parameters.

```bash
python {step2_script} \\
    --input_path {output_dir}/query_dataset_{split}/LLM_response/SLM_prefill/prediction_comparison.csv \\
    --output_path {output_dir}/query_dataset_{split}/LLM_response/SLM_prefill/LLM_continuation_verify \\
    --tp_size {config["continuation_main"].get("tp_size", 2)}"""

            # Add API-specific instructions if using API version
            if "api" in step2_script:
                markdown += f"""

**For API version**: Start SGLang server first:
```bash
python -m sglang.launch_server \\
    --model-path {reference_model} \\
    --tp {config["continuation_main"].get("tp_size", 2)} \\
    --skip-tokenizer-init \\
    --enable-custom-logit-processor
```"""

            markdown += f"""
```

### Step 3: Verification ({split_title})
Use the verification model to check if continuations are divergent:
```bash
python script/data_labeling/step_3_verify.py \\
    --input_csv {output_dir}/query_dataset_{split}/LLM_response/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full.csv \\
    --output_csv {output_dir}/query_dataset_{split}/LLM_response/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full_verify.csv \\
    --verify_model {verify_model} \\
    --tp_size {config["verify"].get("tp_size", 4)}
```

### Step 4: Construct Training Dataset ({split_title})
Convert processed data into structured dataset for router training:
```bash
python script/data_labeling/step_4_construct_label_dataset.py \\
    --data_dir {output_dir}/query_dataset_{split}/LLM_response/SLM_prefill \\
    --csv LLM_continuation_verify/generation_results_data_all_real_full_verify.csv \\
    --output_sub_folder LLM_continuation_verify/divergent_label_dataset \\
    --divergent_column_name divergent
```

### Complete Pipeline Script ({split_title})
You can run all steps sequentially using the main pipeline script:
```bash
python script/data_labeling/data_generation.py \\
    {dataset_flags} \\
    --output_dir {output_dir}/query_dataset_{split} \\
    --temperature {temperature} \\
    --top_p {top_p} \\
    --tp_size {config["reference"].get("tp_size", 2)} \\
    --batch_size 32
```
"""

        markdown += f"""
## GPU Recommendations
- **Step 0**: Requires {config["reference"].get("tp_size", 2)} GPUs for large model
- **Step 1**: Single GPU sufficient for small model  
- **Step 2**: Requires {config["continuation_main"].get("tp_size", 2)} GPUs for continuation
- **Step 3**: Requires {config["verify"].get("tp_size", 4)} GPUs for verification model
- **Step 4**: CPU intensive, minimal GPU needed

## Notes
- All paths are relative to the project root
- Adjust `tp_size` based on available GPU memory
- Use `--resume` flag to continue interrupted runs
- Monitor disk space as intermediate files can be large
- Step 2 always uses temperature=0 and top_p=1 for consistency
- If generating both splits, you can run them in parallel on different GPUs
- Config replacement creates automatic backups at `r2r/utils/model_configs.json.backup`
- Different model configs have different GPU requirements and model paths
"""
        
        return markdown
    
    def save_markdown(self, content: str, output_file: str):
        """Save the markdown content to a file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(content)
        
        print(f"\nPipeline scripts generated successfully!")
        print(f"Output saved to: {output_path.absolute()}")
    
    def replace_model_config(self, model_config: str):
        """Replace the main model config file with the selected recipe."""
        try:
            config_file = self.config_mapping[model_config]
            source_path = Path("script/utils/configs") / config_file
            target_path = Path("r2r/utils/model_configs.json")
            
            if not source_path.exists():
                raise FileNotFoundError(f"Source config file not found: {source_path}")
            
            # Create backup of current config
            if target_path.exists():
                backup_path = target_path.with_suffix('.json.backup')
                shutil.copy2(target_path, backup_path)
                print(f"Backup created: {backup_path}")
            
            # Copy new config
            shutil.copy2(source_path, target_path)
            print(f"Model config replaced: {source_path} -> {target_path}")
            
        except Exception as e:
            print(f"Error replacing config: {e}")
            raise

    def run(self, output_file: Optional[str] = None):
        """Main execution function."""
        try:
            # Get user input
            variables = self.prompt_for_variables()
            
            # Replace config file if requested
            if variables["replace_config"]:
                print(f"\nReplacing model configuration with {variables['model_config']} recipe...")
                self.replace_model_config(variables["model_config"])
            
            # Load model configuration
            config = self.load_model_config(variables["model_config"])
            
            # Generate markdown
            markdown_content = self.generate_markdown(variables, config)
            
            # Determine output file
            if output_file is None:
                splits_suffix = "_".join(variables['generate_splits'])
                output_file = f"generated_pipeline_scripts_{variables['model_config'].lower()}_{splits_suffix}.md"
            
            # Save to file
            self.save_markdown(markdown_content, output_file)
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate R2R data generation pipeline scripts")
    parser.add_argument(
        "--output", 
        "-o", 
        type=str, 
        help="Output markdown file path (default: generated_pipeline_scripts_<config>.md)"
    )
    parser.add_argument(
        "--interactive",
        "-i", 
        action="store_true",
        default=True,
        help="Run in interactive mode (default)"
    )
    
    args = parser.parse_args()
    
    generator = PipelineScriptGenerator()
    generator.run(args.output)


if __name__ == "__main__":
    main()
