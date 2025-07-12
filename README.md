# R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing

<p align="center">
🌐 <a href="https://fuvty.github.io/R2R_Project_Page/"><b>Project Page</b></a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2505.21600"><b>arXiv</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/papers/2505.21600"><b>HuggingFace</b></a>
</p>

Roads to Rome (R2R) intelligently combines small and large language models by routing **only critical, reasoning-divergent tokens to the large model**.

https://github.com/user-attachments/assets/382fabd8-a816-44ba-b100-b8dd047c3bcb

By combining DeepSeek's R1-1.5B and R1-32B models, **R2R-5.6B achieves a 2.8× speedup** over R1-32B while **surpassing R1-7B and R1-14B by 1.6× and 1.1×** in accuracy on challenging math, coding, and QA benchmarks.

```bibtex
@article{fu2025r2r,
    title={R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing}, 
    author={Tianyu Fu and Yi Ge and Yichen You and Enshu Liu and Zhihang Yuan and Guohao Dai and Shengen Yan and Huazhong Yang and Yu Wang},
    journal={arXiv preprint arXiv:2505.21600},
    year={2025},
}
```

⭐ **Feel free to star this repo or cite our paper if you find it useful!**

## 📰 News

* [2025/06] Support Qwen3 model family and sampling.

## 🔗 Interactive Demo

Check out our interactive demo and see R2R in action by visiting our [project page](https://fuvty.github.io/R2R_Project_Page/).


## 🛠️ Environment Setup

Create a new conda environment and install dependencies:

```bash
conda create -n r2r python=3.10
conda activate r2r
pip install -e .
```

Install `flashinfer==0.2.3` based on your CUDA version. For example, for CUDA 12.4, you can install it with:

```bash
pip install flashinfer-python==0.2.3 -i https://flashinfer.ai/whl/cu124/torch2.6/
```

> If you accidentally install the wrong flashinfer, please uninstall it before re-installation.
> ```bash
>pip uninstall flashinfer-python
>rm -rf ~/.cache/flashinfer/
>rm -rf ~/.triton/cache
>```
## 🚀 Usage

### 1. 💬 Run Mix inference with R2R

We provide an interactive example in `interactive_chat.py`. The main `DynamicSimpleSGLangSelector` class follows the SGLang offline Engine API and supports the `.generate()` method for getting responses.

You can download the pre-trained router from [this link](https://huggingface.co/nics-efc/R2R_router/tree/main) and place the file `default_router.pt` under `resource/` folder:

```bash
python script/playground/interactive_chat.py --router_path resource/default_router.pt
```

> The detailed model configurations are in `r2r/utils/config.py`.

### 2. 📊 Benchmark Performance

The following script evaluates R2R's accuracy and speed on AIME24-25, GPQA-Diamond, or LiveCodeBench:

```bash
python script/evaluate/hf_dataset_sglang.py --dataset aime --router_path resource/default_router.pt --use_hybrid
```

Detailed configurations for benchmark datasets and evaluation metrics are available in `script/evaluate/eval_configs/dataset_configs.json`. Moreover, our default router_path and threshold settings are provided through `script/evaluate/eval_configs/r2r_configs.json`.

For speed benchmark, run the following command:
```bash
# R2R speed benchmark
python script/playground/speed_benchmark.py --test_r2r --router_path resource/default_router.pt
# SLM/LLM speed benchmark
python script/playground/speed_benchmark.py --test_slm
python script/playground/speed_benchmark.py --test_llm
```

### 3. 🧪 Train Your Own R2R Router

To train a custom R2R router for any LLM-SLM pair, you need to:
1. Prepare a model preference label dataset
2. Train the router using that dataset

#### 3.1 Dataset Preparation

We provide a complete data generation pipeline in `script/data_labeling/`. You can either use our pre-generated training dataset from [Hugging Face](https://huggingface.co/datasets/nics-efc/R2R_Router_Training/tree/main) and skip to section 3.2, or follow these steps to create your own dataset.

##### Initialize Dataset Conversion

Due to varying column names and data structures across different datasets, 
this step standardizes all datasets into a unified format for downstream 
processing. Customize datasets using `--dataset_config`:

```bash
python script/data_labeling/init_dataset_conversion.py --dataset_config aime,gpqa_extended,Bespoke-Stratos-17k-Code,Bespoke-Stratos-17k-QA --output_dir output/query_dataset
```

> **Alternative**: Skip this step by using our pre-processed dataset [`nics-efc/R2R_query`](https://huggingface.co/datasets/nics-efc/R2R_query/tree/main).

> **Add new dataset:** customize the configuration file to standardize new dataset following the format in `script/data_labeling/support_dataset_config.json`.

##### Step0: Generate LLM Responses

Generate responses using a large language model (default: `DeepSeek-R1-Distill-Qwen-32B`):

```bash
python script/data_labeling/step_0_llm_response.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path output/query_dataset --output_dir output/query_dataset/LLM_response --tp_size 2
```
We recommend using complete LLM responses within the 32K token limit for subsequent processing, saved under the `datasets_finished/` folder. Alternatively, to use the pre-processed dataset, passing `--dataset_path nics-efc/R2R_query --use_hf_dataset` in the instruction above.

For faster and tunable inference, generate responses using SGLang API server:
```bash
# Start SGLang server
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tp 2
# Run API inference
python script/data_labeling_api/step_0_llm_response.py --api_url http://localhost:30000/v1 --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path output/query_dataset --output_dir output/query_dataset/LLM_response --max_concurrent_requests 16
```

##### Step 1: SLM Prefill Analysis

Use the small language model (`DeepSeek-R1-Distill-Qwen-1.5B`) to prefill and find non-identical LLM responses:

```bash
python script/data_labeling/step_1_slm_prefill.py --dataset_path output/query_dataset/LLM_response/dataset_finished --test_model_list deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output_path output/query_dataset/LLM_response/SLM_prefill
```

This generates SLM predictions, top-100 logits, and hidden states.

##### Step 2: LLM Continuation

Use the LLM to continue from SLM's non-identical prefill positions:

```bash
python script/data_labeling/step_2_llm_continuation.py --input_path output/query_dataset/LLM_response/SLM_prefill/prediction_comparison.csv --output_path output/query_dataset/LLM_response/SLM_prefill/LLM_continuation_verify --tp_size 2
```

> **Note**: To use different models or loading path, edit the configuration in `r2r/utils/config.py`

To speed up the process, use API inference instead:

```bash
# Start SGLang server
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tp 2 --skip-tokenizer-init --enable-custom-logit-processor
# Run API inference
python script/data_labeling_api/step_2_llm_continuation.py --input_path output/query_dataset/LLM_response/SLM_prefill/prediction_comparison.csv --output_path output/query_dataset/LLM_response/SLM_prefill/LLM_continuation_verify --max_concurrent_requests 32
```

##### Step 3: Verify

Use `Qwen2.5-72B-Instruct` to verify whether LLM continuation responses are divergent:

```bash
python script/data_labeling/step_3_verify.py --input_csv output/query_dataset/LLM_response/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full.csv --output_csv output/query_dataset/LLM_response/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full_verify.csv --verify_model Qwen/Qwen2.5-72B-Instruct --tp_size 4
```

##### Step 4: Construct Training Dataset

Convert all processed data into a structured dataset for router training:

```bash
python script/data_labeling/step_4_construct_label_dataset.py --data_dir output/query_dataset/LLM_response/SLM_prefill --csv LLM_continuation_verify/generation_results_data_all_real_full_verify.csv --output_sub_folder LLM_continuation_verify/divergent_label_dataset --divergent_column_name divergent
```

#### 3.2 Router Training

Train the router using the prepared dataset:

```bash
python script/train/train_router.py --config resource/default_training_config.json
```

Add `--use_wandb` to track training progress with Weights & Biases.

The training script accepts the config file that specifies model architecture, dataset paths, training parameters, and threshold criteria. Modify it if you wish to alter the training process.

>We also provide a recipe for the Qwen3 series. To use it, simply replace r2r/utils/model_configs.json with model_configs_Qwen3_series.json, and update args.test_model_list to use the corresponding small model as described in Step 1.

## 🙌 Happy to help

If you have questions about any aspect of R2R, please open an issue. We're happy to help and discuss!
