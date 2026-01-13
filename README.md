<div align="center">

  <img src="resource/logo.png" alt="R2R Logo" width="100"/>

  <h1>Roads to Rome (R2R)</h1>
  <h3>Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing</h3>

  <p>
    <a href="https://fuvty.github.io/R2R_Project_Page/">🌐 <b>Project Page</b></a> •
    <a href="https://arxiv.org/abs/2505.21600">📑 <b>arXiv</b></a> •
    <a href="https://huggingface.co/collections/nics-efc/r2r">🤗 <b>HuggingFace</b></a>
  </p>

</div>

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
* [2025/10] Added support for the Qwen3 model family. Router checkpoints are now available [here](https://huggingface.co/nics-efc/R2R_router_collections).

* [2025/09] Accepted by the NeurIPS'25 conference.

* [2025/06] Support sampling on Deepseek's R1-1.5B and R1-32B models.

## 🔗 Interactive Demo

Check out our interactive demo and see R2R in action by visiting our [project page](https://fuvty.github.io/R2R_Project_Page/).


## 🛠️ Environment Setup

Use the following script to create a new Conda environment and install all dependencies:

```bash
bash setup_env.sh
```

`setup_env.sh` installs `flashinfer==0.2.3`. Make sure you install a FlashInfer build that matches your CUDA version. If your system uses a different CUDA version, install the corresponding FlashInfer package for your setup.

<details>
<summary>Troubleshooting</summary>

```bash
pip uninstall flashinfer-python
rm -rf ~/.cache/flashinfer/
rm -rf ~/.triton/cache
```
</details>

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

> 💡 Remember to edit `r2r/utils/model_configs.json` according to your training setup before running the following steps.

<details>
<summary>Click to see detailed training instructions</summary>

#### 3.1 Dataset Preparation

We provide a complete data generation pipeline in `script/data_labeling/`. You can either use our pre-generated training dataset from [Hugging Face](https://huggingface.co/datasets/nics-efc/R2R_Router_Training/tree/main) and skip to section 3.2, or follow the steps below to create your own dataset.

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

> For faster data generation, we provide code using SGLang API server:
> ```bash
> # Start SGLang server
> python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tp 2
> # Run API inference
> python script/data_labeling_api/step_0_llm_response.py --api_url http://localhost:30000/v1 --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path output/query_dataset --output_dir output/query_dataset/LLM_response --max_concurrent_requests 16
> ```


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

> **Note**: To use different models or loading path, edit the configuration in `r2r/utils/model_configs.json`. Pay attention to configs like special token ids and vocabulary size.

> For faster data generation, we provide code using SGLang API server:
> ```bash
> # Start SGLang server
> python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --tp 2 --skip-tokenizer-init --enable-custom-logit-processor
> # Run API inference
> python script/data_labeling_api/step_2_llm_continuation.py --input_path output/query_dataset/LLM_response/SLM_prefill/prediction_comparison.csv --output_path output/query_dataset/LLM_response/SLM_prefill/LLM_continuation_verify --max_concurrent_requests 32
> ```

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

</details>

## 🔗 Pretrained routers

| Small Language Model | Large Language Model | Router Link |
|-----|-----|-------------|
| DeepSeek-R1-Distill-Qwen-1.5B | DeepSeek-R1-Distill-Qwen-32B | [🔗 Link](https://huggingface.co/nics-efc/R2R_router_collections/tree/main/DeepSeek-R1-Distill-Qwen-1.5B%2BDeepSeek-R1-Distill-Qwen-1.5B-32B) |
| Qwen3-0.6B | Qwen3-8B | [🔗 Link](https://huggingface.co/nics-efc/R2R_router_collections/tree/main/Qwen3-0.6B%2BQwen3-8B) |
| Qwen3-0.6B | Qwen3-30B-A3B | [🔗 Link](https://huggingface.co/nics-efc/R2R_router_collections/tree/main/Qwen3-0.6B%2BQwen3-30B-A3B) |
| Qwen3-0.6B | Qwen3-32B | [🔗 Link](https://huggingface.co/nics-efc/R2R_router_collections/tree/main/Qwen3-0.6B%2BQwen3-32B) |
| Qwen3-1.7B | Qwen3-8B | [🔗 Link](https://huggingface.co/nics-efc/R2R_router_collections/tree/main/Qwen3-1.7B%2BQwen3-8B) |
| Qwen3-4B | Qwen3-8B | [🔗 Link](https://huggingface.co/nics-efc/R2R_router_collections/tree/main/Qwen3-4B%2BQwen-8B) |


## 🙌 Happy to help

If you have questions about any aspect of R2R, please open an issue. We're happy to help and discuss!

## 🌟 Related Projects

Explore more efficient LLM projects from us:

<table style="border: none; border-collapse: collapse;" align="center">
<tr>
<td align="center" valign="top" width="25%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/TaH">
<img src="https://raw.githubusercontent.com/thu-nics/TaH/main/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/TaH"><b>TaH</b></a>
<br/><sub>Selective latent thinking for reasoning LLMs</sub>
</td>
<td align="center" valign="top" width="25%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/C2C">
<img src="https://raw.githubusercontent.com/thu-nics/C2C/main/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/C2C"><b>C2C</b></a>
<br/><sub>Communicate through KV-Cache between LLMs</sub>
</td>
<td align="center" valign="top" width="25%" style="border: none; border-right: 1px solid rgba(128, 128, 128, 0.3); padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/FrameFusion">
<img src="https://raw.githubusercontent.com/thu-nics/FrameFusion/main/example/image/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/FrameFusion"><b>FrF</b></a>
<br/><sub>Efficient video token reduction for LVLMs</sub>
</td>
<td align="center" valign="top" width="25%" style="border: none; padding: 10px; min-width: 50px;">
<div style="height: 5em; display: flex; align-items: center; justify-content: center;">
<a href="https://github.com/thu-nics/MoA">
<img src="https://raw.githubusercontent.com/thu-nics/MoA/master/resource/logo.png" style="max-height: 5em; max-width: 100%; height: auto; width: auto;" />
</a>
</div>
<a href="https://github.com/thu-nics/MoA"><b>MoA</b></a>
<br/><sub>Mixture of sparse attention for LLMs</sub>
</td>
</tr>
</table>

