init dataset conversion
```
python script/data_labeling/init_dataset_conversion.py --dataset_config aime,gpqa_extended,Bespoke-Stratos-17k-Math,Bespoke-Stratos-17k-Code,Bespoke-Stratos-17k-QA --output_dir unified_datasets
```
STEP 0: LLM response
Use dataset from init dataset conversion 
```
CUDA_VISIBLE_DEVICES=0,1 python script/data_labeling_api/step_0_llm_response.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path unified_datasets/combined_dataset --output_dir unified_datasets/output --tp_size 2 
```
Use dataset from R2R_dataset
```
CUDA_VISIBLE_DEVICES=0,1 python script/data_labeling/step_0_llm_response.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --dataset_path R2R_dataset/train --output_dir unified_datasets/output --tp_size 2
```
STEP 1: SLM prefill
```
CUDA_VISIBLE_DEVICES=0 python script/data_labeling/step_1_slm_prefill.py --dataset_path r2r_output/data_gen/Bespoke-Stratos-17k-Math-Code-QA/dataset_finished --test_model_list deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --output_path r2r_output/data_gen/Bespoke-Stratos-17k-Math-Code-QA/SLM_prefill
```
STEP 2: LLM continuation
```
CUDA_VISIBLE_DEVICES=0,1 python script/data_labeling/step_2_llm_continuation.py --input_path unified_datasets/output/SLM_prefill/prediction_comparison.csv --output_path unified_datasets/output/SLM_prefill/LLM_continuation_verify
```
STEP 3: Verify
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python script/data_labeling/step_3_verify.py --input_csv unified_datasets/output/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full.csv --output_csv unified_datasets/output/SLM_prefill/LLM_continuation_verify/generation_results_data_all_real_full_verify.csv --tp_size 4
```
STEP 4: Generate dataset
```
python script/data_labeling/step_4_construct_label_dataset.py --data_dir unified_datasets/output/SLM_prefill --csv LLM_continuation_verify/generation_results_data_all_real_full_verify.csv --output_sub_folder LLM_continuation_verify/divergent_label_dataset  --divergent_column_name divergent
```