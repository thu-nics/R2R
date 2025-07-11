# 智能自动化任务调度器使用指南

`auto_scheduler.py` 是一个完全自动化的 LLM continuation 任务调度器，专为简化大规模数据处理而设计。

## 🚀 主要特性

### 1. **自动GPU检测与分配**
- 自动检测当前可用的 GPU 及其显存状况
- 智能分配 GPU 对给不同的 worker
- 支持动态 GPU 资源管理

### 2. **智能任务恢复**
- 通过 worker 参数文件（`worker_args_*.json`）跟踪任务状态
- 自动检测已启动、已完成或失败的任务
- 跳过已完成的 worker，重试失败的任务
- 支持中断后精确继续执行

### 3. **自动合并结果**
- 所有 worker 完成后自动合并结果文件
- 生成详细的执行报告和统计信息
- 保证数据完整性和一致性

### 4. **实时监控**
- 实时显示任务进度和状态
- 自动处理失败的任务
- 支持优雅的中断和清理

### 5. **详细输出控制**
- 启动前显示每个 worker 的完整执行命令
- 实时显示每个 worker 的输出内容
- 支持静默模式，隐藏 worker 输出
- 多线程安全的输出显示

### 6. **安全进程管理**
- 只清理由当前调度器实例启动的进程
- 通过唯一调度器 ID 跟踪进程归属
- 避免误终止其他调度器或系统进程
- 优雅终止与强制终止的分级处理

## 📋 系统要求

### 必需依赖
```bash
pip install pandas numpy torch
```

### 可选依赖（强烈推荐）
```bash
pip install pynvml  # 用于GPU显存检测
```

如果没有安装 `pynvml`，系统将使用默认的 GPU 检测方案。

## 🔧 基本使用

### 最简单的使用方式
```bash
python script/data_labeling/auto_scheduler.py \
    --input_path output/playground/prediction_comparison.csv \
    --output_path output/playground/continuation_auto \
    --num_workers 4
```

### 完整参数示例
```bash
python script/data_labeling/auto_scheduler.py \
    --input_path output/playground/prediction_comparison.csv \
    --output_path output/playground/continuation_auto \
    --num_workers 8 \
    --gpus_per_worker 2 \
    --min_gpu_memory 12.0 \
    --max_parallel_workers 4 \
    --batch_size 32 \
    --max_new_tokens 8192 \
    --temperature 0.0 \
    --show_worker_output
```

### 静默模式
```bash
# 静默模式 - 不显示worker的详细输出
python script/data_labeling/auto_scheduler.py \
    --input_path output/playground/prediction_comparison.csv \
    --output_path output/playground/continuation_auto \
    --num_workers 4 \
    --quiet
```

## 📝 参数说明

### 核心参数
- `--input_path`: 输入 CSV 文件路径（step 1 的输出）
- `--output_path`: 输出目录路径
- `--num_workers`: 总 worker 数量（必需）

### GPU 配置
- `--gpus_per_worker`: 每个 worker 使用的 GPU 数量（默认：2）
- `--min_gpu_memory`: 最小 GPU 空闲显存要求，单位 GB（默认：10.0）
- `--max_parallel_workers`: 最大并行 worker 数量（默认：8）

### 模型参数
- `--tp_size`: 张量并行大小（默认：2）
- `--dp_size`: 数据并行大小（默认：1）
- `--batch_size`: 批处理大小（默认：32）
- `--max_new_tokens`: 最大生成 token 数（默认：8192）
- `--temperature`: 采样温度（默认：0.0）

### 输出控制
- `--show_worker_output`: 显示 worker 的详细输出和执行命令（默认：开启）
- `--quiet`: 静默模式，隐藏所有 worker 输出（与 `--show_worker_output` 相反）

完整参数列表可通过 `python auto_scheduler.py --help` 查看。

## 🎯 工作流程

### 1. 初始化阶段
```
[INFO] === Starting Automatic LLM Continuation Scheduler ===
[INFO] Loading input data from output/playground/prediction_comparison.csv
[INFO] Found 1000 unique data samples (IDs: 0 to 999)
```

### 2. 任务分配阶段
```
[INFO] Task range: data_id [0, 250) (250 samples)
[INFO] Task range: data_id [250, 500) (250 samples)
[INFO] Task range: data_id [500, 750) (250 samples)
[INFO] Task range: data_id [750, 1000) (250 samples)
```

### 3. GPU 检测阶段
```
[INFO] Available GPUs: [0, 1, 2, 3, 4, 5, 6, 7]
[INFO] Allocated GPU pairs: [[0, 1], [2, 3], [4, 5], [6, 7]]
```

### 4. 任务执行阶段
```
[INFO] Found 2 pending tasks out of 4 total
[INFO] Starting worker for range [500, 750) on GPUs [4, 5]
================================================================================
Worker Command for range [500, 750):
CUDA_VISIBLE_DEVICES=4,5 \
python script/data_labeling/step_2_llm_continuation.py \
    --input_path output/playground/prediction_comparison.csv \
    --output_path output/playground/continuation_auto \
    --low 500 \
    --high 750 \
    --tp_size 2 \
    --dp_size 1 \
    --batch_size 32 \
    --max_new_tokens 8192 \
    --temperature 0.0
================================================================================
[INFO] Worker [500, 750) output:
------------------------------------------------------------
[Worker 500-750] Loading input data...
[Worker 500-750] Processing batch 1/10...
[Worker 500-750] Generated continuations for 32 mismatches
[Worker 500-750] Saved results to output/continuation_auto/generation_results_data_500_to_750_real.csv
------------------------------------------------------------
[INFO] Worker [500, 750) completed successfully
[INFO] Progress: 2/4 completed, 2 running, elapsed: 180.5s
```

### 5. 结果合并阶段
```
[INFO] Starting result merge...
[INFO] Loaded 156 rows from generation_results_data_0_to_250_real.csv
[INFO] Loaded 143 rows from generation_results_data_250_to_500_real.csv
[INFO] Merged 4 files into generation_results_data_all_real_merged.csv
[INFO] Total rows in merged file: 612
```

## 📁 输出文件结构

执行完成后，output 目录将包含：

```
output/playground/continuation_auto/
├── generation_results_data_0_to_250_real.csv      # 单个worker结果
├── generation_results_data_250_to_500_real.csv
├── generation_results_data_500_to_750_real.csv
├── generation_results_data_750_to_1000_real.csv
├── generation_results_data_all_real_merged.csv    # 合并后的最终结果
├── worker_args_0_250.json                         # Worker执行参数和状态
├── worker_args_250_500.json
├── worker_args_500_750.json
├── worker_args_750_1000.json
├── scheduler_config.json                          # 调度器配置
├── merge_summary.json                            # 合并统计信息
└── args.json                                     # 全局执行参数
```

## 🔄 断点续传

调度器支持智能断点续传：

1. **自动检测已完成任务**：重新运行时自动扫描 output 目录
2. **跳过完成的 worker**：只运行未完成或失败的任务
3. **保持数据一致性**：确保不会重复处理相同的数据

示例：
```bash
# 第一次运行（可能中断）
python auto_scheduler.py --input_path data.csv --output_path output --num_workers 8

# 重新运行（自动跳过已完成的任务）
python auto_scheduler.py --input_path data.csv --output_path output --num_workers 8
```

## ⚡ 性能优化建议

### 1. GPU 配置
- 确保每个 GPU 有足够的显存（推荐 ≥ 12GB）
- 根据 GPU 数量合理设置 `--num_workers`
- 使用 `--max_parallel_workers` 控制并发度

### 2. 批处理优化
- 增大 `--batch_size` 提高 GPU 利用率
- 根据显存大小调整 `--max_new_tokens`

### 3. 监控资源
```bash
# 监控 GPU 使用情况
nvidia-smi -l 1

# 监控系统资源
htop
```

## 🛠️ 故障排除

### 常见问题

**Q: "No available GPU pairs found!"**
A: 检查 GPU 状态和显存使用情况，降低 `--min_gpu_memory` 参数

**Q: "Worker failed for range [X, Y)"**
A: 检查输入数据格式，查看错误日志，确认模型路径正确

**Q: 合并结果文件为空**
A: 检查单个 worker 的输出文件，确认任务正确完成

**Q: "TypeError: Object of type int64 is not JSON serializable"**
A: 已自动处理numpy/pandas类型转换，无需手动处理。如仍有问题，请检查输入数据格式

**Q: 担心进程清理会影响其他任务**
A: 调度器使用唯一 ID 跟踪进程，只会清理自己启动的 worker，不会影响其他调度器或系统进程

### 调试模式
```bash
# 启用详细日志
export PYTHONPATH=/path/to/your/project
python auto_scheduler.py --is_print --num_workers 1 [其他参数...]
```

## 📊 性能基准

典型性能表现（8 GPU 服务器）：

| Worker数量 | GPU配置 | 并行度 | 处理速度 | 显存使用 |
|-----------|---------|-------|----------|----------|
| 4         | 2 GPU/worker | 4     | ~100 samples/min | ~20GB/worker |
| 6         | 2 GPU/worker | 4     | ~150 samples/min | ~20GB/worker |
| 8         | 1 GPU/worker | 8     | ~200 samples/min | ~12GB/worker |

## 🔗 相关文档

- [原始脚本生成器](./generate_job_scripts.py) - 手动脚本生成方案
- [分布式执行器](./launch_llm_continuation_multi_node.py) - Ray 分布式方案
- [Step 2 详细说明](./step_2_llm_continuation.py) - 核心处理逻辑

## 📞 技术支持

如有问题，请检查：
1. 输入文件格式是否正确
2. GPU 显存是否充足
3. 依赖包是否完整安装
4. 日志文件中的详细错误信息 