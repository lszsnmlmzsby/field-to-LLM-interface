# Field-to-LLM

Field-to-LLM 将二维科学场编码为保留网格位置与标量值的连续记忆，通过交叉注意力接入冻结的 Qwen。模型根据自然语言问题生成数值或坐标答案，覆盖点值读取、区域统计、极值定位和组合查询。

数据构建、训练、评估、推理与性能诊断统一从 `python field.py <命令>` 调用。当前多源实验支持 `1–128` 的整数高宽、14 类查询，以及 `1×1`、`2×2` 场 token 结构。

## 安装与目录

Linux 服务器使用 Python 3.10–3.12，安装脚本创建项目内虚拟环境：

```bash
bash setup.sh
source .venv/bin/activate
python field.py environment --output outputs/environment.json
```

`TORCH_CHANNEL=cu124|cu128|cpu` 选择安装通道，默认 `cu124`。已有 PyTorch 环境使用 `python -m pip install -r requirements.txt`。依赖集中在这一份清单中。

```text
field.py                  统一命令入口
setup.sh                  Linux 环境安装
requirements.txt          Python 依赖
configs/                  实验配方与固定源清单
src/tensor_compression/
  workflows/              应用流程及运行时支持
  downstream/             场读取、查询真值、QA 契约、采样与缓存
  models/                 场记忆、空间编码器、Qwen 加载与模型构建
  data/                   归一化
  utils/                  配置与文件工具
tests/                    功能、数值一致性与恢复测试
docs/sources.md            源规模、字段布局与校验方式
```

入口分发参数，应用流程调用公共实现；评估与训练共用模型加载和生成逻辑。运行资产使用独立路径：

```bash
export FIELD_TO_LLM_ROOT=/absolute/path/to/field_to_llm_assets
export FIELD_TO_LLM_MODEL_DIR=/absolute/path/to/Qwen2.5-14B-Instruct
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
CONFIG=configs/field_to_llm_multisource_v2.yaml
RUN="$FIELD_TO_LLM_ROOT/runs/multisource_v2_1x1"
CACHE="$FIELD_TO_LLM_ROOT/data/multisource_openbc_v2/full_training_arrays_v1"
```

## 数据准备

当前多源配方使用完整 INS-Tracer、Acoustic inclusions、Euler openBC 和 Gray–Scott。下载器按固定清单获取文件并生成校验记录：

```bash
python field.py download-ins --output-dir "$FIELD_TO_LLM_ROOT/raw/ins_tracer"
python field.py download-well --output-dir "$FIELD_TO_LLM_ROOT/raw/the_well"
```

各源规格见[数据源说明](docs/sources.md)。

```bash
python field.py data --config "$CONFIG" --profile full
python field.py prepare --config "$CONFIG" --profile full --output-dir "$CACHE"
```

`data` 构建 train/val/test QA、state 索引和精确场缓存。`prepare` 校验完整训练 QA，生成 mmap 场数组与预分词 token。缓存绑定 QA、tokenizer、模板和 token 上限，同一协议的不同场 token 结构可以共享缓存。源文件、实验裁剪、QA 和训练缓存分别记录身份。

`smoke`、`pilot`、`full` 使用配置中各自声明的数据和训练预算；正式实验使用 `full`。恢复运行时使用其 `resolved_config.json`。

## 训练

### 启动与恢复

下面以三个训练进程、`1×1` 场 token 为例：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  field.py train --config "$CONFIG" --profile full \
  --block-shape 1 1 --output-dir "$RUN" \
  --training-cache "$CACHE" --input-workers 2 \
  --prefetch-updates 2 --prefetch-mib 256
```

`--block-shape 2 2` 选择块 token，并使用独立运行目录。全局 batch、梯度累积、学习率和轮数读取配置。`best.pt` 保存验证选出的模型，`last.pt` 保存完整续训状态。

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  field.py train --config "$RUN/resolved_config.json" --profile full \
  --resume "$RUN/last.pt" --output-dir "$RUN" \
  --training-cache "$CACHE" --input-workers 2 \
  --prefetch-updates 2 --prefetch-mib 256
```

进程数按实际分配填写。改变 checkpoint 的进程数时，加 `--allow-world-size-change` 记录迁移；更换硬件后可用 `--reset-micro-batch-limits` 重建小批次容量记录。全局批次、数据位置和剩余预算继续恢复。Ctrl+C 在完成当前更新后保存退出。

### 输入预取与性能定位

每 rank 默认两个读取进程、两次更新的准备窗口和 256 MiB payload 预算。模型、进程、系统页缓存和当前小批次另外占内存。`--input-workers 0` 使用同步 mmap，`--input-workers 1` 降低进程开销。OOM 时用更小的小批次重试同一次更新。

在续训命令中加 `--profile-updates 20 --stop-after-updates 20`，执行原预算内的 20 次更新并保存计时。完成预算的 checkpoint 保持完成状态。

| 输出 | 用途 |
| --- | --- |
| `input_pipeline.json` | 输入配置、启动时间和预算定义 |
| `input_profile.jsonl` | 输入等待、组装、传输、前向与反向计时 |
| `train.jsonl` | 损失、梯度、显存与更新吞吐 |
| `validation.jsonl` | 各次验证结果 |
| `run_summary.json` | 完成状态、step 和预取峰值 |

`phase=...` 区分启动校验、加载与训练。CPU 和 CUDA 计时按阶段比较，重叠区间分别解读；`input_wait_seconds` 包含输入等待与 rank 协调。训练吞吐窗口覆盖取数至 optimizer 更新，验证和保存时间单独观察。

## 评估

```bash
python -m torch.distributed.run --standalone --nproc_per_node=3 \
  field.py evaluate --config "$RUN/resolved_config.json" --profile full \
  --resume "$RUN/best.pt" --split test --output-dir "$RUN/test_best"
```

`evaluate` 自动选择评估模式，输出逐题预测和总体、任务、来源、形状分组指标。训练用 val 选模型，最终 test 使用所选 checkpoint。

`--eval-suite language` 和 `--eval-suite field-pairs` 分别选择语言配对与换场查询；`--input-mode disabled` 检查场输入依赖。每种诊断使用独立输出目录。

```bash
python field.py score \
  --qa-dir "$FIELD_TO_LLM_ROOT/data/multisource_openbc_v2/full" --split test \
  --predictions "$RUN/test_best/test_predictions.jsonl" \
  --output "$RUN/test_best/test_rescored.json"
```

`score` 离线复评分，`baseline` 评估完整数值文本输入的冻结 Qwen，`benchmark` 比较推理开销，`analyze` 分析逐题预测，`compare-blocks` 汇总场 token 结构结果。使用 `python field.py <命令> --help` 查看参数。

## 推理

```bash
python field.py infer --config "$RUN/resolved_config.json" --profile full \
  --checkpoint "$RUN/best.pt" --mode interface --split val --field-index 0 \
  --question "What is the value at row 1, column 1?"
```

省略 `--question` 进入交互模式，`/reset` 清空对话。`--mode baseline|interface|both` 选择数值文本、场记忆或二者对比；`--response-format json|free` 选择回答形式。坐标从 1 开始，数值位于标准化空间。`--transcript` 保存对话记录。

## 实验协议与开发

当前训练配方集中在 `configs/field_to_llm_multisource_v2.yaml`：425,984 个训练裁剪、14 类查询、两轮训练。裁剪配额为 INS 32,768、Acoustic 65,536、Euler openBC 262,144、Gray–Scott 65,536。轨迹划分、形状划分与评分策略由配方及 QA metadata 固定。

`tests/fixtures/` 中的小型配方服务于字段读取、评分与恢复回归测试。正式运行使用上述多源配置。

```bash
python field.py --help
PYTHONPATH=src python -m pytest $(git ls-files 'tests/test_*.py') -q
```

字段和 QA 公共逻辑放入 `downstream`，模型组件放入 `models`，应用编排放入 `workflows`。新增公共命令在 `field.py` 注册；测试校验数据身份、数值结果、跨进程执行和断点恢复。
