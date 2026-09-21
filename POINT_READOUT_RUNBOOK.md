# 真实场生成式问答 v2：完整实验流程

冻结 Qwen2.5-14B-Instruct，根据整个标准化场和自然语言问题直接生成数值或坐标。
复用现有逐格 CNN、二维 spatial adapter、数值编码和交叉注意力；场侧模块从头训练。
不需要旧服务器的 Stage 1、Direct-QA 或选择题 checkpoint。原选择题入口保持独立。

## 1. 任务、标签与评分

| 任务参数 | 内容 | 输出 |
|---|---|---|
| `single_point` | 指定位置的值 | `[0.3]` |
| `multi_point` | 2–4 个指定位置的值 | 按问题顺序输出数组 |
| `line_profile` | 水平/竖直连续 3–5 点 | 从起点依次输出数组 |
| `region_values` | 默认 2×2 区域各点的值 | 逐行从左到右输出数组 |
| `region_mean` | 区域均值 | `[value]` |
| `region_std` | 区域总体标准差 | `[value]`，方差除以 N |
| `region_min` | 区域最小值 | `[value]` |
| `region_max` | 区域最大值 | `[value]` |
| `region_argmin` | 区域最小值的位置 | `[row, column]` |
| `region_argmax` | 区域最大值的位置 | `[row, column]` |
| `nearest_value_location` | 区域内最接近给定标准化值的位置 | `[row, column]` |

统计/定位区域默认从 2×2、4×4 中采样，可修改 `generation.statistic_region_shapes`。
坐标从 1 开始，相对于完整输入场；区域由左上角、高、宽确定。问题采用英文并随机选择模板。
全部任务均在完整词表上自回归生成，没有选项、候选打分或任务专用预测头。
网络仅接收完整场，以及由 `grid_shape`、`question` 构造的文本。结构化查询、任务 ID、oracle 仅用于标签与审计。

标签从真实场计算：先对**整个输入裁剪**做 float32 总体 z-score（分母加 1e-6），
经 FP16 存储再恢复为 float32。读取、统计都以此为准，区域内部不再次归一化。
统计量使用 float64 聚合。数值参考答案保留 1 位小数，评分比较未舍入真值，绝对误差 ≤0.2 算正确。
维持近似读取目标，不增加原始物理单位或高精度恢复任务。

坐标题要求两个整数值，完整坐标对正确才得分，不采用 0.2 容差，也不拆开行列给分。
接受等价整数写法，如 `2.0`。并列极值/最近点均可答，训练文本采用行优先的第一个位置。
最近值任务先将目标舍入为题目可见的数值，再据此寻找最近点，避免隐藏精度改变标签。
“并列”按存储场上的精确数值判定；相近但不同的极值仍有唯一正确位置。

输出须为单个非空 JSON 数字数组；额外解释、布尔值、非有限值、数量不符、达到生成上限仍无 EOS 均判错。
多点题保留逐值正确率，并报告整题全对率；缺失预测仍计入分母。

主要指标：

- `by_task_type.*.answer_accuracy`：每类任务的整题正确率。
- `macro_task_answer_accuracy`：各任务整题正确率的等权平均。best 只使用验证集 **seen_shapes** 的该指标选择。
- `by_answer_kind`：数值/坐标分组；`by_shape`、`by_field`：形状/物理量分组。
- `valid_rate`：格式与生成完整性通过率。
- `unit_accuracy`：按评分单元加权，一个数值或一个完整坐标对各为一单元。
  兼容字段 `point_accuracy` 与其相同，`values` 等于 `scoring_units`，不要统称为“点值精度”。

统计任务包含聚合难度，精确坐标定位也比近似读值严格。应逐任务解释结果：局部平滑可能让均值题容易，
坐标分低也不直接说明数值读取失效。首轮关注训练闭环、格式有效率及各类任务的可学习性。

## 2. 数据协议与训练量

HDF5 选定字段共用 `[trajectory, time, height, width]` 布局，至少 20 条独立轨迹。
先按轨迹划分 80%/10%/10%，再采帧和裁剪，同一轨迹不跨训练、验证、测试。
一个裁剪称为一个 state，默认每个 state 生成 11 类题各一道。因此题数不等于独立物理场数。
保存来源、归一化参数、原始/标准化裁剪哈希和 JSONL 清单哈希；审计回放所用裁剪。

| profile | 训练 states | 训练题数 | 每个 val/test 题数 | 更新次数 |
|---|---:|---:|---:|---:|
| smoke | 32 | 352 | 88 | 固定 50 |
| pilot | 512 | 5,632 | 528 | 2,816 |
| full | 2,048 | 22,528 | 2,112 | 11,264 |

smoke 形状为 8×8、8×16，无未见形状。pilot/full 训练 8×8、8×16、16×8、16×16，
额外评估 12×20、20×12。可变形状此前已建立，本轮先控制规模研究生成任务。
可修改 YAML 形状，但查询区域须容纳于每个形状内。

默认单卡 batch=1、累积=4。pilot/full 的 `max_updates: null` 按实际题量计算两个 epoch；
选任务子集后自动缩减步数。其他 batch 设置按同形状 batch 数计算，末尾不足一次累积时最多多取
`gradient_accumulation_steps-1` 个 batch。显式 `max_updates` 优先于 epoch 数。
损失为每题等权的答案 token 交叉熵加 0.01 倍重建损失，提示词不计入生成损失。

**v2 需重建数据、开始新训练，不与 v1 数据/checkpoint 混用。** 新目录为 `data/field_qa_v2/`，
构建器拒绝覆盖非空目录。

## 3. 本机上传 GitHub

在本机仓库根目录执行：

```powershell
git status --short
git add --dry-run .
git add .
git diff --cached --stat
git commit -m "Extend real-field generative QA tasks and server workflow"
git push
```

ignore 规则涵盖数据、权重、输出、缓存、环境、日志、下载中间文件、压缩包、办公稿件和常见密钥。
本路线代码、配置、依赖、测试、部署脚本和本文档均允许上传；保留原有历史脚本白名单。
本机配置放 `configs/local_*.yaml`，结果放 `outputs/` 或外部资产目录。
已被 Git 跟踪的文件不受新增 ignore 影响；dry-run 用于核对待加入文件。

## 4. 全新 Linux 服务器

下面以 Ubuntu 22.04/24.04、单卡 RTX A6000 48GB 为例。首先需有能正常运行 `nvidia-smi` 的 NVIDIA 驱动；
否则先由管理员完成驱动安装。PyTorch wheel 提供 CUDA 用户态库，不要求另装 nvcc。
为模型、数据、缓存和 checkpoints 建议预留至少 150GB 可用空间。完整 14B/A6000 的显存、耗时尚未实测，以 smoke 为准。

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-venv tmux
nvidia-smi
df -h
git clone https://github.com/lszsnmlmzsby/tensor-compression.git
cd tensor-compression
git rev-parse HEAD
PYTHON_BIN=python3 TORCH_CHANNEL=cu124 bash scripts/setup_field_qa_server.sh
source .venv/bin/activate
```

私有库需先配置自己的 GitHub 凭据，不要将 token 写入仓库或 clone URL。
脚本支持 Python 3.10–3.12，在 `.venv` 安装并执行 `pip check`。
默认 PyTorch 2.5.1/cu124；新架构 GPU 且驱动支持时可选 `TORCH_CHANNEL=cu128`（PyTorch 2.9.1）。
本地测试环境是 PyTorch 2.9.1/cu128，cu124 配方沿用项目依赖，服务器上仍需运行下面的测试确认。
专用依赖为 `requirements-field-qa.txt`，避免向此环境叠加其他框架。

## 5. 路径、模型和数据

每个新 shell/tmux 窗口进入仓库后重新执行：

```bash
source .venv/bin/activate
export FIELD_TO_LLM_ROOT="$HOME/field_to_llm_assets"
export FIELD_TO_LLM_MODEL_DIR="$FIELD_TO_LLM_ROOT/models/Qwen2.5-14B-Instruct"
export FIELD_TO_LLM_HF_HOME="$FIELD_TO_LLM_ROOT/hf_cache"
export HF_HOME="$FIELD_TO_LLM_HF_HOME"
export PDEBENCH_HDF5="$FIELD_TO_LLM_ROOT/raw/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
mkdir -p "$FIELD_TO_LLM_ROOT/raw" "$FIELD_TO_LLM_ROOT/logs" "$FIELD_TO_LLM_ROOT/environment"
```

下载公开模型与单个真实 CFD 文件：

```bash
hf download Qwen/Qwen2.5-14B-Instruct --local-dir "$FIELD_TO_LLM_MODEL_DIR"
python scripts/download_pdebench_field.py --output-dir "$FIELD_TO_LLM_ROOT/raw"
python scripts/check_field_qa_environment.py --require-cuda \
  --hdf5-path "$PDEBENCH_HDF5" \
  --output "$FIELD_TO_LLM_ROOT/environment/server.json"
python -m pip freeze > "$FIELD_TO_LLM_ROOT/environment/pip-freeze.txt"
git rev-parse HEAD > "$FIELD_TO_LLM_ROOT/environment/code-commit.txt"
python -m pytest tests/test_point_readout.py -q
```

下载器采用官方清单固定 URL/MD5，curl 断点续传，校验通过后才将 `.part` 改为正式 HDF5。
重复执行会校验已有文件。MD5 不符时保留中间文件；确认损坏后移走，再下载。
不需要整套 2D CFD。文件名 `Rand` 指 CFD 初始条件，这里使用数值求解的 PDEBench 场，不引入 IID/相关随机场实验。
已有数据可直接设置 `PDEBENCH_HDF5`。只有部分变量时，构建和环境检查都加 `--fields Vx Vy` 等实际字段名。
检查器验证形状，轴的物理含义须由来源确认，不能将任意轴排列视为兼容。

来源：[PDEBench 官方清单](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/pdebench_data_urls.csv)、
[Qwen 模型](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)、
[PyTorch 安装说明](https://docs.pytorch.org/get-started/previous-versions/)。

## 6. Smoke：检查完整闭环

```bash
python scripts/build_point_readout_qa.py --profile smoke
python scripts/train_point_readout.py --profile smoke --audit-only \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_smoke_audit"
set -o pipefail
python -u scripts/train_point_readout.py --profile smoke \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_smoke" \
  2>&1 | tee "$FIELD_TO_LLM_ROOT/logs/field_qa_v2_smoke.log"
```

确认审计通过、loss/梯度有限、50 步完成，并产生验证预测和 best/last checkpoint。
50 步用于验证训练推理，不要求此时准确率可用。显存不足先检查其他进程，默认已开启梯度检查点且 batch=1。
修改配方后应另起实验目录，不可直接恢复原 checkpoint。

## 7. Pilot：先测耗时，再继续训练

长任务可先 `tmux new -s fieldqa`，在新窗口执行第 5 节环境设置。Ctrl-b、d 脱离，`tmux attach -t fieldqa` 返回。
单卡入口用 python，不用多卡 torchrun。新 shell 使用日志管道前执行 `set -o pipefail`。

```bash
python scripts/build_point_readout_qa.py --profile pilot
python -u scripts/train_point_readout.py --profile pilot \
  --stop-after-updates 100 \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot" \
  2>&1 | tee "$FIELD_TO_LLM_ROOT/logs/field_qa_v2_pilot_first100.log"
```

完整计划仍为 2,816 步，本次只跑 100 步并保存 last，不压缩学习率计划。
从 `train.jsonl` 和 `run_summary.json` 估计耗时；验证的自回归生成有额外开销。
100 步通常尚未验证，可能没有 best。然后继续：

```bash
python -u scripts/train_point_readout.py --profile pilot \
  --resume "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot/last.pt" \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot" \
  2>&1 | tee "$FIELD_TO_LLM_ROOT/logs/field_qa_v2_pilot_resume.log"
```

恢复权重、优化器、学习率、数据位置和随机数状态，要求数据、模型权重、配方一致。
Ctrl-C/SIGTERM 请求在当前优化器更新完成后保存退出；验证期间收到信号会完成验证后退出。
SIGKILL、断电、CUDA 崩溃只能恢复最近已保存 checkpoint。
pilot 每 500 步验证、100 步保存；验证前也保存 last，保护已完成进度。
如果停在验证之前，恢复时会补做该次验证，包括已经达到最终训练步数的情况。

## 8. Full 与任务选择

```bash
python scripts/build_point_readout_qa.py --profile full
python -u scripts/train_point_readout.py --profile full \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_full" \
  2>&1 | tee "$FIELD_TO_LLM_ROOT/logs/field_qa_v2_full.log"
```

默认 11,264 次更新，每 1,024 步验证、256 步保存。full 是独立训练，不能把 pilot checkpoint 当作 full 的严格续训起点。
可先用 `--stop-after-updates 100` 再同目录 resume。改规模时调整 `generation.train_states` 或 `training.epochs`。
将配置复制到 `configs/local_field_qa.yaml` 后，构建/训练均加 `--config configs/local_field_qa.yaml`，使用新的 QA/输出目录。

命令行选任务示例：

```bash
python scripts/build_point_readout_qa.py --profile pilot \
  --tasks single_point region_mean region_argmax \
  --output-dir "$FIELD_TO_LLM_ROOT/data/field_qa_v2/pilot_selected"
python scripts/train_point_readout.py --profile pilot \
  --qa-dir "$FIELD_TO_LLM_ROOT/data/field_qa_v2/pilot_selected" \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot_selected"
```

训练遵循不可变数据集的任务列表，此例 1,536 题、768 次更新。首轮建议默认完整任务集；此参数不要求增加消融实验。

## 9. 独立测试、离线评分与汇报

以 pilot 为例，full 时替换 profile、QA 和 run 路径：

```bash
python scripts/train_point_readout.py --profile pilot \
  --resume "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot/best.pt" \
  --evaluate-only --split test \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot_test"
python scripts/score_point_readout.py \
  --qa-dir "$FIELD_TO_LLM_ROOT/data/field_qa_v2/pilot" --split test \
  --predictions "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot_test/test_predictions.jsonl" \
  --output "$FIELD_TO_LLM_ROOT/runs/field_qa_v2_pilot_test/offline_metrics.json"
```

独立评估使用新空目录。测试集不用于选模型；验证同时列出已见/未见形状，但 best 只由已见形状分数决定。
离线评分不加载 Qwen 或打开 HDF5，使用不可变清单标签。线上结果另有已见/未见形状及资源统计，
两者的总指标、每类任务指标应一致。

每轮保留：

- `metadata.json`、`states.jsonl`：轨迹划分、裁剪来源、题量、任务分布。
- `contract.json`、`resolved_config.json`、`architecture.json`、`training_plan.json`：实验身份、结构与预算。
- `train.jsonl`、`validation.jsonl`、`run_summary.json`：曲线、验证分数、完成状态。
- `test_metrics.json`、`test_predictions.jsonl`：指标与原始生成答案。
- `environment/`：版本、GPU、Git 提交号。

汇报各任务正确率与格式有效率，并展示读值、统计、坐标各类成功/失败样例，判断错误来自格式、数值、区域理解还是坐标定位。

## 10. 本地验证范围

生成式问答测试覆盖全部任务的标签回放、区域统计、并列坐标、严格评分、任务子集、数据隔离、自动预算、
真实小型 Qwen 的反向传播与 KV cache、CUDA BF16、暂停恢复一致性和独立测试评分。
安装脚本已通过 Bash 语法检查，`git add --dry-run .` 已核对仅列出本次代码/配置/文档。
尚未执行完整 Qwen 14B 在 A6000 上的训练，也未在此 Windows 环境实际运行 Linux 安装脚本。

全仓库回归另有两个既有 Stage 1 测试失败，均已在修改前提交 `b20fc55` 的独立快照复现：
`test_non_reentrant_checkpointed_readout_keeps_student_gradient_graph`（检查点重算保存张量数不一致）和
`test_config_defaults_keep_primary_out_of_lower_auxiliary_layers`（辅助层损失配置检查）。
本路线不经过旧 Stage 1 训练入口；服务器先运行第 5 节指定的问答测试。
