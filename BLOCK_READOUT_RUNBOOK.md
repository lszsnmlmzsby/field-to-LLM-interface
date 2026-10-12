# 多点一个 token：真实场可行性实验

使用本手册的托管服务器流程、Qwen 及 PDEBench 数据。只新增局部块编码，
冻结 Qwen，保留 11 类生成任务、单小数位训练答案和绝对误差 0.2 的评分规则。

## 结构与兼容性

`--block-shape H W` 指每个场 token 覆盖的原始点数，和 `data.patch_size`（初始化尺寸）无关。
默认 1×1 仍使用原有参数形状和计算路径；旧配置、旧生成式检查点不用修改。
2×2 时，CNN 不下采样。CNN 后按左上、右上、左下、右下拼接四个位置的特征，
附加四个有效位标志，再投影到 512 维。在块 token 上运行两层空间 Transformer。
位置编码使用每个块左上角在原场中的绝对行列位置，不把块编号当作原场坐标。

独立数值分支从原始标准化输入读取四个 z 值，逐点构造 Fourier 等特征，按同一顺序拼接，
加有效位标志后经 MLP 映射到 512 维。辅助重建头输出四个数，SmoothL1 仅统计有效点。
块内顺序保存在特征槽位中，不取平均、不根据问题挑点，也不向模型提供 oracle。
输入维数为 K×(4+2×Fourier频带数)+K，其中 K=块内点数；1×1 走原路径，不添加 mask 维。
空间分支输入维为 K×CNN输出通道数+K。压缩后交叉注意力接口保持 512 维，Qwen 不变。

奇数形状在 CNN 后仅向右/下补齐；所有补齐位置都有 mask，不参与数值重建。
标准化、标签、问答坐标始终基于未补齐原场。每个块至少有一个真实点，无全空块 token。
token 数为 ceil(H/ph)×ceil(W/pw)。17×90 用 2×2 得到 405 tokens，而不是整除的 382.5。
这里减少的是接口 token 数，不宣称无损压缩，也不保证能精确重建或逐点读取。

压缩比例进入训练身份校验；1×1 与 2×2 的检查点不能互相续训。不得通过非严格加载绕过。
新实验从头训练接口，不使用旧小场 full 检查点作为未压缩对照。

## 数据与预算

新配置：`configs/field_to_llm_block_readout.yaml`。只构建一次同一 profile 的数据供两组共用。

| profile | 训练形状 | 训练场 | 问答数 | 更新次数/组 |
|---|---|---:|---:|---:|
| smoke | 8×8、9×17（检查边界） | 32 | 352 | 20 |
| pilot（本次主实验） | 16×16、16×32、32×16、32×32 | 1,024 | 11,264 | 5,632 |
| full（以后扩展） | 同 pilot | 4,096 | 45,056 | 22,528 |

pilot/full 未见形状：16×64、64×16、17×90、90×17。原 HDF5 两轴至少 90 才能构建完整实验。
pilot 验证和测试各 64 场、704 问答（每种形状 8 场）；轨迹仍按 80/10/10 隔离。
pilot 每 2,816 次更新验证，5,632 次结束；batch=1、累积=4、两轮。任务和学习率两组一致。
full 每种评估形状 32 场，验证和测试各 2,816 问答；每 1,024 次更新验证，加末尾验证共 22 次/组。
先完成 pilot，再决定是否投入 full。

## 服务器命令

当前资源是一张 A800 80 GB，PyTorch 2.4.1+cu124；入口是单设备训练，使用 `python`，不用多进程 `torchrun`。
两组顺序运行。80 GB 能否覆盖真实 14B、最大场形状和验证阶段，仍需实测。
继续使用已申请环境；不运行 `setup_field_qa_server.sh`，不直接安装仓库的完整 requirements，避免替换平台 Torch/CUDA。
包的“available=True”只说明可以找到包，不能证明 Qwen 实现可导入或版本相容。

目录固定为：

```text
/workspace/wyx/f2llm/
  tensor-compression/                  # Git 仓库
  tensor_llm_assets/raw/               # PDEBench 原始 HDF5
  tensor_llm_assets/data/              # 固定的 QA 数据清单
  tensor_llm_assets/models/            # 本地 Qwen 权重和 tokenizer
  tensor_llm_outputs/runs/             # 检查点、训练日志和评估
  tensor_llm_outputs/logs/             # 终端输出日志
  tensor_llm_outputs/environment/      # 环境检查结果
  hf_cache/                           # Hugging Face 缓存
```

每次登录先 `source`（不能用 `bash` 代替，因为环境变量要留在当前 shell）。
如果新容器提示 git/curl/tmux 不存在，可在当前 root 容器按平台允许的方式恢复这些工具：

```bash
apt-get update
apt-get install -y git curl tmux
source /workspace/wyx/f2llm/enter_f2llm.sh
git status --short
git rev-parse HEAD
```

服务器应包含 `21f9d79` 的实验实现，以及本手册配套的最新环境检查脚本。
如果工作区干净、远端已有需要的更新，用 `git pull --ff-only` 同步；有本地修改时先检查，不强制覆盖。
以下命令中的新 `--model-dir` 检查参数需要先同步本次修改到服务器。

```bash
export BLOCK_CONFIG=configs/field_to_llm_block_readout.yaml
df -h "$F2LLM_HOME"
free -h
python scripts/check_field_qa_environment.py --require-cuda \
  --model-dir "$FIELD_TO_LLM_MODEL_DIR" \
  --output "$F2LLM_HOME/tensor_llm_outputs/environment/before_data.json"
```

检查实际 Qwen 类导入、本地配置、tokenizer/chat template，以及索引引用的全部 safetensors 分片。
分片检查读取头部及索引，不加载 14B 权重，也不代替官方的全文件校验和。
需要同时保留原数据（约 51.2 GiB）、14B 权重、缓存和输出空间；不能只看 GPU 显存。
模型加载还需要主机 RAM；`free -h` 之外也应确认平台给当前容器的 RAM 配额。
当前 `pip check` 的 ninja 平台提示需要记录，但这个实验没有调用 ninja 编译扩展，先以导入/测试结果判断。

下载一份 2D CFD Rand 128 数据即可，不需要整个 PDEBench。源地址和 MD5 来自官方
[下载清单](https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/pdebench_data_urls.csv)。
下载可以在没有 GPU 的准备时段执行（若平台提供），tmux 只防 SSH 断开，不延长资源租期。

本实验下载完整的官方 HDF5，通过官方整文件 MD5 校验后再构建 QA；局部缓存或未完成文件不能替代它。

```bash
python -u scripts/download_pdebench_field.py \
  --output-dir "$FIELD_TO_LLM_ROOT/raw"
python scripts/check_field_qa_environment.py --require-cuda \
  --model-dir "$FIELD_TO_LLM_MODEL_DIR" --hdf5-path "$PDEBENCH_HDF5" \
  --output "$F2LLM_HOME/tensor_llm_outputs/environment/before_training.json"
python -m pytest tests/test_block_readout.py tests/test_point_readout.py -q
```

下载中断后重复同一命令，会保留并续传 `.part`，MD5 成功后才变成正式 HDF5。
若之前已用 aria2 分块下载该 `.part`，应继续使用 aria2 及其 `.aria2` 控制文件，不能切回按文件长度续传的 curl。
若校验失败，先检查部分文件，不绕过校验直接构建训练数据。
训练用的 `data/` 是从 `raw/` 生成的 JSONL 清单，场数据仍从原始 HDF5 按需读取；训练时不能移动/删除原文件。

先检查训练/生成闭环，包含奇数形状。构建器拒绝覆盖已有数据；已经成功构建则跳过 build。
输出目录也必须是新的。审计会重放 train/val/test 标签，但不进行测试集预测或用测试集选检查点。

```bash
python scripts/build_point_readout_qa.py --config "$BLOCK_CONFIG" --profile smoke
python scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile smoke \
  --audit-only --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_audit_smoke"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile smoke \
  --device cuda --block-shape 1 1 --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_smoke_1x1"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile smoke \
  --device cuda --block-shape 2 2 --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_smoke_2x2"
```

主实验先分别做 200 次更新的计时段。它使用正式 pilot 形状和原定两轮的学习率日程；
不使用随机场或改小 epochs/max_updates 来估时。每组使用独立的新输出目录，数据只生成一次。

```bash
python scripts/build_point_readout_qa.py --config "$BLOCK_CONFIG" --profile pilot
python scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --audit-only --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_audit_pilot"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --device cuda --block-shape 1 1 --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_pilot_1x1" \
  --stop-after-updates 200
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --device cuda --block-shape 2 2 --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_pilot_2x2" \
  --stop-after-updates 200
```

另一个终端可用 `nvidia-smi -l 5` 观察峰值和进程。200 次更新尚不触发 pilot 的完整验证；
smoke 也不覆盖最大评估场。因此，这两项通过后仍应监控 pilot 第一次包含 17×90/90×17 的验证。

在首次 200 次更新结束、尚未续训时，可从第 20 到第 200 次更新估算纯训练时间：

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['FIELD_TO_LLM_RUNS_DIR'])
for name in ['block_pilot_1x1', 'block_pilot_2x2']:
    path = root / name
    rows = [json.loads(line) for line in (path / 'train.jsonl').read_text().splitlines()]
    rows = [row for row in rows if 20 <= row['step'] <= 200]
    if len(rows) < 2:
        raise ValueError(f'{name}: need at least two timing records')
    first, last = rows[0], rows[-1]
    seconds = (last['elapsed_seconds'] - first['elapsed_seconds']) / (last['step'] - first['step'])
    updates = json.loads((path / 'training_plan.json').read_text())['planned_updates']
    print(f'{name}: {seconds:.2f} s/update; {updates * seconds / 3600:.2f} h training only')
PY
```

每组 pilot 总时间约为 `5632 × 秒/更新 + 2 × 完整验证秒数 + 加载/保存开销`，两组顺序运行则相加。
验证耗时记录在 `validation.jsonl` 的 `elapsed_seconds`；首次 200 步只能测训练部分。
不要把 smoke 的更新速度外推到 pilot，也不要默认 2×2 的总训练速度会快四倍，Qwen 文本解码仍保留。
`train.jsonl` 的计时在每次进程启动时归零，上述估算不能混用不同续训段的首末行。

完成计时后，去掉停止参数、加 `--resume last.pt`，两组依次继续，直到 5,632 次更新及末尾验证结束：

```bash
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --device cuda --block-shape 1 1 --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_pilot_1x1" \
  --resume "$FIELD_TO_LLM_RUNS_DIR/block_pilot_1x1/last.pt"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --device cuda --block-shape 2 2 --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_pilot_2x2" \
  --resume "$FIELD_TO_LLM_RUNS_DIR/block_pilot_2x2/last.pt"
```

普通中断信号会在完整 optimizer 更新后保存；强制结束/租期结束可能回退到最后一次保存。
资源到期前主动停止并确认 `last.pt`/`run_summary.json` 已落盘，换同一环境后严格续训，不跨组加载。

独立评估两个 best（避免将训练期显存和独立推理显存混用）：

```bash
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --device cuda --block-shape 1 1 --evaluate-only --split val \
  --resume "$FIELD_TO_LLM_RUNS_DIR/block_pilot_1x1/best.pt" \
  --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_eval_1x1"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --device cuda --block-shape 2 2 --evaluate-only --split val \
  --resume "$FIELD_TO_LLM_RUNS_DIR/block_pilot_2x2/best.pt" \
  --output-dir "$FIELD_TO_LLM_RUNS_DIR/block_eval_2x2"
python scripts/compare_block_readout.py \
  --reference "$FIELD_TO_LLM_RUNS_DIR/block_eval_1x1" \
  --compressed "$FIELD_TO_LLM_RUNS_DIR/block_eval_2x2"
```

比较工具要求除 block_shape 外训练身份一致，并校验双方问题 ID 集合。输出每任务整题/逐元素准确率、
seen/heldout、场 token 范围、推理峰值分配显存、总耗时及生成 token 数。
耗时包含完整评估过程，且两组回答长度可能不同，不能直接作为正式加速比。
形状和物理量细分见各自 `val_metrics.json` 的 `by_shape` / `by_field`。
重点看四个读取任务；统计题的高分不能代替逐点能力。当前报告不包含自动划分块内/跨块查询的分析。
完成方案选择后，再以新目录将评估命令的 `--split val` 改为 `--split test`，比较器也传 `--split test`。

## 交互查看压缩模型

```bash
python -u scripts/chat_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --block-shape 2 2 --split val --mode both --response-format json --single-turn \
  --checkpoint "$FIELD_TO_LLM_RUNS_DIR/block_pilot_2x2/best.pt"
```

这里的 both 仍然表示“原始 Qwen 文本矩阵 vs 压缩接口”。1×1 和 2×2 接口之间的比较使用上面的独立评估。
未来需要文本 baseline 时，benchmark 脚本也支持 `--config "$BLOCK_CONFIG" --profile pilot --block-shape 2 2`
来配对压缩接口预测。完整文本矩阵若超出 prompt/context 限额会报错，不允许截断。
压缩配置不改变 baseline 的输入文本，仅用于核对被比较接口的身份。

## 文件管理与验证范围

新脚本/配置/测试已加入 Git 白名单；数据、权重、检查点及 JSONL 继续被忽略。
训练产生 `resolved_config.json`、`architecture.json`、`train.jsonl`、`validation.jsonl`、`best.pt`、`last.pt`。
块大小记录在配置和 architecture 中，每条评估预测记录 field_tokens。
本地 CPU 小型 Qwen 测试验证槽位排列、奇数边界、梯度、冻结主干、保存恢复和旧 1×1 路径；
本地验证环境的 Torch 与服务器 Torch 2.4.1 不同，因此服务器仍需运行依赖导入、测试和 smoke。
真实 Qwen 14B 的 A800 80 GB 显存、速度和精度需要服务器 smoke/pilot 实测。
