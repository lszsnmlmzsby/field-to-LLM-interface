# 单卡生成式场问答性能测试

独立入口：`scripts/benchmark_readout_efficiency.py`。读取已有训练的 `resolved_config.json`、QA 和 checkpoint；训练目录只读，不启动训练、不评分准确率。默认对照完整数值序列化与训练后的接口，使用同一个冻结 Qwen、同一数值精度和同一批问题。

## 本次选择

| 训练目录（位于 `$FIELD_TO_LLM_RUNS_DIR`） | checkpoint | 来源状态 |
|---|---|---|
| `ins_tracer_full_1x1` | `best.pt` | 已完成训练，逐格接口 |
| `ins_tracer_full_2x2` | `best.pt` | 已完成训练，2×2 接口 |
| `multisource_v2_3gpu_b48_full_1x1` | `last.pt` | 大数据训练中间版本 |

目录来自“评估 apt 安装对服务器的影响”聊天和现有运行手册。脚本会核验服务器上实际存在的文件、QA 身份和模型配置，保存 checkpoint 的 SHA256、step、epoch、cursor；缺少文件时直接报错。读取中间 checkpoint 时通过同一个文件句柄完成哈希与加载，兼容训练器的原子保存。结果标记来源训练状态。

两组 INS 使用相同 QA 和选择种子，因此可逐题比较 1×1 与 2×2。大数据版本使用自身 QA 与生成预算，单独呈现；其同题序列化对照由脚本自动运行。

## 测量口径

- 单卡、batch=1、完整词表 greedy，每个答案内部启用文本 KV cache。每个请求重新标准化、编码输入场并构建 memory，不跨问题复用场特征或 K/V。
- 主计时从内存中的原始数值数组开始，包含标准化、提示构造与分词、数据传输、field encoder、空间 adapter、标量分支、cross-attention、prefill、逐 token 解码和输出解码。使用 GPU 同步的墙钟时间。
- 场到 memory 的计时包含 encoder、adapter 和标量分支；cross-attention 的 K/V 投影在现有实现中延迟到首次使用，计入 prefill。各段相加与总时间会有少量 Python 调度开销差异。
- 原始 HDF5 读取在测试准备阶段完成，读盘耗时另存 `source_read_audit.json`；主结果是“原始 CPU 数组 → 回答”，不含模型加载、文件审计、预热、评分或写盘。标准化调用与原数据相同的函数，其中的小型完整性哈希也计时。
- 多源数据从原始 HDF5 重放，校验所选场与 QA 缓存、归一化记录及题目一致；正式请求内重新标准化。
- 序列化使用完整矩阵和原有精确数值表示，不截断网格、不选择题目涉及的点、不减少数值精度。超过模型上下文的请求记录 `context_exceeded`，显存不足记录 `oom`。
- 序列化先在尚未安装接口的 LLM 上运行，其显存不含闲置接口参数；随后加载同一骨干上的接口。两种方法均单独预热。显存报告总峰值、相对于请求开始时常驻内存的增量峰值及 reserved 峰值。
- 汇总显存取已执行完生成循环的请求峰值；OOM 请求峰值单列为 `oom_peak_allocated_gib`，原始记录完整保留。

默认五档：1–64、65–256、257–1024、1025–4096、4097–16384 个网格点。每档选 24 道既有验证题，按任务/来源轮转抽取，并保留该档实际最大的网格，覆盖已有的 128×128 边界。固定选择种子，两种方法配对；每档不足 24 题时保留全部，空档也在清单中标明。这是独立的性能采样，不改训练集、数据划分、问题或训练预算，也不生成新的裁剪。`selection.json` 保存每道题的形状、ID 和分组。

默认运行两种解码：

1. `natural`：沿用训练配置中的生成上限和 EOS。报告完整回答 p50/p95、TTFT、吞吐、实际生成 token 数、完成/达到长度上限等状态。`completed_answer_*` 只统计正常结束的回答；`request_ms_*` 同时包含达到上限的已执行请求。
2. `fixed`：每题固定生成 32 token，忽略 EOS，作为输入规模与计算开销的辅助对照。单独分组，不作为完整回答或准确率结果。跨大小观察趋势时优先查看这一组，再结合自然生成结果解释实际使用成本。

每档每种方法/解码尝试预热到 2 个可执行请求，再正式重复 3 次；全档不可执行时保留失败记录。吞吐按 batch=1 串行有效请求时间计算，包括失败尝试；它描述单请求服务成本。加速比只在相同题目、重复编号和解码模式下，两边都完成的请求上计算，并报告配对覆盖数和双方输出长度。

## 服务器运行

先同步新脚本到现有仓库。沿用当前服务器环境；正式测量期间选中的 GPU 应独占。若训练仍占用全部卡，先让训练正常保存并暂停，测完继续原训练。脚本不会停止任何训练进程。

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
cd /workspace/wyx/f2llm/tensor-compression
nvidia-smi

# 这里选物理 GPU 0；如使用其他空闲卡，修改编号。
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

可先执行审计，只检查既有数据、原始场重放与 checkpoint 类型/数据身份，不加载 Qwen、不运行 GPU 计时。完整模型/分词器/训练配置匹配会在正式运行加载模型后严格检查：

```bash
python scripts/benchmark_readout_efficiency.py \
  --run-dir "$FIELD_TO_LLM_RUNS_DIR/ins_tracer_full_1x1" \
  --checkpoint-name best.pt --training-status completed \
  --output-dir "$FIELD_TO_LLM_RUNS_DIR/efficiency_audit_$(date +%Y%m%d_%H%M%S)" \
  --audit-only
```

三组顺序运行，每次只占用一张卡。每组内部自动运行序列化与接口、自然生成与固定长度两套测试：

```bash
BENCH_ROOT="$FIELD_TO_LLM_RUNS_DIR/efficiency_$(date +%Y%m%d_%H%M%S)"

for block in 1 2; do
  python -u scripts/benchmark_readout_efficiency.py \
    --run-dir "$FIELD_TO_LLM_RUNS_DIR/ins_tracer_full_${block}x${block}" \
    --checkpoint-name best.pt --training-status completed \
    --output-dir "$BENCH_ROOT/ins_${block}x${block}" \
    --split val --area-edges 64 256 1024 4096 16384 \
    --per-bin 24 --repeats 3 --warmup 2 --fixed-tokens 32 || break
done

python -u scripts/benchmark_readout_efficiency.py \
  --run-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_full_1x1" \
  --checkpoint-name last.pt --training-status in_progress \
  --output-dir "$BENCH_ROOT/multisource_v2_1x1_intermediate" \
  --split val --area-edges 64 256 1024 4096 16384 \
  --per-bin 24 --repeats 3 --warmup 2 --fixed-tokens 32
```

每组最多 120 道独立问题，正式执行 120×3×2×2=1440 次请求，另加预热。实际可执行数和耗时由上下文、网格大小和生成长度决定。首次核对运行链路可使用 `--per-bin 2 --repeats 1`，另建输出目录并标为试跑；论文表格使用正式配置结果。

## 返回结果

- `contract.json`：代码版本、checkpoint 步数/哈希、来源训练状态、原配置、模型/分词器身份、GPU/软件环境和计时范围。
- `selection.json`：固定采样清单、每档题数与任务/来源构成；空档明确保留。
- `requests.jsonl`：每次请求的延迟分解、显存、实际输出长度/文本、完成状态、重复编号。
- `warmup.jsonl`：单独保存的预热记录。
- `summary.csv` / `summary.json`：各大小档的 p50/p95、TTFT、显存、串行吞吐及配对加速比（配对表在 JSON）。
- `size_trends.png`：自然/固定长度生成的延迟、TTFT 和显存曲线；环境缺少 matplotlib 时保留 CSV/JSON，并提示未绘图。

优先发回三组 `contract.json`、`selection.json`、`summary.json`、`summary.csv` 和曲线图。逐题记录保留用于核对输出长度与失败原因。尚未在远程服务器上执行本测试，实际 GPU 时间/显存以服务器生成的文件为准。

