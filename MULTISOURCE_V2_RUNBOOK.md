# 多来源能力扩量、语言与任务实验 v2

本方案落实 2026-10-21 关于扩量、自然语言、任务能力及 VLM 经验的讨论。
配置为 `configs/field_to_llm_multisource_v2.yaml`；旧配置与已有结果保留用于对照。
本地测试只能检查程序逻辑，真实 Qwen14B 的显存、速度与效果须在服务器验证。

## 修改方案与实验边界

1. 增加独特裁剪引用，而非仅重复改写已有问题。INS 32,768、Acoustic 65,536、
   Euler openBC 262,144、Gray–Scott 65,536，共 425,984 个训练裁剪。
   每个 val/test 仍为 2,048 个裁剪，四来源各 512 个。
2. 完整源仍是既有 INS 与三个完整 The Well 来源；官方轨迹划分、INS 80/10/10、
   高宽 1～128、形状保留池、FP16 舍入后的逐裁剪 z-score、两轮训练均保留。
   不新增 IID、periodicBC、多变量联合输入或时间预测。裁剪数不等于独立模拟数。
3. 使用可执行查询定义生成问句和精确答案；多个语言表达共享同一语义定义。
   训练使用四种表达族，验证与测试另各保留一种表达族。仍使用明确的一基行列坐标、
   严格 JSON 数值/坐标答案；语言模型不会接收内部程序、真值或来源标签。
4. 在原 11 类任务上新增两点差、两个子区域均值之差、阈值覆盖比例。
   扩展多点、反向/间隔线读取及随机整数矩形。每个场最多 11 种题型，按能力权重
   无放回选择；小场无法执行的任务明确记录。训练 QA 上限 4,685,824，
   两轮最多 9,371,648 次问题呈现，以构建输出的实际数量为准。
5. 新增语言配对测试、同问题换真实场测试、分难度指标、数值零答案基线与关闭场通路的诊断。
   语言/关闭输入诊断均单独运行，不加入周期性验证或 best checkpoint 的选择。
6. 保持当前模型结构和优化目标。任务课程、空间辅助监督、LoRA/解冻是后续对照，
   本版不自动启用；学习率 warmup 不代表任务课程。现有重建只约束独立数值分支。

这次同时调整数量、语言和任务，因此新旧结果差异不能单独归因于数据量。
旧 QA/checkpoint 不应换名后用于新版训练；新版从头训练并使用独立目录。
新验证裁剪并不保证与旧版逐个相同。需要比较旧模型时，旧配置、旧 QA 与旧结果仍可复现；
本版 1×1/2×2 必须共享同一份新 QA。

## 查询与难度

完整输入场的高宽仍在合法整数域随机采样。读取点数/线长上限为 32，逐值区域面积
上限为 32，统计/定位区域面积上限为 4,096。面积上限控制本轮任务和输出成本，
不是固定尺寸列表。矩形可为任意合法整数高宽，包括狭长形状。

均值和标准差不采整幅输入场，以避免 z-score 的整场均值≈0/标准差≈1捷径；
常量子区域仍保留并单独报告。阈值比例按标准化数值判断，使用更细的答案精度与
容差（3 位小数、绝对容差 0.02）；其他数值沿用 1 位小数、容差 0.2，坐标必须
精确匹配合法位置。并列极值接受合法并列坐标。

能力权重 `readout:4/localization:3/aggregation:2/composition:1` 用于抽题，
不是保证最终题目严格按该比例分配；受无放回选择和小场可用任务影响，实际比例
记录在 metadata。默认最多11题而非把14类题的所有语言版本都重复加入训练。
prompt/answer token 上限为 2048/256。2026-10-22 的服务器日志确认存在 788-token
题目，原来的 768 上限不足，现已提高。2048 是允许上限，不是已测出的全集最大值；
实际最大长度以服务器真实分词器的完整预检报告为准。
批次只填充到该批次实际最长序列，不会把每道题都填充到2048。
完整预检统计各划分全部超限题及最大长度，保存报告后仍拒绝超限训练，不截断或删题。

## 1. 同步代码并完成源下载（无需 GPU）

以下命令由用户在远端 Bash 执行，本地代理未连接服务器。
将本次代码、配置、源文件清单同步后：

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
cd /workspace/wyx/f2llm/tensor-compression
export MULTI_CONFIG=configs/field_to_llm_multisource_v2.yaml

python -u scripts/download_the_well.py \
  --datasets acoustic_scattering_inclusions euler_multi_quadrants_openBC gray_scott_reaction_diffusion \
  --output-dir "$FIELD_TO_LLM_ROOT/raw/the_well" \
  --endpoint https://hf-mirror.com --workers 2
```

已验证来源复用收据，未完成文件续传。Euler 负轨迹数元数据兼容仍保留，原始文件
不改写。必须获得三个来源的 `complete_source`，且完整 INS 收据有效，才能构建。
不要为赶进度删掉某些文件、轨迹或 split。

## 2. 构建与审计（无需 GPU）

```bash
python -u scripts/build_point_readout_qa.py --config "$MULTI_CONFIG" --profile smoke
python -u scripts/build_point_readout_qa.py --config "$MULTI_CONFIG" --profile full

python -u scripts/train_point_readout.py \
  --config "$MULTI_CONFIG" --profile full --token-check-only --device cpu \
  --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_tokens"
```

新数据写入 `$FIELD_TO_LLM_ROOT/data/multisource_openbc_v2/{smoke,full}`。
构建拒绝覆盖非空目录；中断留下 `.build_in_progress`，不能用于训练。
更换构建输出位置时，训练必须通过 `--qa-dir` 指向同一新目录。
`qa_fields.hdf5` 是实验裁剪缓存，完整原始源继续留在 `raw/`。
token 检查已包含源裁剪/真值审计，使用服务器上的真实分词器扫描 train/val/test 并保存
长度报告，不加载模型权重；执行它之前无需再单独运行 `--audit-only` 重复审计。
它不代替 GPU smoke；训练启动时仍会检查实际加载的题目。

`metadata.json` 保存来源、通道、任务、语言、难度、覆盖统计及来源和 QA 的哈希。
额外的 `diagnostics_val.jsonl` / `diagnostics_test.jsonl` 使用对应 split 的场，
每个来源默认选16个场，每条语义查询配一条已见表达和一条保留表达；不进入训练。
smoke 对应每个来源2个诊断场，仅用于流水线检查。

### 已构建完成，但遇到 788 > 768 报错后的恢复

日志中训练4,685,677题、val/test各22,483题已完成构建和数据审计。错误发生在加载
Qwen权重之前；这些数据可以原样复用，无需重新下载、构建或修改metadata。
同步修正后的训练脚本和v2配置，然后只重跑分词预检，使用新的输出目录：

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
cd /workspace/wyx/f2llm/tensor-compression
export MULTI_CONFIG=configs/field_to_llm_multisource_v2.yaml
TOKEN_CHECK_DIR="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_tokens_$(date +%Y%m%d_%H%M%S)"

python -u scripts/train_point_readout.py \
  --config "$MULTI_CONFIG" --profile full --token-check-only --device cpu \
  --output-dir "$TOKEN_CHECK_DIR"
```

`token_preflight.json` 记录各划分的实际最大prompt/answer长度与超限统计。预算超限时
完成扫描并写报告后退出失败，不加载模型；模板或EOS格式错误仍立即失败。
预检通过后继续下面的几何检查、smoke、正式训练，使用同一份修正后的配置。
当时的修复只调整文本长度上限；字段、问题、划分和训练轮数不变。
随后按三张 A800 的分配和用户要求，将本配置的全局 batch 上限改为48，见下文。
若已有checkpoint，它的原训练上限属于严格续训身份，应使用其保存配置续训，
不要直接换成本次配置。本次报错发生在训练开始前，没有这个续训冲突。

本次修复的本地验证：预检/旧训练/性能回归106通过、1跳过；错误提示最后调整后
定向检查11通过；新版训练/续训及长问句动态padding检查3通过。跳过项仍是实际
Qwen分词器资产缺失；尚未测出服务器这份完整QA的最大token长度。

## 3. 真实 Qwen 检查与正式训练

`--batch-size` 始终表示所有 GPU 合计的 batch 上限，full/pilot 默认48、累积1、两轮。
2026-10-26 起，`--micro-batch-size` 默认8，表示每卡一次前后向的上限。
原分配按题数均分，例如48题分为16+16+16；新版默认在同一个全局批次内按完整文本长度
均衡各卡负担，因此各卡题数可能不同。全局问题集合不变，尾批仍全部保留，
不足3题允许某卡不处理样本。每次更新使用实际全局题数归一化。
同一全局批次可在每卡拆成若干小批次累积；发生前后向 CUDA OOM 时，清除本卡这次
更新的全部梯度、恢复随机/缓冲状态，然后减小小批次重算，其他卡等待后统一同步梯度。
重试上限按8→4→2→1降低（尾批可能更小）；降到1仍OOM会明确失败。
2026-10-25 起，新配置默认按场面积和完整文本长度分别记忆小批次上限，
不再让一次大输入的 OOM 永久限制后续所有小输入。2026-10-26 起，上限改为8，
每个小批次还受4096个填充后文本token的执行预算约束，具体语义见4.2。
该恢复只针对前后向显存不足，不能修复模型加载、优化器状态、通信阶段或其他异常。
成功降低的每卡上限会保存到 checkpoint 并在续训时恢复。

长文本会增加实际显存占用；2048是允许上限，仍按各小批次的实际最长文本padding。
自动拆分保留全部问题、完整文本和全局平均损失，但浮点运算顺序变化不保证位级一致。
全局48相比原16减少每轮更新次数；精确次数以 `training_plan.json` 为准，不保证恰好1/3。
学习率仍为5e-5/gate 5e-4，warmup比例和两轮不变，不自动将学习率乘3。
保存/验证仍按原2048/8192次更新触发；因此两次检查之间通常会覆盖更多问题。
验证、测试按场分给三卡生成；新配置每卡最多合批8个同场问题，独立维护文本上下文
和结束符。汇总完整预测检查QA ID无遗漏、无重复，再计算总指标。

源裁剪审计只由rank0完整执行一次；分词预检按问题分片，全体合起来仍覆盖所有要求的题。
模型文件哈希只在rank0读取一次，各rank模型/分词器/配置/数据身份还会相互核对。
目前每个进程仍加载完整QA元数据和独立的有界token缓存，因此CPU内存需求会增加。
三张卡分别保留一份模型，不会将显存合成一块240GB。

已完成的完整源下载和v2 QA可以直接复用，不要因batch调整重新构建。
同步新增的 `scripts/readout_distributed.py`、训练脚本和配置后，在远端 Bash 设置：

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
cd /workspace/wyx/f2llm/tensor-compression
export MULTI_CONFIG=configs/field_to_llm_multisource_v2.yaml
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
```

先执行三卡几何检查和smoke；smoke覆盖小场流水线，不能证明完整长文本一定能用每卡8。
三个进程各持有一张卡，`--device` 保持auto，不要为每个进程指定同一个cuda:0。

```bash
(
  set -e
  for block in 1 2; do
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
      scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile full \
      --block-shape "$block" "$block" --geometry-check-only \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_geometry_${block}x${block}"

    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
      scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile smoke \
      --batch-size 48 --micro-batch-size 8 \
      --block-shape "$block" "$block" --console-every-updates 10 \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_smoke_${block}x${block}"
  done
)
```

geometry 只审计、分词并测试边界形状的代表题，保存预检范围；保留完整数据清单校验，
不重放几百万条完整 QA。它不等于全量数据审计，也不覆盖优化器、最长文本组合与
完整生成阶段的显存。上一步 token-check 和正式训练仍使用完整数据。
smoke 仅20次更新，不能称为正式训练。通过后，在持久终端中执行：

```bash
(
  set -e
  for block in 1 2; do
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
      scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile full \
      --batch-size 48 --micro-batch-size 8 \
      --block-shape "$block" "$block" --console-every-updates 200 \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_full_${block}x${block}"
  done
)
```

两次训练各从头开始、顺序执行，每次用三张卡，同样的裁剪/查询/两轮。全局batch上限48，累积1；
同形状尾批仍可能不足48。每2048次更新保存，每8192次更新全量验证；
best仍根据已见尺寸上的宏平均任务答案准确率选择，输入关闭诊断不参与。
训练时间需根据实际 training_plan、稳定 step 时间和一次完整验证时间估算，
不能保证24小时完成。验证场编码与分词缓存继续启用。
启动日志应含 `global_batch=48 world_size=3 per_device_micro_batch_limit=8`。
`questions_in_update`是三卡实际总题数；`per_rank_execution`记录每卡处理题数、小批次拆分
和OOM重试次数，`per_rank_memory`记录各卡显存峰值，顶层峰值为各卡的最大值而非相加。
若希望先观察正式数据上的显存和速度，可在full命令加 `--stop-after-updates 20`：
它保留完整两轮调度，只提前保存停止；之后按下一节继续，不应把20步称为完整训练。

## 4. 中断后续训

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_full_1x1"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
  scripts/train_point_readout.py \
  --config "$RUN/resolved_config.json" --profile full \
  --resume "$RUN/last.pt" --output-dir "$RUN" \
  --console-every-updates 200
```

正在训练2×2时改为它的目录。不要添加覆盖参数，不要拿旧数据实验的 checkpoint
续训新版。正常终止会请求在完整更新处保存；强制断电/SIGKILL只能恢复最近 checkpoint。
严格续训要求进程数相同，三卡checkpoint请继续用三卡；旧全局16的checkpoint不能直接
改变为48续训，本次48实验使用新目录从头开始。评估允许使用不同进程数。
可以在续训命令追加 `--micro-batch-size 8` 降低每卡计算块，仍保持全局48；
它属于执行设置，不更改训练身份。checkpoint包含每个rank的随机状态、小批次上限，
以及统一的优化器、调度器、epoch/cursor和待完成验证标记，由rank0原子写入。
torchrun启动器被终止时可能强制结束worker，不能保证信号处理有足够时间保存；
计划内短跑优先用 `--stop-after-updates`，强制收回资源后使用最后一次完整的 `last.pt`。

### 4.1 已有三卡训练启用执行加速（2026-10-25）

本节保留上一版改动的说明；下方续训命令已同步为2026-10-26的执行设置。
之前启用自适应上限16后的日志未证明训练加速，最新对比方法见4.2。

同步本次 `scripts/train_point_readout.py`、`scripts/train_tensor_qwen_cross_attention.py`
和配置后，已有完整源数据、QA和checkpoint可继续使用。
旧 `resolved_config.json` 没有新增开关，因此第一次续训显式启用：

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
cd /workspace/wyx/f2llm/tensor-compression
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_full_1x1"

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
  scripts/train_point_readout.py \
  --config "$RUN/resolved_config.json" --profile full \
  --resume "$RUN/last.pt" --output-dir "$RUN" \
  --micro-batch-size 8 --adaptive-micro-batch \
  --length-balanced-batches --micro-batch-token-budget 4096 \
  --eval-batch-size 8 --eval-parity-check-questions 32 \
  --console-every-updates 200
```

不加覆盖参数，不重建数据，不修改全局 `--batch-size`。三卡数量仍须一致。
原数据每轮4,685,677题、两轮9,371,354次呈现、208,570次更新保持不变。
全局每批上限48，同形状尾批仍可不足48；此次变化的是每卡物理计算块，
允许便宜输入恢复到8，不同卡也可使用不同计算块。梯度仍按全局实际题数归一化。
学习率、优化器状态、下一批位置、验证/保存间隔继续从原checkpoint恢复。

按成本分桶仅是显存适配启发式，不能保证每个新桶第一次就适配。
初次遇到较贵输入可能打印 `oom_retry`，会清除未完成更新的所有本地梯度，
恢复随机/缓冲状态后重算完整本地更新，不跳题、不额外执行优化器更新。
学习到的分桶上限写入各rank的checkpoint；旧checkpoint的统一上限4不会锁死所有新桶。
改变计算块可能改变浮点累加次序，不能保证逐位相同的参数轨迹。

验证、测试保留全部问题、原生成长度上限、贪心策略和评分标准；同一场只编码一次，
共享只读场记忆，问题之间不共享文本历史。解码OOM时减小合批重试同一组待答问题。
每次验证、每张卡对最先进入合批的最多32题再做逐条生成对照；若有差异，
当前组改用逐条结果，该卡本次剩余验证回退到逐条。输出 `batch_parity_checked`、
`batch_parity_mismatches`，并记录各卡最终 `evaluation_batch_limit`。
这是抽查，不保证未抽查问题在BF16下逐位相同，也不是科学任务准确率证明。
若要求全量逐条生成基准，可将 `--eval-batch-size` 设为1。

看性能时使用新增日志：

- `training_window.questions_per_second`、`seconds_per_update`：最近日志窗口内纯训练，
  含本次更新的重试、同步等待与优化器开销，排除验证、保存和日志写盘。
- `per_rank_execution.micro_batch_sizes`：各卡实际拆分；`cost_bucket`为面积的二进制
  数量级和文本长度的128-token分桶，不是裁剪形状限制。
- `local_compute_seconds`含数据读取/分词/前后向/重试；
  `synchronization_and_wait_seconds`含同步及等待其他卡，不能直接当作网络传输时间。
- `validation ... elapsed_seconds`：完整验证实测用时，含生成对照、汇总与评分。

旧日志稳定训练约2.1–2.2秒/更新、完整验证约2.1小时，只可作为粗略对照。
需要相似输入成本和足够多更新的远端测量才能判断收益；未实测前不承诺加速倍数。
若主动做短跑，在上面的完整续训命令追加 `--stop-after-updates 200`，它表示额外200步，
不会改变两轮总预算；之后移除该参数再次续训。启动的全量审计和分词预检仍会执行。

全局batch按成本变化并非天然有害，但会改变梯度噪声、每轮更新次数以及基于step的
学习率调度；若简单对不同大小的batch等权，还会改变样本对每次更新的相对权重。
因此这次对正在进行的实验先采用物理计算块自适应，保留既定优化过程。

### 4.2 按长度均衡、减少padding与固定checkpoint测速（2026-10-26）

上一版日志的稳定片段约2.29秒/更新，旧片段约2.14秒/更新，不能据此声称训练加速。
其中一次更新三卡各16题，本地计算却分别约1.83、1.50、3.09秒，较快的卡需要等待。
这些不是同批输入的严格对照，但提示应优先减少负载不均衡、padding和OOM重算。

本版不拆Qwen权重，不修改完整源、QA、全局更新的问题集合、训练两轮、损失权重或学习率：

- 完整分词预检顺便保留每题实际的prompt+answer长度，三卡一次汇总，随后各卡仅查表。
  当前4,685,677题约占每卡17.9 MiB的CPU内存，不额外重做一次全集分词。
- 每个既有全局同形状batch内部，先将相近长度问题分组，再按估计的填充token成本分卡；
  各卡内部也按长度执行，减少短问题被长问题填充。这个成本只是启发式，须以实测为准。
- 每卡最多8题，并逐个小批次应用4096个填充后文本token的预算。例如最长800token时，
  上限会降到4；最长400token时可尝试8。场面积仍由原自适应OOM分桶保护。
  4096不是问题长度上限，不截断任何文本；即使单题超过执行预算，也完整地用1题计算。
  原2048/256的prompt/answer完整预检限制仍保留。这不是显存不会OOM的保证。
- 按每个小批次选择上限，避免一题长文本限制同次更新中的所有短题；保留完整重算回退。
  新日志记录真实/填充token数、`padding_fraction`及窗口内各卡`rank_oom_retries`之和。

每题仍按全局实际题数加权。重新分卡、排序、合批会改变浮点累加顺序，不能承诺参数逐位
一致；不会主动改变样本权重或任务。换了执行策略的收益与数值差异可先通过以下入口检查。

先同步训练脚本、`readout_distributed.py`和配置；在原训练已停止并完成保存后执行：

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
cd /workspace/wyx/f2llm/tensor-compression
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_full_1x1"
BENCH="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_execution_benchmark_$(date +%Y%m%d_%H%M%S)"

python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
  scripts/train_point_readout.py \
  --config "$RUN/resolved_config.json" --profile full \
  --resume "$RUN/last.pt" --benchmark-updates 30 --benchmark-warmup-updates 3 \
  --output-dir "$BENCH"
```

三种策略固定为`original_micro4`、`balanced_micro4`、`balanced_budgeted_micro8`，分别表示
原均分/每卡4、长度均衡/每卡4、长度均衡/每卡最多8且token预算4096。最后一种继承checkpoint
已有的分桶显存限制，三者均允许OOM回退；报告包含最终上限和实际拆分成本。
使用相同checkpoint的后续30次全局更新问题；每种先预热3次，权重保持固定，不执行optimizer
或scheduler更新，不写任何checkpoint，也不覆盖原运行目录。输出目录必须是另一个空目录。
这是明确的短程执行诊断，不是缩减后的正式训练或准确率评测。

终端的`benchmark_policy=`和`$BENCH/execution_benchmark.json`保存各方案的题/秒、秒/更新、
padding比例、OOM次数、各卡显存峰值、逐次loss差及首个全局更新的未裁剪梯度相对L2差。
计时包含前后向、同步和梯度裁剪，排除启动审计/预检、optimizer.step、验证和CPU梯度比较。
因此不能直接与包含优化器的训练窗口时间等同，也不能用它预测整轮验证耗时。
前置全量审计/分词预检仍会执行；30步可能没有覆盖所有场尺寸或最长文本，不自动挑选策略。

新版默认采用长度均衡、每卡最多8和4096-token预算；按4.1的命令即可恢复原训练。
如果实测`balanced_micro4`更快，将4.1中的三个执行选项改为：

```bash
  --micro-batch-size 4 --no-adaptive-micro-batch \
  --length-balanced-batches --micro-batch-token-budget 0 \
```

若只有原分配更快，再把`--length-balanced-batches`改为`--no-length-balanced-batches`。
关闭自适应时，checkpoint中已有的更低统一OOM上限仍会保留，以启动日志和实际拆分为准。
这些开关不修改全局batch或续训身份，无需重建QA。测速后恢复训练仍从原`last.pt`开始。

## 5. 正式测试、语言测试和输入依赖诊断

训练结束后对每个结构执行以下四种独立评测。后三种是补充诊断，不能替代正式test。

```bash
(
  set -e
  for block in 1 2; do
    RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_full_${block}x${block}"
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
      scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test \
      --output-dir "$RUN/test_best"

    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
      scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test --eval-suite language \
      --output-dir "$RUN/test_language"

    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
      scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test --eval-suite language \
      --input-mode disabled --output-dir "$RUN/test_language_no_field"

    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=3 \
      scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test --eval-suite field-pairs \
      --diagnostic-pairs 128 --diagnostic-max-candidates 4096 \
      --output-dir "$RUN/test_field_pairs"
  done
)
```

语言测试比较相同语义查询的两种表达；同时报告正确率和配对同时正确率，避免把
两种问法都答错当作成功。关闭输入结果只与同一language套题的正常输入结果比较，
不与完整test直接比较。关闭的是场通路，不是给模型喂一张会被编码的全零场。
输入关闭后某类题仍正确不自动证明作弊：常量、近零、输出先验都需要分组检查。

换场诊断只从同一 test split 中选择相同形状的真实场，保持问题逐字相同，重新执行
查询计算答案。两场的可接受答案集合必须不重叠，才保留该对；报告两者同时答对率。
最多128对、检查4096个候选，不追加或更改训练数据。它有意选择答案不同的样本，
不能当作整个测试分布的准确率；找不到有效配对时报告 `unavailable`，不算通过。
保存 `field_pair_queries.jsonl` 和 `field_pair_selection.json` 便于复核。
本版尚未单独实现仅改 min/max 或操作顺序的最小语义对照套件。

离线复评分不需要 GPU：

```bash
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_3gpu_b48_full_2x2"
python scripts/score_point_readout.py \
  --qa-dir "$FIELD_TO_LLM_ROOT/data/multisource_openbc_v2/full" --split test \
  --predictions "$RUN/test_best/test_predictions.jsonl" \
  --output "$RUN/test_best/test_rescored.json"
```

## 验证记录

2026-10-26 长度均衡与固定checkpoint测速回归：**178 passed、2 skipped，97.18秒**。
覆盖真实小型Qwen的长短文本padding减少、逐题/全局加权梯度一致性、尾批与空rank、
低token预算不截断问题、逐小批次自适应、三进程长度汇总和协调报错、旧checkpoint恢复、
连续/中断续训参数及随机状态一致，以及三方案测速不推进优化器、不改写原checkpoint。
两项跳过仍为缺少真实Qwen分词器资产和Windows原生Gloo不可用；三进程测试使用文件通信
替身，OOM由测试注入。没有远端Qwen14B/NCCL测速或准确率结果，不据本地用例宣称加速。

2026-10-25 执行加速主回归：**163 passed、2 skipped，106.27秒**。
随后加入共享场K/V零拷贝批维展开，相关专项回归：**42 passed、1 skipped，31.85秒**；
两组存在重叠，不应相加作为独立用例数。覆盖不同长度提示的合批/逐条生成一致性，
缓存开关、1×1/2×2结构、独立EOS与长度上限、完整QA顺序与评分、关闭场诊断、
模拟解码OOM/舍入差异后的不丢题回退、按成本记录OOM上限、全局加权梯度、
三个CPU进程由旧执行配置续训到新配置、参数与随机状态一致性。
CUDA OOM仍由测试注入；三进程测试使用文件通信替身，原生Gloo在本机不可用。
另一个跳过项是缺少实际Qwen分词器资产。

主回归首次发现一处旧测试替身未返回新增的计时/合批诊断字段，已补齐后重跑通过。
补充回归首次因Windows临时路径的8.3短名与长名字符串比较失败；指定项目内明确的
临时目录后通过，未改变生产路径规则。CLI帮助与 `git diff --check` 通过。

在本机RTX 5060 Laptop GPU / torch 2.9.1+cu128上，默认SDPA后端的小型BF16
共享场注意力合批与逐条结果比较通过（本次最大绝对差0），K/V展开与原张量共享存储。
强制Flash Attention检查失败，原因是本机Windows PyTorch未编译该内核；生产代码
仍由PyTorch自行选择SDPA后端，不强制它。该局部检查不证明Linux torch 2.4.1的
融合内核、Qwen14B生成逐题一致性、三张A800显存或整体加速倍数。
未在远端执行本次改动，也未重新执行完整源下载、QA构建或正式训练。

2026-10-22 三卡改造的本地回归：**164 passed、2 skipped，125.44秒**。
随后验证通信边界修正的相关检查 **38 passed、1 skipped**；最后补充单rank显存不足
和协调失败传播后的分布式专项 **11 passed、1 skipped，26.80秒**。
覆盖真实小型Qwen的全局加权梯度、尾批/空rank、三个进程的模型一致性、完整预测合并、
分词超限汇总、模拟部分反传后OOM重算、单rank重试、单写入者失败传播，及训练中断
续训后的逐参数和各rank随机状态一致性。GPU OOM由测试注入，没有实测A800 OOM。

两类跳过是实际Qwen分词器资产缺失，以及本机Windows PyTorch的Gloo transport不可用。
实际尝试原生Gloo初始化报 `makeDeviceForHostname(): unsupported gloo device`；
三个CPU进程的集成测试使用文件交换/同步屏障作为**测试通信替身**，真实执行小型Qwen
前后向和训练/评估/保存/续训逻辑。这不证明NCCL或GPU驱动/互联可用。
生产三卡CUDA入口使用NCCL；必须在服务器完成三卡smoke，再评估速度、显存和效果。
本地测试使用 `MKL_THREADING_LAYER=SEQUENTIAL`、`OMP_NUM_THREADS=1`、`PYTHONUTF8=1`，
避免Windows重复OpenMP库及中文路径读取问题。CLI帮助与 `git diff --check` 通过。

2026-10-21 本地 Windows 回归：**253 passed，1 skipped，136.33 秒**。
覆盖新查询、构建/划分/配额/完整源清单、语言与换场诊断、旧 INS/多来源协议、
块几何、缓存和训练性能语义、聊天入口，以及小型 Qwen 连续训练与中断续训逐参数相等。
端到端集成测试中的生成文本使用固定替身，以验证评测接口和离线复评分一致性；
另外的既有测试覆盖小型模型真实生成，但均不代表科学任务准确率。

随后针对几何预检范围与换场 split 防护的最后修正，完成 **46 项相关回归 + 2 项
新版训练/恢复集成 + 1 项真实小模型几何前后向检查，全部通过**。

唯一跳过项是本地没有真实 Qwen 分词器资产。构建、训练、评分三个 CLI 的 `--help`
检查和 `git diff --check` 均通过。早期 pytest 遇到旧临时目录权限错误，改用独立
`--basetemp` 后解决；CLI 初查遇到本机重复 OpenMP 运行库，使用
`MKL_THREADING_LAYER=SEQUENTIAL` 与 `OMP_NUM_THREADS=1` 后通过，未使用不安全的
重复运行库忽略开关。新集成测试夹具的参数/1×1 GroupNorm 配置问题已修正并纳入回归。

没有执行远端下载、完整 QA 构建或 Qwen14B 训练，未测服务器 GPU 显存、速度及准确率。
