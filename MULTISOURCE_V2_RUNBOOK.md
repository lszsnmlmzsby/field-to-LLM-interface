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
本次只调整文本长度上限；字段、问题、划分、batch和训练轮数均不变。
若已有checkpoint，它的原训练上限属于严格续训身份，应使用其保存配置续训，
不要直接换成本次配置。本次报错发生在训练开始前，没有这个续训冲突。

本次修复的本地验证：预检/旧训练/性能回归106通过、1跳过；错误提示最后调整后
定向检查11通过；新版训练/续训及长问句动态padding检查3通过。跳过项仍是实际
Qwen分词器资产缺失；尚未测出服务器这份完整QA的最大token长度。

## 3. 真实 Qwen 检查与正式训练

```bash
(
  set -e
  for block in 1 2; do
    python -u scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile full \
      --block-shape "$block" "$block" --geometry-check-only \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_geometry_${block}x${block}"

    python -u scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile smoke \
      --block-shape "$block" "$block" --console-every-updates 10 \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_smoke_${block}x${block}"
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
    python -u scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile full \
      --block-shape "$block" "$block" --console-every-updates 200 \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_v2_full_${block}x${block}"
  done
)
```

两次训练各从头开始、顺序执行，同样的裁剪/查询/两轮。batch上限16，累积1；
同形状尾批仍可能不足16。每2048次更新保存，每8192次更新全量验证；
best仍根据已见尺寸上的宏平均任务答案准确率选择，输入关闭诊断不参与。
训练时间需根据实际 training_plan、稳定 step 时间和一次完整验证时间估算，
不能保证24小时完成。验证场编码与分词缓存继续启用。

## 4. 中断后续训

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_full_1x1"
python -u scripts/train_point_readout.py \
  --config "$RUN/resolved_config.json" --profile full \
  --resume "$RUN/last.pt" --output-dir "$RUN" \
  --console-every-updates 200
```

正在训练2×2时改为它的目录。不要添加覆盖参数，不要拿旧数据实验的 checkpoint
续训新版。正常终止会请求在完整更新处保存；强制断电/SIGKILL只能恢复最近 checkpoint。

## 5. 正式测试、语言测试和输入依赖诊断

训练结束后对每个结构执行以下四种独立评测。后三种是补充诊断，不能替代正式test。

```bash
(
  set -e
  for block in 1 2; do
    RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_full_${block}x${block}"
    python -u scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test \
      --output-dir "$RUN/test_best"

    python -u scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test --eval-suite language \
      --output-dir "$RUN/test_language"

    python -u scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test --eval-suite language \
      --input-mode disabled --output-dir "$RUN/test_language_no_field"

    python -u scripts/train_point_readout.py \
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
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_v2_full_2x2"
python scripts/score_point_readout.py \
  --qa-dir "$FIELD_TO_LLM_ROOT/data/multisource_openbc_v2/full" --split test \
  --predictions "$RUN/test_best/test_predictions.jsonl" \
  --output "$RUN/test_best/test_rescored.json"
```

## 验证记录

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
