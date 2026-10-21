# 多来源二维场读数实验

2026-10-21 的扩量、语言与任务能力实验使用独立 v2 配置，最新步骤见
[MULTISOURCE_V2_RUNBOOK.md](MULTISOURCE_V2_RUNBOOK.md)。下文保留 v1 配额和命令，
用于复现已有实验；不要将两版 QA 或 checkpoint 混用。

本轮配置是 `configs/field_to_llm_multisource.yaml`。先把本次代码、配置及
`configs/source_manifests/` 同步到服务器，再执行下面的命令。本文命令面向服务器的
Bash；本地测试通过不代表已经在 A800 / Qwen 14B 上完成验证。

## 已固定的实验范围

用户确认：高、宽仍在 1～128 的整数范围采样；训练裁剪场状态由 16,384 增加为
131,072，保持两轮；较大尺寸另做实验。本阶段不加入 IID 合成场。
为减少下载体积，本轮只保留 Euler openBC，移除周期边界来源的下载和训练要求。
新增完整源约 3.097 TB，加上已有 INS 约 3.118 TB；periodicBC 仍可作为后续独立实验。

| 来源 | full 训练状态数 | 每个 val / test 状态数 | 使用的标量通道 |
|---|---:|---:|---|
| INS-Tracer | 32,768 | 512 | u、v、b |
| acoustic_scattering_inclusions | 32,768 | 512 | pressure、density、speed_of_sound、velocity 的两个分量 |
| euler_multi_quadrants_openBC | 32,768 | 512 | density、pressure、energy、momentum 的两个分量 |
| gray_scott_reaction_diffusion | 32,768 | 512 | A、B |
| 合计 | 131,072 | 2,048 | 四类物理场各占 25% |

以前 openBC 的 16,384 是预算分配：Euler 家族的 32,768 个状态由 openBC 和 periodicBC
平分，与文件大小或可用裁剪数量无关。现在 Euler 家族预算全部分给 openBC，因此与
INS 相同，保持本轮总训练状态数和两轮预算不变。

每个来源内均衡分配标量通道；每个通道按轨迹数量加权抽取源文件，避免文件切片大小
改变采样概率。向量分量分别作为二维标量场输入，未把多个物理通道拼成一次输入。
沿用裁剪场内 z-score、现有 FP16 存储精度和 11 类数值生成任务，因此仍研究
**归一化标量场的读数能力**，不训练时间推进或 PDE 求解。

131,072 指唯一的来源/轨迹/时间/通道/裁剪引用，**不是 131,072 次彼此独立的物理模拟**。
来自同一轨迹的帧和裁剪仍有相关性。训练 QA 上限 1,441,792 条；小尺寸容纳不了的
多点/线等任务会记录为不适用，实际条数以 `metadata.json` 为准。两轮会重复这些固定 QA。
保持同形状组批，batch 上限 16、累积 1；每种形状的尾批可小于 16，实际更新次数写入
`training_plan.json`，不能简单用 QA 数除以 16。

随机尺寸不是预设若干矩形。所有满足边界的整数尺寸都进入候选池，按固定种子划分
训练与形状保留池，并保留最大面积、单行/单列等边界检查。验证/测试包含见过与保留的
尺寸；有限次抽样不保证每个尺寸都出现。The Well 保留官方 train/valid/test；INS 保持
原来以轨迹划分的 80/10/10。完整 The Well 轨迹内容若跨划分重复会报错；只共享初态
会被记录，不等同于完整轨迹泄漏。INS 原有收据没有完整轨迹内容哈希，此项检查范围也
记录在元数据中。

一幅 512×512 标量场确实能无重叠切成 16 个 128×128 块；裁剪更多随机位置和尺寸则有
更多候选。本轮仍随机采样 1～128 的高宽及合法位置，不改成固定 16 块。网格点数是
128×128 的 16 倍，不等于独立轨迹数或信息量也必然是 16 倍。先确定轨迹所属 split，
再在该 split 中裁剪；同一轨迹的不同块不能分别进入训练和验证/测试。

新增物理源会出现空间常值裁剪。本配置明确保留它们，归一化后为全零，不再重新抽样
丢弃；并列极值接受任一合法并列坐标。旧单来源配置的常值拒绝规则保持原样。静态场
没有时间索引，不用重复时间帧增加数量。结果新增来源、物理家族、静态/动态、常值/
非常值分组，防止容易的常值题抬高总体结果却掩盖其他表现。常值归一化也意味着不同
绝对常数值无法由这个输入接口区分。

## 1. 完整下载和验证（无需 GPU）

```bash
source /workspace/wyx/f2llm/enter_f2llm.sh
cd /workspace/wyx/f2llm/tensor-compression
export MULTI_CONFIG=configs/field_to_llm_multisource.yaml

python -u scripts/download_the_well.py \
  --datasets acoustic_scattering_inclusions euler_multi_quadrants_openBC gray_scott_reaction_diffusion \
  --output-dir "$FIELD_TO_LLM_ROOT/raw/the_well" \
  --endpoint https://hf-mirror.com \
  --metadata-endpoint https://hf-mirror.com \
  --manifest-only

python -u scripts/download_the_well.py \
  --datasets acoustic_scattering_inclusions euler_multi_quadrants_openBC gray_scott_reaction_diffusion \
  --output-dir "$FIELD_TO_LLM_ROOT/raw/the_well" \
  --endpoint https://hf-mirror.com \
  --metadata-endpoint https://hf-mirror.com \
  --workers 2
```

默认下载本轮三个完整的当前版本来源，均通过 HF 镜像传输；命令也显式列出这三个来源。
固定版本清单随代码保存；已有目标目录的清单优先，禁止续传时悄悄换版本。
命令会显示每个来源的版本、文件数、精确字节数。明确排除发布者单独标记的
`old_and_problematic_data`，不截取当前正式源的帧、轨迹或 split。

此前已经启动的四源下载进程不会自动采用新配置；先停止旧进程，再用上面的命令续传。
已经下载的 periodicBC 文件不会被删除，但不会参与本轮构建。三个保留来源仍完整下载，
不通过删减帧或轨迹降低它们的体积。
中断后重跑同一下载命令会续传 `.part`，已验证且大小/时间戳未变的文件复用验证记录。
初次验证包括完整文件哈希、HDF5 布局、全部字段有限值及完整轨迹哈希，数 TB 的扫描
本身也会耗时；不能以下载进度条到 100% 代替验证完成。

三个 HF 源按发布者 LFS SHA256 校验，每个完整来源最终生成 `source_verified.json`。

若 Euler openBC 报 `n_trajectories=-1550`，而数组轨迹轴一致为 10，这是已观察到的
发布文件元数据错误。将支持 `publisher_sha256_verified_euler_negative_count_v1` 的
本次代码同步到服务器后，重跑上面的下载命令即可。已完整下载的 `.part` 会重新核对
发布者 SHA256、实际布局并全量扫描，通过后自动改为正式文件名；不重新传输完整文件，
也不修改 HDF5 属性。已验证文件继续复用收据。日志及每文件/最终验证记录的
`metadata_corrections` 保存声明与有效轨迹数。若字段间轨迹数不一致或其他校验失败，
仍保留文件并报错；不要手工改属性或跳过该分片。此兼容不改变完整源范围或 QA 预算。

需要重新逐字节检查这三份已下载来源时：

```bash
python -u scripts/download_the_well.py \
  --datasets acoustic_scattering_inclusions euler_multi_quadrants_openBC gray_scott_reaction_diffusion \
  --output-dir "$FIELD_TO_LLM_ROOT/raw/the_well" --verify-only --rehash
```

已有完整 INS 文件和验证收据继续复用，不重新下载。源数据详情见
[THE_WELL_DATASETS.md](THE_WELL_DATASETS.md)。

下载时间可能超过网页终端连接时间。若服务器已有 `tmux`，可先用
`command -v tmux` 确认，再在 `tmux new -s multisource` 内执行长任务；用
`tmux attach -t multisource` 返回。没有 tmux 时，下载也可以用下面的后台版本代替
前台下载命令（不要同时对同一目录运行两个下载器）：

```bash
mkdir -p /workspace/wyx/f2llm/tensor_llm_outputs/logs
LOG=/workspace/wyx/f2llm/tensor_llm_outputs/logs/well_download_$(date +%Y%m%d_%H%M%S).log
nohup python -u scripts/download_the_well.py \
  --datasets acoustic_scattering_inclusions euler_multi_quadrants_openBC gray_scott_reaction_diffusion \
  --output-dir "$FIELD_TO_LLM_ROOT/raw/the_well" \
  --endpoint https://hf-mirror.com --workers 2 \
  >"$LOG" 2>&1 </dev/null &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

必须看到上述三个 The Well 来源的 `complete_source` 成功输出，且已有 INS 完整源凭据仍有效，
才进入构建。`tail` 的 Ctrl+C 只停止
查看日志，不停止后台下载。持久终端/后台进程只处理终端断线，不延长 GPU 配额或
容器寿命；训练遇到资源到期仍依赖下面的 checkpoint 续训。

## 2. 构建固定 QA（无需 GPU）

```bash
python -u scripts/build_point_readout_qa.py --config "$MULTI_CONFIG" --profile smoke
python -u scripts/build_point_readout_qa.py --config "$MULTI_CONFIG" --profile full
```

只有全部完整源凭据通过检查才会开始构建。输出位于
`$FIELD_TO_LLM_ROOT/data/multisource_openbc_v1/{smoke,full}`。
新名称将本轮四来源 QA 与此前五来源方案隔离；不要把已有五来源 QA 或 checkpoint
换名后用于此配置。已有下载的三个保留来源及 INS 可以复用。
完整源仍留在 `raw/`；`qa_fields.hdf5` 只是本实验采样裁剪的精确缓存，不是完整源。
缓存保存已有 FP16 舍入后的数值，不额外量化、缩放、平滑或降采样。

`metadata.json` 记录实际 QA/状态数、来源和通道比例、常值比例、尺寸、划分、拒绝原因
及文件哈希。构建拒绝覆盖非空目录；失败构建保留 `.build_in_progress`，不能当作完成。
使用另一个输出目录重建时须让训练的 `--qa-dir` 指向同一新目录。

smoke 仅 160 个训练状态、320 个 val/test 状态，尺寸 1～8，3 类任务、20 次更新。
它验证新入口、反向传播及验证流程，不能作为 full 训练或大尺寸显存验证。full 才是
上述 131,072/2,048 状态与 11 类任务的正式实验。

## 3. 审计和服务器检查

独立审计（训练启动也会自动回放所用 split 的源裁剪与全部 QA）：

```bash
python -u scripts/train_point_readout.py \
  --config "$MULTI_CONFIG" --profile full --audit-only \
  --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_openbc_audit"
```

GPU 时间开始后，用真实 Qwen 对两种结构分别检查：

```bash
(
  set -e
  for block in 1 2; do
    python -u scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile full \
      --block-shape "$block" "$block" --geometry-check-only \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_openbc_geometry_${block}x${block}"

    python -u scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile smoke \
      --block-shape "$block" "$block" --console-every-updates 10 \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_openbc_smoke_${block}x${block}"
  done
)
```

full geometry 对包括 128×128、1×128、128×1 在内的边界执行 batch 16 前后向。
它不做优化器更新、不验证完整生成成本，因此仍需 smoke 的真实训练/生成检查。
这些步骤通过后再开始正式训练；若显存不够，请贴回失败形状及显存输出再调整，脚本
不会自动减少正式样本或降低输入尺寸。

## 4. 两次独立正式训练

```bash
(
  set -e
  for block in 1 2; do
    python -u scripts/train_point_readout.py \
      --config "$MULTI_CONFIG" --profile full \
      --block-shape "$block" "$block" --console-every-updates 200 \
      --output-dir "$FIELD_TO_LLM_RUNS_DIR/multisource_openbc_full_${block}x${block}"
  done
)
```

两个模型使用同一固定 QA 和两轮预算，各自从头训练。`1x1` 表示每个格点一个场 token；
`2x2` 表示每个块覆盖最多四个格点，是独立的结构对比。循环按顺序执行，不是两张 GPU
并行，也不是先训练一个结构再接着微调另一个。使用新输出目录，不覆盖旧 INS 结果。

默认每 2,048 次更新保存、每 8,192 次更新完整验证，结束也会验证；每次验证覆盖全部
固定验证 QA，保持逐题生成、EOS、最大生成长度、容差和最佳模型选择规则。原始
`seen_shapes.macro_task_answer_accuracy` 仍决定 `best.pt`，新增分组指标不改变这一规则。

## 5. 中断续训

每次新登录先 `source enter_f2llm.sh`；没有完成的当前结构，从它自己的 `last.pt` 继续：

```bash
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_openbc_full_1x1"  # 正在训练 2x2 时改为对应目录
python -u scripts/train_point_readout.py \
  --config "$RUN/resolved_config.json" --profile full \
  --resume "$RUN/last.pt" --output-dir "$RUN" \
  --console-every-updates 200
```

恢复 sidecar、优化器、学习率调度、随机状态、epoch 和批次位置；原模型与数据身份必须
一致。不要对已有 INS 训练使用新配置续训，它是新的数据实验。正常 Ctrl+C/SIGTERM 会
请求在完成的更新处保存；断电/SIGKILL 只能恢复最近成功写出的 checkpoint。长验证
开始前已保存，待验证标记也会恢复。不要给续训加覆盖参数。

## 6. 最终测试与离线复评分

各自训练完成后，用验证选出的 `best.pt` 在 test 上独立测试：

```bash
(
  set -e
  for block in 1 2; do
    RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_openbc_full_${block}x${block}"
    python -u scripts/train_point_readout.py \
      --config "$RUN/resolved_config.json" --profile full \
      --resume "$RUN/best.pt" --evaluate-only --split test \
      --output-dir "$RUN/test_best"
  done
)
```

生成 `test_predictions.jsonl` 和 `test_metrics.json`。同一个 test 集对两个结构各测一次；
不依据 test 选训练步数。已有预测重算指标不需要模型或 GPU：

```bash
RUN="$FIELD_TO_LLM_RUNS_DIR/multisource_openbc_full_2x2"
python scripts/score_point_readout.py \
  --qa-dir "$FIELD_TO_LLM_ROOT/data/multisource_openbc_v1/full" --split test \
  --predictions "$RUN/test_best/test_predictions.jsonl" \
  --output "$RUN/test_best/test_rescored.json"
```

## 加速边界与实验判断

训练缓存原生 chat 模板的分词结果（CPU 有界缓存），避免每轮重复分词；完整 token
预检仍覆盖全部题目。QA 裁剪缓存减少随机读取巨大源文件；启动审计仍回读源裁剪。
验证/测试在同一场的连续题目间复用场编码，在同题生成 token 间复用场 K/V 投影，
每道题仍单独重置文本 KV。参数变化、训练模式或重新绑定场会清空推理投影缓存，
训练梯度不复用旧图。未缩减验证集、任务、生成长度或数值精度。

本地小模型测试检查缓存前后 loss/梯度和生成/评分一致；真实 A80080GB/Qwen14B 的
显存、速度与准确率尚未实测。本轮预计训练更久，因为状态数扩大八倍，并增加验证
状态；缓存加速不意味着可在 24 小时内完成。用 `training_plan.json` 的真实更新数、
稳定后的 step 用时及一次完整验证用时估算，再安排资源和续训。

增加数据值得做，但不是已证明的唯一瓶颈。训练/验证都差更像优化或接口能力不足；
训练好而跨来源验证差才更支持覆盖不足的解释。本轮同时增加数量、物理多样性和常值
支持，不能把提升单独归因于“数据量”。先完成这组试验并分析按来源/任务的学习曲线；
若要分离数量效应，后续在同一新来源混合比例上做小规模对照。大尺寸、IID、整类物理
来源保留作为 OOD 测试尚未纳入，本轮不能证明这些泛化能力。

## 本地验证记录

本轮移除 periodicBC 后，来源校验、实际多来源构建及配置相关测试 32 项通过。
其中检查了默认下载只包含三个完整 The Well 源，以及单一 Euler 来源获得与 INS
相同的裁剪配额。`git diff --check` 通过；未执行远端下载或训练。

相关 9 个测试文件共覆盖 177 项通过、1 项跳过，包含旧单来源路径、多来源真实微型
HDF5 集成、下载清单/收据、缓存前后训练 loss/梯度与生成结果一致、连续训练和严格
续训参数逐值一致、独立测试与离线分组评分一致。跳过的是需要本地真实 Qwen 分词器
资产的检查。下载、构建和训练三个 CLI 的 `--help` 正常，`git diff --check` 通过。
此前五来源方案另已实际运行四个 The Well 源的 `--manifest-only`，固定版本和字节数
输出正常；该步骤不下载 HDF5。本段记录的是此前实现时执行的检查。

Windows 联合回归曾遇到临时目录短/长路径不同和默认 GBK 读取 UTF-8 的旧测试问题；
仅给测试进程设置仓库内临时目录并启用 Python `-X utf8` 后，两项一并复测通过。
未为通过测试修改生产逻辑。完整真实源下载/全字段扫描、远程 Qwen14B 的显存、吞吐
和最终准确率仍需按上文在服务器验证。
