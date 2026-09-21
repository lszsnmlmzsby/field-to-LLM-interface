# 标准化点值生成实验

本路线研究冻结 Qwen 能否直接生成数值场中的近似标准化值。它复用已提交的
逐格 CNN、二维 spatial adapter 和交叉注意力，不改变原选择题入口，也不训练
raw 反标准化、比较题或随机场。这里没有训练效果或 A6000 实测性能的承诺。

## 清理说明

此次检查发现未提交的分块压缩原型修改了三个已有文件，并留下三个新文件。
这些更改已备份到本地 `outputs/rollback_backup_20260915_233744/`，随后撤回：

- `scripts/train_tensor_qwen_cross_attention.py`
- `src/tensor_compression/downstream/patch_qa_prompt.py`
- `src/tensor_compression/downstream/variable_shape.py`
- `src/tensor_compression/downstream/block_memory.py`
- `scripts/build_block_compressed_qa.py`
- `configs/field_to_llm_block_compressed.yaml`

前三个文件恢复为当前 HEAD。后三个原型文件从活跃代码目录移除。备份属于本地
历史，不上传 Git。已有可变形状训练代码与实验成果不撤回。

## 任务与答案

| 参数名称 | 查询 | 输出顺序 |
|---|---|---|
| `single_point` | 一个坐标 | 一个数值 |
| `multi_point` | 2～4 个不同坐标 | 按问题列出的顺序 |
| `line_profile` | 连续的水平或竖直短剖面 | 从起点向右或向下，包含起点 |
| `region_values` | 默认 `2×2` 小区域 | 每行从左到右，各行从上到下 |

问题使用英文，与已有训练语言保持一致。每题输出一个 JSON 数组，包括单点：
`[-0.7]` 或 `[-0.7, 1.2, 0.0]`。这是受格式约束的短答案生成，不是任意长篇
自然语言解释。推理使用完整词表上的贪心生成，没有候选值、选项排序、答案
白名单或按真值截断输出。EOS 终止，生成上限耗尽而没有 EOS 则记失败。

神经网络只接收完整标准化场和公开提示文字。`oracle.query_spec`、真值、任务
ID、参考答案都不能成为推理输入；坐标仅以自然语言出现。任务 ID 仅用于数据
构建和分组报告。

## 数据和评分协议

- 输入 HDF5 中选定变量必须共享 `[trajectory,time,height,width]` 维度。不同变量
  的同一 trajectory 必须表示同一条轨迹；不支持自动猜测或转置其他 HDF5 布局。
- 原轨迹按固定 seed 分为 80%/10%/10%；每份至少两条，当前入口至少需要 20 条
  原轨迹。这只是工程下限，小样本不能代表充分的泛化证据。
- 先划分轨迹，再采样时间和裁剪窗口。同轨迹所有变量留在同一集合；禁止重复的
  源变量/轨迹/时间/窗口/形状组合。样本量超出可用组合时明确报错。
- 非有限值和常数场被排除，次数写入 metadata。普通查询不按答案值、数值差距
  或预测难度筛选。四种查询默认每个场状态各一道，避免人为偏向极值样本。
- 真值为 CPU float32 总体标准差归一化、加 `1e-6`、再经 FP16 舍入的场值：
  `z = ((x - mean) / (std_population + 1e-6)).half().float()`。
- 默认参考回答保留一位小数，评分比较未作文本舍入的真值，命中条件
  `abs(prediction - target) <= 0.2`。这是锁定的可行性协议，不是行业精度标准。
- 只接受一个非空、有限数字构成的 JSON 数组。科学计数法与正常空格有效；
  布尔值、NaN、Infinity、嵌套数组、额外说明和多个候选回答无效。
- 数值按顺序逐项比较。数量不符或未正常结束时整题各点均记错，不从错误输出中
  搜索任意“命中”的子串。数值相同可以合法出现，不按重复数值判断重复位置。
- 主指标包括回答有效率、逐点命中率、整题全对率、各任务逐点命中率的宏平均，
  并按形状和变量报告。所有失败都在分母内。不同长度任务看分项指标，不只看总分。
- 验证中分别报告训练已见形状与未见形状；best checkpoint 只按验证集的已见
  形状任务宏平均选择。test 不参与训练过程或 checkpoint 选择。
- `states.jsonl` 保存源位置、归一化审计值以及原场/标准化场 hash；QA 文件保存
  问题、参考文本和 oracle。metadata 对四个 JSONL 文件记录 SHA256。读入时校验
  文件，训练前重放每个裁剪场并重算每个答案。无需扫描并哈希完整大 HDF5 文件。

默认 `smoke` 有 32 个训练场状态，`pilot` 512 个，`full` 2048 个；全选四种查询时
分别生成 128/2048/8192 条训练问答。三者是独立数据和训练预算，不能用 resume
将 smoke 改为 full。smoke 仍然按轨迹隔离，其验证指标不是训练集过拟合指标。

## 代码入口

- `src/tensor_compression/downstream/point_readout.py`：任务定义、提示、真值与评分。
- `src/tensor_compression/downstream/point_readout_data.py`：真实场采样、划分、审计和重放。
- `scripts/build_point_readout_qa.py`：无需模型的构建命令。
- `scripts/train_point_readout.py`：单设备训练、断点恢复和生成评估。
- `scripts/score_point_readout.py`：离线重新评分，无需 HDF5 或模型权重。
- `configs/field_to_llm_point_readout.yaml`：模型、任务和 smoke/pilot/full 配置。

## 上传代码与工作站准备

本地先检查差异，再提交并推送；以下命令不会上传备份、数据或权重：

```bash
git status --short
git diff --check
git add .gitignore README.md POINT_READOUT_RUNBOOK.md configs/field_to_llm_point_readout.yaml scripts/build_point_readout_qa.py scripts/train_point_readout.py scripts/score_point_readout.py src/tensor_compression/downstream/point_readout.py src/tensor_compression/downstream/point_readout_data.py tests/test_point_readout.py
git commit -m "Add real-field standardized numerical generation experiments"
git push
```

下面为 Linux 工作站命令。进入项目 checkout 后拉取，或首次使用 `git clone <你的仓库地址>`。
将路径替换为工作站的实际路径：

```bash
git pull --ff-only
conda activate tcenv
pip install -r requirements.txt
export FIELD_TO_LLM_ROOT=/data/point_readout
export PDEBENCH_HDF5=/data/pdebench/your_real_fields.hdf5
export FIELD_TO_LLM_MODEL_DIR="$FIELD_TO_LLM_ROOT/models/Qwen2.5-14B-Instruct"
export FIELD_TO_LLM_HF_HOME="$FIELD_TO_LLM_ROOT/hf_cache"
hf download Qwen/Qwen2.5-14B-Instruct --local-dir "$FIELD_TO_LLM_MODEL_DIR"
nvidia-smi
```

预训练 Qwen 与本项目训练后的场侧 checkpoint 是不同资产。新入口从头训练场侧，
不要求旧服务器的 Stage 1/Direct-QA checkpoint。它不接受旧选择题 checkpoint 作为
`--resume`，也不提供隐式权重迁移。Qwen 参数冻结，但其激活反向传播仍占显存。

## 先检查已有 HDF5

```bash
python -c 'import os,h5py; f=h5py.File(os.environ["PDEBENCH_HDF5"],"r"); f.visititems(lambda n,x: print(n,x.shape,x.dtype) if isinstance(x,h5py.Dataset) else None); f.close()'
```

确认四维顺序和轨迹含义。只有部分变量时，可以在构建命令添加 `--fields Vx Vy`，
或者只写一个实际存在的变量名。若字段布局不是要求的形式，应先明确转换。

## 最小实验与数据检查

```bash
python scripts/build_point_readout_qa.py --profile smoke
python scripts/train_point_readout.py --profile smoke --audit-only --output-dir "$FIELD_TO_LLM_ROOT/audits/smoke"
CUDA_VISIBLE_DEVICES=0 python scripts/train_point_readout.py --profile smoke --output-dir "$FIELD_TO_LLM_ROOT/runs/smoke"
```

只开展单点和多点时，第一条命令改为：

```bash
python scripts/build_point_readout_qa.py --profile smoke --tasks single_point multi_point --output-dir "$FIELD_TO_LLM_ROOT/data/point_readout/smoke_points"
CUDA_VISIBLE_DEVICES=0 python scripts/train_point_readout.py --profile smoke --qa-dir "$FIELD_TO_LLM_ROOT/data/point_readout/smoke_points" --output-dir "$FIELD_TO_LLM_ROOT/runs/smoke_points"
```

训练跟随数据 manifest 中选定的任务，不必重复传 `--tasks`。原始轨迹不足、裁剪
超出源场、答案过长等均报错，不自动缩小数据或截断问题。已有输出目录不覆盖。

默认单卡 `batch_size=1`、梯度累积 4，不再要求选择题的三条原子组；同批次形状
一致，无空间 padding，也不重复数据补齐 batch。训练在设定更新步数内循环数据，
loss 对每个问题等权，问题内部对答案 token 求均值。没有选择题或 matched-group loss。

## 较大实验、恢复与测试

```bash
python scripts/build_point_readout_qa.py --profile pilot
CUDA_VISIBLE_DEVICES=0 python scripts/train_point_readout.py --profile pilot --output-dir "$FIELD_TO_LLM_ROOT/runs/pilot"
```

确认运行代价与结果后再选 full，避免在 A6000 上直接启动未知显存规模：

```bash
python scripts/build_point_readout_qa.py --profile full
CUDA_VISIBLE_DEVICES=0 python scripts/train_point_readout.py --profile full --output-dir "$FIELD_TO_LLM_ROOT/runs/full"
```

从最近已保存的更新恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_point_readout.py --profile pilot --resume "$FIELD_TO_LLM_ROOT/runs/pilot/last.pt" --output-dir "$FIELD_TO_LLM_ROOT/runs/pilot"
```

中断发生在未保存的更新时，从最近 `last.pt` 重做这些更新。checkpoint 保存场侧参数
与 buffer、优化器、scheduler、数据游标和 RNG。数据、模型结构、tokenizer 或训练预算
变化时拒绝恢复。本地模型分片在启动时进行 SHA256 审计，首次大权重审计需要额外
磁盘读取；Hub 模型记录已解析的 commit。设备路径可以迁移，但必须提供同一数据
和预训练模型资产；同一次实验不要在本地目录与 Hub 名称两种加载方式之间切换。

使用独立输出目录，对 best checkpoint 明确执行 test：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_point_readout.py --profile pilot --resume "$FIELD_TO_LLM_ROOT/runs/pilot/best.pt" --evaluate-only --split test --output-dir "$FIELD_TO_LLM_ROOT/evaluation/pilot_test"
python scripts/score_point_readout.py --qa-dir "$FIELD_TO_LLM_ROOT/data/point_readout/pilot" --split test --predictions "$FIELD_TO_LLM_ROOT/evaluation/pilot_test/test_predictions.jsonl" --output "$FIELD_TO_LLM_ROOT/evaluation/pilot_test/rescored.json"
```

离线评分按 QA ID 对齐，缺失预测计错，重复或未知 ID 报错。它忽略预测文件中
已有的 score 字段，重新根据数据集 oracle 评分。在线评估还报告生成时间与 CUDA
峰值 allocated 显存；时间包含场编码、生成、逐条数据读取及结果记录，不代表纯 kernel 时间。

## 验证

```bash
python -m pytest tests/test_point_readout.py tests/test_tensor_qwen_cross_attention.py tests/test_variable_shape_fields.py tests/test_mixed_shape_fields.py -q
```

新测试使用生成的 HDF5 测试夹具与小型随机初始化 Qwen，只验证实现；这些不是实验
数据或真实任务精度。测试覆盖真值重放、顺序、错误格式、数据污染、梯度边界、
缓存与无缓存生成一致性，以及中断恢复后的参数一致性。真实 Qwen14B/A6000 的
可行性仍需执行上面的 smoke 命令测量。
