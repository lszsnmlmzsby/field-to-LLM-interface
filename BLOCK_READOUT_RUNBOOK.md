# 多点一个 token：真实场可行性实验

沿用 `POINT_READOUT_RUNBOOK.md` 的服务器环境、Qwen 及 PDEBench 数据。只新增局部块编码，
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
full 每种评估形状 32 场；本次不必运行 full。

## 服务器命令

在服务器仓库根目录、已激活的 tcenv 环境中执行。保持已有的 `FIELD_TO_LLM_ROOT` 和 `PDEBENCH_HDF5`。

```bash
git pull
export FIELD_TO_LLM_MODEL_DIR="$FIELD_TO_LLM_ROOT/models/Qwen2.5-14B-Instruct"
export BLOCK_CONFIG=configs/field_to_llm_block_readout.yaml
python -m pytest tests/test_block_readout.py tests/test_point_readout.py -q
```

先检查训练/生成闭环，包含奇数形状。构建器拒绝覆盖已有数据；已经成功构建则跳过 build。
输出目录也必须是新的。两组依次运行，不同时占用 A6000。

```bash
python scripts/build_point_readout_qa.py --config "$BLOCK_CONFIG" --profile smoke
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile smoke \
  --block-shape 1 1 --output-dir "$FIELD_TO_LLM_ROOT/runs/block_smoke_1x1"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile smoke \
  --block-shape 2 2 --output-dir "$FIELD_TO_LLM_ROOT/runs/block_smoke_2x2"
```

主实验：

```bash
python scripts/build_point_readout_qa.py --config "$BLOCK_CONFIG" --profile pilot
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --block-shape 1 1 --output-dir "$FIELD_TO_LLM_ROOT/runs/block_pilot_1x1"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --block-shape 2 2 --output-dir "$FIELD_TO_LLM_ROOT/runs/block_pilot_2x2"
```

需要先估算时间时，第一次训练可添加 `--stop-after-updates 200`，会按原完整日程保存并退出。
之后去掉该参数，使用相同命令、相同输出目录，并加 `--resume .../last.pt` 继续。
不得为了短跑修改 epochs/max_updates 后再恢复。示例：

```bash
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --block-shape 2 2 --output-dir "$FIELD_TO_LLM_ROOT/runs/block_pilot_2x2" \
  --resume "$FIELD_TO_LLM_ROOT/runs/block_pilot_2x2/last.pt"
```

独立评估两个 best（避免将训练期显存和独立推理显存混用）：

```bash
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --block-shape 1 1 --evaluate-only --split val \
  --resume "$FIELD_TO_LLM_ROOT/runs/block_pilot_1x1/best.pt" \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/block_eval_1x1"
python -u scripts/train_point_readout.py --config "$BLOCK_CONFIG" --profile pilot \
  --block-shape 2 2 --evaluate-only --split val \
  --resume "$FIELD_TO_LLM_ROOT/runs/block_pilot_2x2/best.pt" \
  --output-dir "$FIELD_TO_LLM_ROOT/runs/block_eval_2x2"
python scripts/compare_block_readout.py \
  --reference "$FIELD_TO_LLM_ROOT/runs/block_eval_1x1" \
  --compressed "$FIELD_TO_LLM_ROOT/runs/block_eval_2x2"
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
  --checkpoint "$FIELD_TO_LLM_ROOT/runs/block_pilot_2x2/best.pt"
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
真实 Qwen 14B 的 A6000 显存、速度和精度需要服务器 smoke/pilot 实测。
