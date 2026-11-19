# 实验协议

配置文件定义数据范围、初始化方式、查询目标和训练预算。共享流程按配置执行；每个实验使用自己的 QA 身份与运行目录。

## 数值问答

以下配置均使用 `field.py data/train/evaluate/infer`，操作方式统一见 [README](../README.md)。

| 配置（`configs/` 下） | 职责 |
| --- | --- |
| `field_to_llm_point_readout.yaml` | PDEBench 单源数值生成，11 类读值与定位任务 |
| `field_to_llm_block_readout.yaml` | 块 token 结构实验 |
| `field_to_llm_ins_tracer.yaml` | INS-Tracer 轨迹划分与数值生成 |
| `field_to_llm_multisource.yaml` | 四来源、131,072 个训练裁剪、两轮训练 |
| `field_to_llm_multisource_v2.yaml` | 四来源、425,984 个训练裁剪、14 类查询、两轮训练 |

v2 裁剪配额为 INS 32,768、Acoustic 65,536、Euler openBC 262,144、Gray–Scott 65,536。高宽独立取 1–128 的整数，形状划分和轨迹划分记录在 QA metadata。语言配对、换场与关闭场输入分别产生补充诊断结果。任务权重、评分容差和生成长度以配方为准。

完整官方源由下载清单和校验记录定义。裁剪数描述实验采样预算；训练数组是由完整训练 QA 生成的执行缓存。源、QA、tokenizer 与 checkpoint 身份共同约束恢复。

## 变形状选择题

| 配置 | 数据与训练 |
| --- | --- |
| `field_to_llm_variable_shape.yaml` | PDEBench 矩形场，原始 HDF5 输入与 scratch 初始化 |
| `field_to_llm_variable_shape_mixed.yaml` | 真实场、相关合成场和 IID 场，65,536 个完整配置训练状态 |

`field.py shape-data --config <配置> --profile <smoke|pilot|full>` 构建对应数据；`field.py paper-train --config <配置>` 训练交叉注意力模型；`shape-evaluate` 独立评估 checkpoint，`shape-summary` 汇总运行指标。合成场由对应配方显式选择。

## 论文两阶段模型

论文模型使用 `16×16` 场、256 个网格对齐 token，以及冻结的 Qwen2.5-14B-Instruct。训练包括场文本对齐、Direct-QA 初始化和交叉注意力学习。

```bash
python field.py paper-align --config configs/field_to_llm_stage1.yaml
export FIELD_TO_LLM_ALIGNMENT_CHECKPOINT=/absolute/path/to/alignment_best.pt
python field.py paper-data --config configs/field_to_llm_stage1.yaml
python field.py paper-matched-data --config configs/field_to_llm_stage1.yaml
python field.py paper-direct --config configs/field_to_llm_direct_qa.yaml
export FIELD_TO_LLM_DIRECT_CHECKPOINT=/absolute/path/to/adapter_best.pt
python field.py paper-train --config configs/field_to_llm_cross_attention.yaml
export FIELD_TO_LLM_CROSS_CHECKPOINT=/absolute/path/to/cross_attention_best.pt
python field.py paper-benchmark --config configs/field_to_llm_benchmark.yaml
```

各训练命令支持在 `field.py` 前加 `python -m torch.distributed.run --standalone --nproc_per_node=N`，将原命令开头的 `python` 替换为该启动器。`PDEBENCH_HDF5` 指向完整、通过官方 MD5 校验的源文件。`paper-baseline` 提供序列化场的冻结模型基线。

## 初始化消融

`field_to_llm_cross_attention_scratch.yaml` 从原始 HDF5 读取场，联合训练随机初始化的场编码器、空间模块、值分支和交叉注意力桥。它复用固定 matched-QA，默认进行 3,000 次更新的验证筛选，入口为 `paper-train`。

以下配置通过 `paper-ablation direct` 和 `paper-ablation dense` 分阶段执行：

| 配置 | 初始化条件 |
| --- | --- |
| `field_to_llm_stage1_reference.yaml` | 学习得到的空间适配器与全部潜在通道 |
| `field_to_llm_stage1_adapter_ablation.yaml` | 确定性随机空间适配器与全部潜在通道 |
| `field_to_llm_no_learned_stage1_ablation.yaml` | 相同适配器重置，输入保留标准化标量通道 |

`paper-compare` 比较验证结果。测试阶段使用 `stage1_test_protocol_lock.example.json` 的协议锁，将所选运行、数据审计及 checkpoint 固定下来。协议锁包含代码内容指纹；整理后的新代码使用相应的新审计记录，既有记录随原提交保留。

## 实现定位

`workflows/train_point_readout.py` 编排数值问答训练和评估，`chat_point_readout.py` 编排交互推理；两者共用生成与 checkpoint 加载实现。`train_tensor_patch_text_alignment.py`、`train_tensor_llm_adapter.py`、`train_tensor_qwen_cross_attention.py` 分别实现论文对齐、Direct-QA 和交叉注意力阶段，后一阶段复用前一阶段的模型与运行工具。

输入运行时由 `readout_distributed.py` 负责分片和通信，`readout_prefetch.py` 负责有界 CPU 队列，`readout_input_runtime.py` 负责预取计划和阶段计时；不可变数组格式由 `downstream/readout_training_cache.py` 定义。
