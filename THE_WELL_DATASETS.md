# 本轮数据源清单与实际布局

2026-10-21 更新：完整源范围与下列布局保持一致；新的训练配额为 INS 32,768、
Acoustic 65,536、Euler openBC 262,144、Gray–Scott 65,536，合计 425,984。
具体步骤见 [v2 实验方案](MULTISOURCE_V2_RUNBOOK.md)。下文的 131,072/32,768
配额描述对应保留的旧配置，不是 v2 配额。

核查日期：2026-10-20。下表区分发布者介绍、当前固定版本文件清单及实际 HDF5 文件头。
本地核查以元数据和小型测试文件为主。用户提供的服务器日志已显示 acoustic 完整源
下载及验证成功；Euler 尚在下载，已收到三个 test 分片的实际文件头。其余来源的
完整状态以服务器下载脚本扫描全部文件后生成的收据为准。

体积单位为十进制 GB/TB；HDF5 字节数不含 README、stats 等很小的辅助文件。本轮移除
Euler periodicBC，保留三个 The Well 源，合计 **3,097,472,054,896 字节 = 3.097 TB**。
加上已有 INS 源为 **3,117,722,698,656 字节，约 3.118 TB**；
还需要 QA、模型、训练输出空间。

| 数据源 | 完整当前 HDF5 字节数 | 约 GB / TB | 原网格 | 存储帧数 | 文件数 train/valid/test |
|---|---:|---:|---|---|---|
| acoustic_scattering_inclusions | 291,873,226,752 | 291.873 GB | 256×256 | 102，全部 36 文件头已核查 | 28 / 4 / 4 |
| euler_multi_quadrants_openBC | 2,647,658,116,720 | 2.648 TB | 512×512 | 101，三个 test 分片文件头已确认；未遍历全部文件 | 100 / 20 / 20 |
| gray_scott_reaction_diffusion | 157,940,711,424 | 157.941 GB | 128×128 | 1,001，全部 18 文件头已核查 | 6 / 6 / 6 |
| 已有 INS-Tracer | 20,250,643,760 | 20.251 GB | 128×128 | 101 | 一个完整文件；本项目按轨迹分割 |

periodicBC 本轮不下载、不构建 QA；其工具支持和固定清单保留为后续可选来源，已经存在
的文件也不会被删除。它单独增加 2,647,655,177,000 字节（约 2.648 TB），未计入上表合计。

## 声学夹杂介质

[发布者介绍](https://polymathic-ai.org/the_well/datasets/acoustic_scattering_inclusions/)描述
多材料中的声波传播，包含不连续界面和不规则夹杂区域。它补充波纹、尖锐介质边界以及
随空间变化的静态参数场，与 INS 的流动场不同。

当前 [HF 正式 data 目录](https://huggingface.co/datasets/polymathic-ai/acoustic_scattering_inclusions/tree/c17cd1d16fd351594ccea628e575650b404c4a91/data)
的全部 36 个文件头已核查：**2,798 / 400 / 400 条轨迹，总计 3,598**。两个训练分片
各 99 条，其余各 100 条；每条 102 帧。官网的“4,000 条、101 步”不能直接当成这份
当前文件清单的实际规格。本实验明确使用完整当前 `data/`，而非为凑官网数量纳入
发布者标记的 `old_and_problematic_data`。

| 本轮真实 key | 单文件布局（N 为该文件轨迹数） | 用途 |
|---|---|---|
| `t0_fields/pressure` | `[N,102,256,256]` | 动态声压 |
| `t1_fields/velocity` | `[N,102,256,256,2]` | 两个速度分量分别作为标量场 |
| `t0_fields/density` | `[N,256,256]` | 静态材料密度，不沿时间复制 |
| `t0_fields/speed_of_sound` | `[N,256,256]` | 静态声速，不沿时间复制 |

这里“静态”是时间不变，不表示空间常值。部分裁剪可能恰好位于均匀介质内，故另有
`is_constant` 指标。使用每裁剪 z-score 后，本轮评估形状/相对读数，不评估材料参数
的绝对物理量恢复。

## Euler 开放边界（本轮）与周期边界（可选）

[openBC 介绍](https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_openBC/)
与 [periodicBC 介绍](https://polymathic-ai.org/the_well/datasets/euler_multi_quadrants_periodicBC/)
描述从分块初态演化出的激波、接触间断与稀疏波。本轮只使用 openBC，因此不能检验
两种边界条件间的泛化；openBC 通过已固定的 HF 版本下载，避免新增另一套约 2.648 TB
的周期边界数据。文件切片不同不会改变采样权重。

官方介绍的 **10,000 条轨迹、约 5.17 TB 是两种边界的合计**，不能给每个边界各算一次。
按其 10 组参数、每组 500 条推算，每个边界 5,000 条。这里尚未遍历 Euler 全部文件头
复核轨迹总数；下载后的 `split_trajectory_counts` 才是实际最终汇总。

此前对可选 periodic 源的实际抽查：train 文件 400 条，valid/test 文件 50 条，实际存储 **101 帧**，
而官网描述 100 个时间步。openBC 本轮选择 HF 的重新分片版本，该版本
[官方 YAML](https://huggingface.co/datasets/polymathic-ai/euler_multi_quadrants_openBC/blob/84e3eb65faf286533714295515caf9b99b4717a7/euler_multi_quadrants_openBC.yaml)
列出 512×512、101 帧。用户提供的服务器文件头确认：test 中 gamma=1.13、1.22 的
`chunk_0` 各 40 条，gamma=1.13 的 `chunk_40` 实际 10 条，均为 101 帧、512×512；
不能套用 Flatiron 对应文件的每片 50 条。HF train 每片 21,181,261,748 字节，
valid/test 各包含 10 个此大小文件和 10 个 5,295,335,348 字节文件。
完整下载器会读取每片真实布局，不根据文件体积猜测索引。

其中 gamma=1.13 的 test `chunk_40` 发布者 SHA256 已通过，但根属性
`n_trajectories=-1550`，与三个标量字段和动量字段共同确认的 10 条轨迹矛盾。
校验器只对通过发布者 SHA256 的 openBC 负数轨迹属性启用兼容处理：要求所有
sample-varying 字段轨迹轴一致且为正，并包含 density/energy/pressure/momentum，
随后继续检查全部坐标、字段形状、有限值与轨迹哈希。验证记录的 `metadata_corrections`
保留声明值、实际值及字段依据；原始 HDF5 字节不修改。其他布局矛盾仍报错，
不能把此例推断为全部 140 个分片都已验证。

| key | 布局（openBC 上述三个实际文件头） | 本轮输入 |
|---|---|---|
| `t0_fields/density` | `[N,101,512,512]` | 一个标量场 |
| `t0_fields/pressure` | `[N,101,512,512]` | 一个标量场 |
| `t0_fields/energy` | `[N,101,512,512]` | 一个标量场 |
| `t1_fields/momentum` | `[N,101,512,512,2]` | 两个动量分量分别作为标量场 |

动量不是速度，代码不会把它改名或除以密度再当速度。512×512 源保留完整，训练从中
取高宽 1～128 的裁剪，未将整幅场缩放到 128×128。

512×512 的单帧、单标量通道可无重叠分为 16 个 128×128 块，但本轮采样保持随机高宽
和位置，不固定为这种平铺。训练预算是人工分配的：先前 Euler 的两种边界平分该家族
32,768 个裁剪，现在 openBC 独占 32,768，与 INS 相同。原网格面积决定可用空间和
裁剪候选数，不自动决定实际训练量，也不能把同一轨迹的 16 块视为 16 条独立轨迹。
所有裁剪继承轨迹的 train/valid/test 划分，禁止先平铺再将同一轨迹的块随机拆到各 split。

openBC 来自
[固定 HF 版本](https://huggingface.co/datasets/polymathic-ai/euler_multi_quadrants_openBC/tree/84e3eb65faf286533714295515caf9b99b4717a7/data)。
下述 periodicBC 信息仅供后续可选实验使用，不是本轮下载要求。其公开 HF 地址本次
返回 401，未核实可用镜像；其完整文件来自
[发布者官方注册表](https://github.com/PolymathicAI/the_well/blob/808ef1b6a7b1f2e726c1fd0b9749f4e636d7b58d/the_well/utils/registry.yaml)
列出的 Flatiron 地址。固定清单包含全部 30 个 HDF5 及 stats，服务器无需另访问 GitHub
才能知道该下载什么；但 Flatiron 的服务器端下载速度仍未知。

## Gray–Scott 反应扩散

[发布者介绍](https://polymathic-ai.org/the_well/datasets/gray_scott_reaction_diffusion/)描述
两个化学组分 A/B 的反应扩散和图案形成。它补充斑点、条纹及不同参数下的空间纹理，
帮助检验接口是否只适应流体速度场。

[固定 HF 版本](https://huggingface.co/datasets/polymathic-ai/gray_scott_reaction_diffusion/tree/913ef5679d5c4b4ccd6fcd52c055509dfbf190f9/data)
全部 18 个文件头已核查：**960 / 120 / 120 条轨迹，总计 1,200**，每条 **1,001 帧、
128×128**。两个真实 key 为 `t0_fields/A` 和 `t0_fields/B`，均为
`[N,1001,128,128]`。所有参数组、官方三个 split 都包含在下载清单中。

## 版本、校验和与完整性范围

| 来源 | 固定版本 | 校验依据 |
|---|---|---|
| acoustic | `c17cd1d16fd351594ccea628e575650b404c4a91` | 发布者 HF LFS SHA256 |
| Euler openBC | `84e3eb65faf286533714295515caf9b99b4717a7` | 发布者 HF LFS SHA256 |
| Euler periodicBC（可选，本轮排除） | 官方注册表提交 `808ef1b6a7b1f2e726c1fd0b9749f4e636d7b58d` | 完整字节数、布局/全字段检查、本地 SHA256；没有发布者逐文件 checksum |
| Gray–Scott | `913ef5679d5c4b4ccd6fcd52c055509dfbf190f9` | 发布者 HF LFS SHA256 |

清单在 `configs/source_manifests/<dataset>/source_manifest.json`，包含精确文件路径、
字节数、可用的发布者哈希和排除理由。三个 HF 来源从镜像传输时仍核对固定官方版本
的哈希。periodicBC 的 HTTP ETag/Last-Modified 仅用于版本变化检测，不冒充密码学
校验和。INS 的历史收据也是本地 SHA256/完整扫描，不是发布者校验和。

续传复用验证记录时会检查文件大小和修改时间；这种缓存不能证明文件没有被故意
修改后恢复时间戳。`--verify-only --rehash` 强制重读完整字节；训练启动另外回读每个
采样原始裁剪，并检查原始/归一化哈希、缓存逐值相等和全部 QA 真值。

构建出的 131,072 个训练裁剪及 QA 与这些 TB 级完整源是两层不同对象。更多源数据
提供可选的轨迹、时间、参数和空间结构；本轮并不对源中的每个像素或每帧都生成 QA。
采样预算、划分及命令见 [运行手册](MULTISOURCE_READOUT_RUNBOOK.md)。
