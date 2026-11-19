# 数据源

当前多源实验使用 INS-Tracer 和三个固定版本的 The Well 数据集。HDF5 完整文件清单位于 `configs/source_manifests/`；下载与 QA 构建使用同一份源身份。

## 完整源与实验采样

以下字节数对应仓库固定清单，单位为十进制 GB/TB：

| 数据源 | HDF5 字节数 | 网格 | train/valid/test 文件数 |
| --- | ---: | --- | --- |
| Acoustic inclusions | 291,873,226,752 | 256×256 | 28 / 4 / 4 |
| Euler openBC | 2,647,658,116,720 | 512×512 | 100 / 20 / 20 |
| Gray–Scott | 157,940,711,424 | 128×128 | 6 / 6 / 6 |
| INS-Tracer | 20,250,643,760 | 128×128 | 单文件，按轨迹划分 |

四源合计约 3.118 TB；QA、训练数组、权重和运行结果分别配置空间。Euler periodicBC 有独立固定清单，由显式配置选择。实验裁剪配额见 [README](../README.md#实验协议与开发)。

## 字段布局

| 来源 | 标量字段 | 向量字段 |
| --- | --- | --- |
| Acoustic | `t0_fields/pressure`、`density`、`speed_of_sound` | `t1_fields/velocity` |
| Euler | `t0_fields/density`、`pressure`、`energy` | `t1_fields/momentum` |
| Gray–Scott | `t0_fields/A`、`B` | — |
| INS-Tracer | `sol/u`、`sol/v`、`sol/b` | 分别作为标量输入 |

The Well 动态标量通常为 `[N,T,H,W]`，向量为 `[N,T,H,W,C]`；静态材料场为 `[N,H,W]`。读取器根据实际 HDF5 结构选择轨迹、帧和分量。逐裁剪 z-score 定义问答中的标准化数值空间。

## 获取与校验

README 的 `download-ins`、`download-well` 命令获取完整源。The Well 下载器使用已固定的 manifest；`--verify-only` 检查已有文件，`--rehash` 重做文件摘要及有限值扫描。INS 下载器记录源信息与本地完整文件校验。各来源的发布者校验类型保存在对应清单中。

完整性由源清单和校验收据确定，QA manifest 记录实际使用的源身份、轨迹划分与裁剪。数据准备操作统一见 [README](../README.md#数据准备)。
