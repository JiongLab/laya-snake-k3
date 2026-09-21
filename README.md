# Laya 在 K3 上玩贪吃蛇

面向 **SpaceMIT K3、Bianbu 4.0.5、RISC-V 64 位、8 GB 内存**的离线演示。每步调用本地 Laya 多语言模型，默认使用 ONNX FP16 + AI Core，保留可见的环路安全保护。

> 当前 `v0.1.0` 为预发布候选版：通用测试、新环境初始化和 K3 实际安装已通过；设备重启后，完整硬件验收按录屏优先安排暂缓。请先看 [验收状态](docs/VALIDATION.md)，不要将历史现场版性能等同于新包已通过全部验收。

这不是大模型聊天，也不是模型直接读取像素：规则层生成各方向的描述，Laya 给出方向概率；安全保护介入时界面明确标注 `SHIELD`。不需要 API Key。

## 安装和复现

1. 阅读 **[从模型下载到运行的完整复现指南](docs/REPRODUCE.md)**，先安装文档中锁定的 Bianbu 系统依赖。
2. 从 [Releases](https://github.com/JiongLab/laya-snake-k3/releases) 下载 `.deb` 及 `SHA256SUMS` 并校验。
3. 安装应用，然后以普通桌面用户初始化：

```sh
sudo apt install ./laya-snake-k3_0.1.0_riscv64.deb
laya-snake-k3-setup
laya-snake-k3
```

初始化会创建独立 Python 环境，并下载约 1.9 GB 的模型文件。之后推理完全离线。建议预留至少 5 GB 磁盘，不要同时运行多个模型游戏。

终端建议至少 104 列 × 35 行。空格暂停、R 重开、Q 退出。默认只计算方向；两个辅助指标显示 `NOT COMPUTED`，用 `--full-metrics` 可以恢复三问题推理。启动约 25 秒，不能与每步推理时间混为一谈。

## 资源占用

以下针对默认的 **12×8 棋盘、方向单问题、ONNX FP16 / SpaceMIT、单个游戏进程**。线程数是程序配置，不等于持续利用率；文件大小也不等于运行内存。

| 资源 | 当前配置或可确认的数据 | 说明 |
|---|---|---|
| 设备内存 | 历史实测设备为 **8 GB RAM** | 这是设备容量，不是程序占用，也不是已验证的最低内存要求 |
| 运行内存 | **当前 ONNX 游戏的稳定 RSS、启动峰值仍待专项测量** | 不引用早期 PyTorch 分类实验的数据代替；内存还包含运行库、分词器和中间张量，不能只按 FP16 权重大小估算 |
| AI Core | **4 个 AI 工作线程**，自动绑定 **8、10、12、14** | 没有默认占满 Core 8–15；实际忙碌程度随推理阶段变化，尚无利用率百分比实测 |
| 普通 CPU | ONNX CPU 执行配置为 intra-op **1**、inter-op **1**；PyTorch/OpenBLAS/OMP 线程上限为 **4** | 分词、游戏逻辑、画面及指定 CPU 算子仍消耗 CPU；各线程池不一定同时活跃，这不是整个进程的总线程数 |
| 模型磁盘 | 约 **1.9 GB** | 包括原 SDK 权重、社区 ONNX 和分词/配置文件；保留 SDK 权重用于 PyTorch 对照，并非默认同时加载两份模型 |
| 安装包下载 | `.deb` 为 **14,473,918 字节（约 13.80 MiB）** | 对应 v0.1.0，包含应用和 Python wheels，不含模型及系统依赖；不是安装后的总占用 |
| 磁盘预算 | 建议至少预留 **5 GB 可用空间** | 为模型、用户环境与初始化留余量；这是准备建议，不是实测安装总量。额外源码、重复模型目录和持续保存的记录需另计 |
| 网络 | 首次安装/下载需要网络；初始化后模型推理离线 | 不调用云端模型，不需要 API Key |

建议只运行一个模型游戏实例；`--full-metrics` 会增加问题数量与计算量，不能直接沿用默认模式的资源和延迟数据。当前没有功耗或温度实测，不宣称低功耗或特定散热要求。

资源数据补测时应分别记录启动峰值与稳定运行 RSS、整个进程与各线程的 CPU 利用率，并注明系统版本、是否渲染、是否有其他负载及采样时长。新包硬件验收仍未完成，详见 [验收记录](docs/VALIDATION.md)。

## 代码、上游与验证

- [我们自己的程序做了什么](docs/REPRODUCE.md#我们自己的程序做了什么)：原项目与适配工作的分工、自动绑核、推理优化和兼容性边界。
- 游戏来源：[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)，Apache-2.0；保留原游戏规则和保护层。
- SDK 模型：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 的 multilingual 子目录。
- ONNX 模型：[mizchi/laya-multilingual-onnx](https://huggingface.co/mizchi/laya-multilingual-onnx)，社区独立转换，不冒充官方 ONNX 发布。
- [适配过程与性能依据](docs/PORTING.md) · [上游贡献范围](docs/UPSTREAM.md) · [源码构建](docs/BUILD.md) · [验收记录](docs/VALIDATION.md) · [第三方许可](docs/THIRD_PARTY.md)。

先前固定配置在真实 K3 的 1,200 步长蛇测试中，排除首步初始化后平均 109.79 ms/步，P95 115.00 ms，最大 122.44 ms。桌面渲染、其他程序和系统负载会影响速度。这不是所有输入或硬件下的性能承诺。

Transpose 等存在兼容性问题的路径固定留在 CPU，其余已验证路径使用 Core 8/10/12/14；不根据输入长度重载模型，不静默切换到全 CPU。是应用级算子分工适配，不是厂商内核修复。

仅发布应用及适配源码，不包含模型、系统运行库、个人路径、凭据或原设备配置。模型来源和每个文件的 SHA-256 见 [models.lock.json](models.lock.json)。
