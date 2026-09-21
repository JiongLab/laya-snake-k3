# Laya 在 K3 上玩贪吃蛇

面向 **SpaceMIT K3、Bianbu 4.0.5、RISC-V 64 位、8 GB 内存**的离线演示。每步调用本地 Laya 多语言模型，默认使用 ONNX FP16 + AI Core，保留可见的环路安全保护。

> 当前 `v0.1.0` 为预发布候选版：通用测试和新环境初始化已通过，完整安装包硬件验收因设备失去响应尚未完成。请先看 [验收状态](docs/VALIDATION.md)，不要将历史现场版性能等同于新包已通过全部验收。

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

## 代码、上游与验证

- 游戏来源：[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx)，Apache-2.0；保留原游戏规则和保护层。
- SDK 模型：[convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 的 multilingual 子目录。
- ONNX 模型：[mizchi/laya-multilingual-onnx](https://huggingface.co/mizchi/laya-multilingual-onnx)，社区独立转换，不冒充官方 ONNX 发布。
- [适配过程与性能依据](docs/PORTING.md) · [上游贡献范围](docs/UPSTREAM.md) · [源码构建](docs/BUILD.md) · [验收记录](docs/VALIDATION.md) · [第三方许可](docs/THIRD_PARTY.md)。

先前固定配置在真实 K3 的 1,200 步长蛇测试中，排除首步初始化后平均 109.79 ms/步，P95 115.00 ms，最大 122.44 ms。桌面渲染、其他程序和系统负载会影响速度。这不是所有输入或硬件下的性能承诺。

Transpose 等存在兼容性问题的路径固定留在 CPU，其余已验证路径使用 Core 8/10/12/14；不根据输入长度重载模型，不静默切换到全 CPU。是应用级算子分工适配，不是厂商内核修复。

仅发布应用及适配源码，不包含模型、系统运行库、个人路径、凭据或原设备配置。模型来源和每个文件的 SHA-256 见 [models.lock.json](models.lock.json)。
