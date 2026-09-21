# 来源、许可与二进制组件

项目整体 Apache-2.0，保留根目录 LICENSE 和 NOTICE。模型不打进应用包，按原来源分别下载；相关模型卡声明 Apache-2.0，本项目不改变其许可。[官方模型卡](https://huggingface.co/convaiinnovations/laya/blob/1c5edc17a7acd8701df6fc341c0d179f1c62c982/README.md)

游戏源自 [laya-mlx](https://github.com/mizorewww/laya-mlx) 的 Apache-2.0 代码。SDK `laya==0.3.4`、Transformers、Hugging Face Hub、tokenizers 为 Apache-2.0；其余 Python wheels 的具体许可保留在各 wheel 的 `dist-info` 元数据及许可证文件中。系统 PyTorch、ONNX Runtime、SpaceMIT EP/TCM 等由用户从 Bianbu 官方仓库安装，不由本 `.deb` 再分发。

发布包的 wheel 清单严格对应 `requirements-runtime.txt`。除 tokenizers 外为公开 PyPI 的原 wheel，没有修改源码。tokenizers 0.22.2 是 K3 上从该版本源码构建的 `cp39-abi3-linux_riscv64` wheel，依赖该 Bianbu 环境，未声称 manylinux 通用兼容；其 Apache 许可证另附 [tokenizers-LICENSE](licenses/tokenizers-LICENSE)。构建方法见 [BUILD.md](BUILD.md)。

模型卡记录的 ONNX 转换链：Convai 多语言模型 → `aac6fef/laya-multilingual-mlx` → 社区 ONNX 导出。发布清单锁定最终 ONNX 文件；不宣称本项目重新导出了模型，也不要求体验者拥有 Apple 设备。

本项目与上述维护者不存在隐含的官方认证关系。贡献 PR 的接受情况以对应仓库为准。
