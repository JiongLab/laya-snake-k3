# 适配过程：从模型体验到游戏发布

## 最终技术路线

上游游戏逻辑 → Laya SDK 构造问题/分词 → 社区多语言 ONNX FP16 → SpaceMIT EP 与 CPU 固定分工 → SDK 输出方向概率 → 可见安全保护 → 走一步。整个游戏复用一个模型会话，每步真正推理，不缓存旧方向代替模型。

## 1. 先运行原模型

最初从 [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) 下载英文模型和 `multilingual` 子目录，固定修订 `1c5edc17a7acd8701df6fc341c0d179f1c62c982`。英文用于英文工单，多语言用于中文；最终游戏只需多语言版本，发布安装不再下载无关英文权重。

在 Bianbu 的独立 venv 中复用系统 PyTorch、numpy、safetensors；tokenizers 0.22.2 因 RISC-V wheel 不直接可用而在 K3 原生构建。没有将 x86/ARM wheel 强行安装。

首先用分类、退款意图和紧急程度问题验证 SDK 能实际推理。Laya 是 typed-decision 模型，输出选项概率、是/否概率或评分，不是自回归文本生成器。小样本出现过错误判断，不能把概率当作可靠业务保证。

## 2. 解决 Python / PyTorch 的导入兼容

PyTorch 2.9 在 Python 3.14 下不支持这里的 `torch.compile` 路径，Transformers 5.0 的 ModernBERT 在导入时就会遇到装饰器检查。

`snake_k3/k3_compat.py` 仅在这组版本、仅在 ModernBERT 导入期间临时绕过可选 compile 装饰器，随后立即恢复原 `torch.compile`。CPU 推理走 eager；没有永久 monkey-patch 系统模块、修改权重或假装启用编译优化。这是已验证版本组合的局部适配，不是适用于任意 PyTorch/Transformers 的全局修复。

## 3. 接入开源贪吃蛇

从 [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) 提取 Snake 模块，固定提交 `fc1df62828a3fedf4d8229fdac1cbd85f1cdf337`。保留游戏规则、环路保护、真实方向概率和终端画面，将 Apple MLX 后端替换为已安装的 Laya SDK。

游戏不是模型从原始棋盘学会策略：规划器先描述每个方向是否碰撞、是否安全以及食物进度，模型基于描述选择。保护层发生干预时保留原概率和首选，并单独记录实际执行动作。

## 4. 启动与每步计算优化

- CPU 加载时不再生成随后必被 checkpoint 覆盖的随机参数；使用 `no_init_weights()`，保留严格权重加载与正常 RoPE 缓冲区。
- 去掉游戏启动时不必要的 6 次隐藏预热，首次真实推理直接用于游戏。
- 默认只询问移动方向，两个辅助指标标记为未计算；`--full-metrics` 恢复三问题，不伪造或复用概率。
- 固定 PyTorch/OpenBLAS 线程数，避免默认线程过多造成竞争。

历史早期 PyTorch 三问题游戏平均约 1053 ms/步；方向单问题优化后约 519 ms/步。这些来自先前独立实验，不是同一时间、同一循环的严格消融，不能逐项相乘推导加速比。

## 5. ONNX 与 AI Core

模型并非只能用 PyTorch。最终选择 [mizchi/laya-multilingual-onnx](https://huggingface.co/mizchi/laya-multilingual-onnx) 的固定快照。SDK 输入顺序为：`input_ids`、`attention_mask`、`marker_pos`、`marker_mask`、`qtype`；输出 `logits` 和 `act_logits`。继续使用原 SDK 的校准与答案格式，避免把 logits 直接当概率。

仅安装 ONNX Runtime 不代表启用了 AI Core。Bianbu 同时存在通用 ORT 和厂商 ORT；后端显式选择 `/usr/lib/python3.14/dist-packages` 中的厂商实现，并检查版本和 Provider。没有替换系统通用 ORT。

最终 EP 设置在 `snake_k3/onnx_backend.py`：4 工作线程，亲和性 `8;10;12;14`；精度级别 2；禁用 FP16 epilogue。保留已验证的矩阵、逐元素、归一化等加速；Transpose、Concat 等不可靠路径交给 CPU。

为什么不“全部 AI Core”：算子支持列表不等于当前模型所有形状/组合均能可靠执行。曾复现 Concat 相关误差，以及扩展配置下 Transpose/相邻算子组合的跨批次误差。局部固定分工可以保留速度并通过原精度标准。这里没有宣称修复厂商内核。

INT8 做过实验，但不能仅凭位宽判断更快；当前正式路径仍为已验证的 FP16，不发布未经验证的 INT8 配置。

## 6. 修复运行中停顿

早期为绕过输入形状问题，曾按长度白名单在两种配置间切换。这会重建模型会话，长蛇后期触发约 5 秒停顿和重复初始化警告，属于失败方案，不应复现或发布。

随后先固定为约 160 ms 的保守矩阵配置，消除了重载，但损失速度。继续做算子消融后，将 Transpose 固定在 CPU，其余已经验证的扩展加速保持启用，恢复约 110–120 ms 的稳定推理。

兼容性对照包含 121 个不同输入、两轮重复：早期/后期单问题及三问题、批次 1/2/3/4、实际 59–64 token 和填充到 65/70/80/96/128 token。最大概率绝对差 0.001681（0.1681 个百分点），低于原 0.002 门槛，首选结果一致；不能据此外推任意长文本或任意模型。

历史 1,200 步长蛇测试排除首步后平均 109.79 ms，P95 115.00 ms，最大 122.44 ms；第 738 步填满 96 格，零死亡、零保护介入。首步初始化约 2236 ms。安装包自身的验证情况单独见 [VALIDATION.md](VALIDATION.md)。

## 7. 从现场目录到可发布软件

本发布移除原设备用户名/绝对目录，使用用户数据目录；兼容补丁归入模块；依赖及模型锁定；补齐安装、哈希验证、源码构建、许可证和回归测试。模型权重、桌面截图、SSH 信息和其他项目配置不进入仓库。

通用的方向单问题优化提交上游；K3 的版本适配和 EP 分工保留在此仓库。上游接受前，复现以本仓库 Release 为准，不要求读者手工重演全部失败实验。
