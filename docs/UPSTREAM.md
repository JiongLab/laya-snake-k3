# 上游贡献与独立包的边界

## 已提交

[mizorewww/laya-mlx PR #4](https://github.com/mizorewww/laya-mlx/pull/4)：`feat(snake): add optional movement-only inference mode`。

新增可选 `--fast`，每步仅计算方向；不改变默认三问题、移动描述或安全保护。未计算辅助指标用 `None`/JSON `null` 和 `NOT COMPUTED` 表示，通关/重开时也不显示虚假的 0 概率。用真实 tiny checkpoint 比较方向、消耗 token 和辅助指标；本地全套 50 项通过、1 项原有跳过。

PR 已提交不等于已合并，状态以链接为准。没有把 K3 的速度宣传为 Mac 上的速度。

## 保留在 K3 包

| 改动 | 当前归属 | 原因 |
|---|---|---|
| SpaceMITExecutionProvider、Core 8/10/12/14 配置 | 本仓库 | K3/运行库版本特定 |
| Transpose 等算子固定分工 | 本仓库 | 特定模型与 EP 的实测兼容配置，不是通用 ONNX 内核修复 |
| Python 3.14 / Torch 2.9 的局部 ModernBERT 导入适配 | 本仓库 | 锁定旧版本组合的适配，不宜向通用框架提交无关版本补丁 |
| PyTorch 加载、线程控制、去掉隐藏预热 | 本仓库 | 与当前 K3 eager 后端有关；MLX 的编译预热语义不同 |
| Debian 包、原生 tokenizers wheel、固定模型清单 | 本仓库 | Bianbu RISC-V 分发职责 |

这些项目“暂不提交上游”是工程边界判断，不是已被维护者拒绝。若后续厂商运行库修复相关算子，可在新版本矩阵中复验后减少 CPU 分工；不能未经测试直接删掉禁用列表。

## 维护与更新

每次准备新 Release：

1. 检查游戏上游新提交和 PR 状态；只移植与现有功能有关的改动。
2. 对比模型卡/权重修订与 Bianbu EP/TCM 版本，不自动追踪 `latest`。
3. 修改模型时更新 URL 和哈希，重新生成参考对照；保留旧版本结果。
4. 运行通用测试、K3 121 输入兼容测试、真实 60 步与长蛇回归、终端操作检查。
5. 新包通过后发布新的 tag 和 SHA256SUMS；不覆盖旧发布资产。

仓库 CI 持续执行通用测试；硬件测试明确需要 K3，不能被 CI 的绿色状态替代。新的上游/依赖版本仅产生待评估变更，不自动进入用户安装路径。
