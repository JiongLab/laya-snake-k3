# v0.1.0 — K3 复现包（预发布）

交付中文复现指南、固定模型下载清单、K3 适配源码、Debian 应用包、16 个版本锁定的 Python wheels 以及通用/硬件测试。

上游通用优化已提交：[laya-mlx #4](https://github.com/mizorewww/laya-mlx/pull/4)。未声明已合并。

## 验证状态

- 本地 40 项通过、3 项需要 K3 的测试跳过；包含真实 `.deb` 构建和解包检查。
- K3 新 venv 初始化、依赖检查和 6 个模型文件哈希通过。
- 安装包的新环境游戏验收因设备 SSH 无响应未完成，故标记 prerelease，不作为已完成硬件验收的正式稳定版。
- 历史现场版有完整 39 项回归、121 输入对照及 1,200 步约 110 ms/步的结果；这些不替代新包验收。

## 资产

- `laya-snake-k3_0.1.0_riscv64.deb`：应用、文档、运行时 wheels；不含模型和系统库。
- `laya-snake-k3-runtime-wheels-0.1.0.tar.gz`：源码安装用的相同 wheels。
- `laya-snake-k3-source-0.1.0.tar.gz`：本发布对应源码、文档和测试。
- `SHA256SUMS`：以上资产校验值。

目标环境严格限定为文档中的 Bianbu 4.0.5 K3，不适用于普通 x86/ARM Linux。建议在独立用户数据目录试用，不覆盖现有可用部署。
