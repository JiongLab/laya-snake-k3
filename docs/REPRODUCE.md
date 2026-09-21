# 从下载模型到 K3 玩贪吃蛇：完整复现指南

## 1. 复现什么，不复现什么

目标是在一台 SpaceMIT K3 上，从公开来源安装 Laya，运行离线贪吃蛇，并核对推理结果、AI Core 使用情况和延迟。采用 **Bianbu 4.0.5 / riscv64 / Python 3.14 / 8 GB 内存**；其他系统版本尚未验证。

这不是视觉识别或强化学习训练：游戏规则层描述“碰撞、是否安全、食物方向”等选项，Laya 对选项分类。上游环路保护可覆盖不安全的首选，界面明确显示 `SHIELD`。没有训练/改写权重，没有远程 API，没有 Hermes 依赖。

建议先读完系统版本与磁盘要求。安装阶段需要网络和 sudo；初始化与游戏用普通桌面用户运行。模型下载后运行离线。至少预留 5 GB 磁盘；源码构建另需 Rust 工具链空间。

## 2. 固定来源与版本

| 层 | 固定来源/版本 | 用途 |
|---|---|---|
| 原游戏 | `mizorewww/laya-mlx`，`fc1df62828a3fedf4d8229fdac1cbd85f1cdf337` | 游戏、界面、保护层 |
| 本适配 | `JiongLab/laya-snake-k3`，Release `v0.1.0` | K3 后端、安装、测试 |
| SDK | PyPI `laya==0.3.4` | 原始输入构造、温度与概率处理 |
| SDK 权重/配置 | HF `convaiinnovations/laya`，`1c5edc17a7acd8701df6fc341c0d179f1c62c982` 的 `multilingual/` | 多语言模型；也用于 CPU 基线 |
| 社区 ONNX | HF `mizchi/laya-multilingual-onnx`，`d9d003d543e63d6d3375c21d44624136bd1e0bad` | FP16 权重、FP32 决策尾部 |
| Transformers / tokenizers | `5.0.0` / `0.22.2` | 模型结构与分词 |
| PyTorch | 系统包 `2.9.1+dfsg-1~exp1ubuntu2`，Python 中为 `2.9.1+debian` | SDK 兼容及 CPU 基线 |
| SpaceMIT EP / ORT | `2.0.6` / `1.24.2+spacemit.a1` | AI Core 执行 |
| TCM | `3.0.0+5` | 官方计算运行库 |

ONNX 是社区独立转换，不是 Convai 官方 ONNX 发布。它的模型卡记录了上游转换链；本项目以文件哈希与实测输出验证兼容，不将“同名模型”当成权重完全相同的证明。[社区模型卡](https://huggingface.co/mizchi/laya-multilingual-onnx/blob/d9d003d543e63d6d3375c21d44624136bd1e0bad/README.md)

每个下载文件的固定 URL 和 SHA-256 均在仓库根目录 `models.lock.json`。不要把 `main`/`latest` 替换进复现命令。

## 3. 准备系统依赖

在 K3 上核对：

```sh
uname -m
cat /etc/os-release
python3 --version
free -h
df -h .
```

预期架构为 `riscv64`，系统为 Bianbu 4.0.5，Python 为 3.14.x。使用 Bianbu 自带的官方软件源；不要套用普通 Ubuntu x86/ARM 源。实际验证设备使用 `archive.spacemit.com/bianbu4` 的 `resolute` 和 `resolute-porting` 软件包。

```sh
sudo apt update
apt-cache policy python3-spacemit-ort spacemit-onnxruntime spacemit-tcm python3-torch
sudo apt install \
  python3-venv curl \
  'python3-torch=2.9.1+dfsg-1~exp1ubuntu2' \
  'python3-numpy=1:2.3.5+ds-3ubuntu1' \
  'python3-safetensors=0.6.2-1.1' \
  'python3-spacemit-ort=2.0.6' \
  'spacemit-onnxruntime=2.0.6' \
  'spacemit-tcm=3.0.0+5' \
  python3-rich python3-regex python3-yaml python3-requests \
  python3-filelock python3-fsspec python3-packaging python3-certifi \
  python3-idna python3-pil qterminal
```

若锁定版本不可用，先停止，不改装最新版本来冒充同一复现环境。本发布不替换系统运行库，也不执行系统降级。已有较新版本时应使用对应版本的独立验证环境，不能假定沿用这里的数值结果。

## 4. 安装应用包

从 [v0.1.0 发布页](https://github.com/JiongLab/laya-snake-k3/releases/tag/v0.1.0) 下载：

- `laya-snake-k3_0.1.0_riscv64.deb`
- `SHA256SUMS`

在保存这两个文件的目录执行：

```sh
sha256sum --ignore-missing -c SHA256SUMS
sudo apt install ./laya-snake-k3_0.1.0_riscv64.deb
laya-snake-k3-setup
```

校验输出必须明确包含 `.deb: OK`。安装包包含应用源码和锁定的 Python wheels，不包含系统库和模型；安装包本身不会以 root 身份下载模型。`setup` 才以当前用户身份创建 `~/.local/share/laya-snake-k3/venv`，从包内 wheels 安装依赖并下载模型。RISC-V 原生 tokenizers wheel 已包含，不要求体验者先编译 Rust。

默认数据目录为 `${XDG_DATA_HOME:-$HOME/.local/share}/laya-snake-k3`。希望放到另一块磁盘时，初始化和启动都使用相同变量：

```sh
export LAYA_K3_HOME="$HOME/laya-data"
laya-snake-k3-setup
laya-snake-k3
```

已有模型可离线导入；来源目录必须包含 `models.lock.json` 所列的相对目录结构：

```sh
laya-snake-k3-setup --model-source /path/to/already-downloaded/models
```

导入是复制并核对哈希，不引用其他用户的 Python 环境。损坏文件不会被静默接受；下载中断的 `.part` 可继续下载。已存在但校验失败的目标文件需要自行移走后重试，程序不会直接覆盖。

## 5. 单独下载/校验模型

软件包安装后可以单独执行：

```sh
python3 /usr/share/laya-snake-k3/scripts/models.py \
  --directory "${LAYA_K3_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/laya-snake-k3}/models"
```

增加 `--verify-only` 则完全不联网，只核对现有文件。模型布局：

```text
models/
  multilingual/
    model.safetensors
    rl_agent_config.json
    encoder/config.json
    tokenizer/tokenizer.json
    tokenizer/tokenizer_config.json
  multilingual-onnx-comparison/
    model.onnx
```

虽然默认走 ONNX，仍保留原 SDK 权重用于 CPU 对照和复现早期运行链。ONNX 推理会使用 SDK 的分词器、问题构造与概率处理，不能只下载一个 `.onnx` 就运行。

## 6. 运行与控制

```sh
laya-snake-k3
laya-snake-k3 --full-metrics
laya-snake-k3 --backend torch
```

第一条默认：12×8、seed=7、方向单问题、ONNX/SpaceMIT、4 AI 工作线程、完成一次推理走一步。第二条增加两项辅助模型估计，会更慢。第三条使用本地 PyTorch FP32 作为对照，并不是默认加速路径。

终端至少 104 列 × 35 行；空格暂停，R 重开，Q/Ctrl+C 退出。初始化约 25 秒，第一步还会有约 2 秒运行库初始化；后续步长不能与启动耗时混算。默认辅助指标显示 `NOT COMPUTED / FAST MODE`，不是 0 概率或报错。启动时一组算子禁用提示是固定分工配置；游戏中反复初始化则不正常。

也可从应用菜单打开“Laya 贪吃蛇 K3”。自定义 `LAYA_K3_HOME` 时应从已设置该变量的终端启动，系统菜单不会自动继承临时终端变量。

## 7. 可执行验收

记录文件用独占创建，避免覆盖旧证据。下面使用新临时目录：

```sh
result_dir=$(mktemp -d)
laya-snake-k3 --headless --steps 60 --record "$result_dir/early.jsonl"
laya-snake-k3 --headless --initial-length 72 --steps 1200 \
  --record "$result_dir/late.jsonl"
python3 /usr/share/laya-snake-k3/scripts/check-record.py "$result_dir/late.jsonl" --steps 1200
```

预期：推理次数等于步数，零死亡，模型/执行动作与记录一致。该固定场景下长蛇应填满棋盘并自动进入下一局。具体时间见 [验收记录](VALIDATION.md)，不要用其他板卡、后台多实例或首步耗时直接对比稳定阶段均值。

检查线程实际执行核心，不只看主进程所在 CPU：

```sh
pgrep -af 'python -m snake_k3'
ps -L -p <上一步的游戏PID> -o tid,psr,comm
```

AI 工作线程应出现在 8、10、12、14；主线程和 CPU 算子仍可在 0–7。进程整体 `taskset` 到 8–15 不是本项目的配置方法。

源码测试方法见 [BUILD.md](BUILD.md)。硬件测试必须显式启用，通用 CI 通过不等于已经在 K3 实测。

## 8. 排错与卸载

| 现象 | 检查/处理 |
|---|---|
| 缺少环境，提示 setup | 用当前桌面用户运行 `laya-snake-k3-setup`，不要 sudo |
| 找不到指定系统包版本 | 检查 Bianbu 4 软件源；不要混装不同架构或更新 EP 后复用旧基准 |
| SHA256 不匹配 | 检查模型来源和修订；保留/移走损坏文件再重新下载 |
| Python 3.14 上 torch.compile 报错 | 必须使用包内局部兼容适配，不要直接绕过入口导入旧 ModernBERT |
| 泛用 ONNX Runtime 被先导入 | 启动新进程，不在已导入通用 ORT 的 notebook 会话里切换后端 |
| 每步明显变慢 | 检查是否 `--full-metrics`/`--backend torch`、是否有重复模型进程、是否运行多个游戏 |
| 内存不足 | 关闭重复模型实例；完整三问题与其他 AI 应用会增加资源占用 |

```sh
sudo apt remove laya-snake-k3
```

卸载仅删除应用包，不删除用户模型、venv 或记录。需要清理时先确认实际 `LAYA_K3_HOME` 位置，再由用户自行处理；本项目不提供递归删除用户数据的脚本。
