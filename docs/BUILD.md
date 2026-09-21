# 从源码构建与测试

## 普通开发与通用测试

```sh
git clone https://github.com/JiongLab/laya-snake-k3.git
cd laya-snake-k3
git checkout v0.1.0
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest -q
```

通用测试无需权重，也不会启动 K3 模型；硬件测试显式跳过。包含游戏规则、概率/保护语义、未计算指标、记录回放、用户路径与文件哈希检查。仓库 CI 持续运行同一套通用测试。

## K3 源码运行

先按 [REPRODUCE.md](REPRODUCE.md) 安装锁定的系统依赖。发布源码附带的运行时 wheels 需要单独获取：从 Release 下载 `laya-snake-k3-runtime-wheels-0.1.0.tar.gz`，核对 SHA256SUMS 后，在仓库根目录解压。

```sh
tar -xzf laya-snake-k3-runtime-wheels-0.1.0.tar.gz
sh setup.sh
sh start-snake.sh
```

wheel 压缩包只包含 `wheelhouse/` 下的锁定 Python wheels；模型依旧单独下载。不要用开发机的 `.venv` 复制到 RISC-V 板卡。

## 不使用预构建 tokenizers wheel

本项目 wheel 来源于 tokenizers 0.22.2 的 K3 原生编译，没有改源码。下面是重新构建路径，不保证生成文件逐字节等于 Release 中的 wheel：

```sh
sudo apt install build-essential pkg-config libssl-dev rustc cargo python3-dev python3-venv
python3 -m venv build-env
build-env/bin/python -m pip install 'maturin==1.9.4'
build-env/bin/python -m pip download --no-deps --no-binary=:all: \
  'tokenizers==0.22.2' --dest sources
tar -xzf sources/tokenizers-0.22.2.tar.gz -C sources
build-env/bin/python -m pip wheel --no-deps --no-build-isolation \
  sources/tokenizers-0.22.2 --wheel-dir wheelhouse
```

Rust/Cargo 需能访问其依赖源。原生编译只在构建阶段发生；最终体验者安装 `.deb` 不需要编译器。构建依赖变化可能影响二进制，需在目标系统重新跑硬件测试。

其他 wheels 都是公开 PyPI 的纯 Python 包，下载时限制目标平台，并从刚生成的本地 tokenizers wheel 找到原生依赖：

```sh
build-env/bin/python -m pip download --no-deps --only-binary=:all: \
  --platform linux_riscv64 --python-version 314 --implementation cp \
  --abi cp314 --abi abi3 --find-links wheelhouse --dest wheelhouse \
  -r requirements-runtime.txt
```

期望 16 个固定版本 wheel，不能混入另一版本。`setup.sh` 使用 `--no-deps`，因为数值计算和其他 native 依赖来自已准备的 Bianbu 系统包；随后以实际 SpaceMIT 路径运行 `pip check`，不是忽略依赖冲突。

## 构建 Debian 包

需要 `dpkg-deb`。在 Linux 可安装 `dpkg`；macOS 可用 Homebrew `dpkg` 仅构建/解包，不在 macOS 安装 RISC-V 软件。

```sh
python3 scripts/build-deb.py
dpkg-deb --info dist/laya-snake-k3_0.1.0_riscv64.deb
dpkg-deb --contents dist/laya-snake-k3_0.1.0_riscv64.deb
```

应用位于 `/usr/share/laya-snake-k3`，命令位于 `/usr/bin`，菜单项位于 `/usr/share/applications`。没有 postinst 网络下载、服务启动或用户数据清理动作。`.deb` 架构是 riscv64，因为包内 tokenizers wheel 是原生组件。

## 在新用户目录里验证

```sh
export LAYA_K3_HOME=$(mktemp -d)
sh setup.sh --model-source /path/to/verified/models
LAYA_K3_TESTS=1 "$LAYA_K3_HOME/venv/bin/python" -m pytest -q -s
```

运行硬件测试前另需该 venv 能访问 pytest（Bianbu `python3-pytest` 或在 venv 安装 `pytest==9.1.1`）。设置变量启用的 3 个硬件用例包含快速/完整指标各 60 步以及 121 输入跨长度/批次精度和线程池稳定性。未启用时明确显示 skipped。

不要同时运行多个游戏或模型测试；这台 8 GB K3 的验收是串行测试。完整系统重装不是这个命令的含义：它隔离应用数据和 Python 环境，但仍使用宿主 Bianbu 的系统库。
