# scGPT 环境需求文档

## 1. 概述

本文档详细描述了 scGPT（单细胞生成预训练模型）的软件和硬件环境需求，帮助用户正确配置运行环境。

---

## 2. 系统要求

### 2.1 操作系统

| 操作系统 | 版本要求 | 支持状态 |
| :--- | :--- | :--- |
| Linux | Ubuntu 18.04+ / CentOS 7+ | ✅ 推荐 |
| macOS | macOS 10.15+ | ✅ 支持 |
| Windows | Windows 10/11 | ✅ 支持（Intel XPU推荐） |

### 2.2 硬件要求

#### 最低配置（CPU模式）
- CPU: Intel Core i7 / AMD Ryzen 7 或更高
- 内存: 16GB RAM
- 存储: 10GB 可用空间（用于模型和数据）

#### 推荐配置（GPU模式）
- GPU: NVIDIA CUDA-enabled GPU（如 RTX 3090/4090, A100）
- 显存: 12GB+
- 内存: 32GB+ RAM
- 存储: 50GB+ 可用空间

#### Intel XPU配置（本项目适配）
- XPU设备: Intel Arc GPU / Intel Data Center GPU Max Series
- 显存: 8GB+
- 内存: 32GB+ RAM
- 存储: 50GB+ 可用空间

---

## 3. Python环境

### 3.1 Python版本
- **推荐**: Python 3.8 ~ 3.10
- **支持**: Python ≥ 3.7.12, < 4.0

### 3.2 核心依赖包

| 依赖包 | 最小版本 | 推荐版本 | 说明 |
| :--- | :--- | :--- | :--- |
| pandas | ≥1.3.5 | 1.3.5+ | 数据处理 |
| scvi-tools | ≥0.16.0, <1.0 | 0.16.4+ | 单细胞分析工具 |
| scanpy | ≥1.9.1 | 1.9.1+ | 单细胞数据分析 |
| torch | ≥1.13.0 | 2.0+ | 深度学习框架 |
| torchtext | * | 0.14.0+ | 文本处理（注意：与Intel XPU可能不兼容） |
| numba | ≥0.55.1 | 0.55.2+ | 数值计算优化 |
| scikit-misc | ≥0.1.4 | 0.1.4+ | 科学计算工具 |
| umap-learn | ≥0.5.3 | 0.5.3+ | 降维算法 |
| leidenalg | ≥0.8.10 | 0.8.10+ | 聚类算法 |
| datasets | ≥2.3.0 | 2.3.2+ | HuggingFace数据集 |
| typing-extensions | ≥4.2.0 | 4.2.0+ | 类型提示 |
| scib | ≥1.0.3 | 1.0.3+ | 单细胞整合评估 |
| cell-gears | <0.0.3 | 0.0.2 | 细胞齿轮分析 |
| orbax | <0.1.8 | 0.1.7 | JAX检查点管理 |

### 3.3 开发依赖包

| 依赖包 | 版本 | 说明 |
| :--- | :--- | :--- |
| pytest | ^5.2 | 测试框架 |
| black | ^22.3.0 | 代码格式化 |
| tensorflow | ^2.8.0 | 可选深度学习框架 |
| flash-attn | ^1.0.1 | Flash注意力优化（可选） |
| torch-geometric | ^2.3.0 | 图神经网络 |
| dcor | ~0.5.3 | 距离相关系数 |
| wandb | ^0.12.3 | 实验日志记录 |
| plotly | ^5.3.1 | 可视化工具 |

---

## 4. Intel XPU 特殊配置

### 4.1 PyTorch XPU版本

本项目已适配Intel XPU平台，推荐使用以下配置：

```bash
# Intel专版PyTorch
pip install torch==2.12.0+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

### 4.2 验证XPU环境

```python
import torch

# 检查PyTorch版本
print(f"PyTorch版本: {torch.__version__}")

# 检查XPU可用性
print(f"XPU可用: {hasattr(torch, 'xpu')}")
if hasattr(torch, 'xpu'):
    print(f"XPU设备数: {torch.xpu.device_count()}")
    print(f"XPU是否可用: {torch.xpu.is_available()}")
```

### 4.3 环境变量设置（Linux）

```bash
export SYCL_PI_LEVEL_ZERO_ENABLE_IMMEDIATE_COMMANDLIST=1
export SYCL_PI_LEVEL_ZERO_GPU_DEVICE_TIER=1
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:$LD_LIBRARY_PATH
```

---

## 5. 安装步骤

### 5.1 使用pip安装（推荐）

```bash
# 基础安装
pip install scgpt

# 安装带Flash Attention支持（需要GPU）
pip install scgpt "flash-attn<1.0.5"

# 如果遇到orbax版本问题
pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
```

### 5.2 从源码安装

```bash
# 克隆仓库
git clone https://github.com/bowang-lab/scGPT.git
cd scGPT

# 安装依赖
pip install -e .

# 安装额外依赖
pip install flash-attn wandb
```

### 5.3 Intel XPU环境安装

```bash
# 安装Intel专版PyTorch
pip install torch==2.12.0+xpu -f https://developer.intel.com/ipex-whl-stable-xpu

# 安装IPEX（可选，增强XPU性能）
pip install intel-extension-for-pytorch==2.12.0+xpu -f https://developer.intel.com/ipex-whl-stable-xpu

# 安装scGPT
pip install scgpt
```

---

## 6. 环境验证

### 6.1 基础验证

```python
# 验证核心依赖
import pandas as pd
import scanpy as sc
import torch
import scvi

print("✅ pandas版本:", pd.__version__)
print("✅ scanpy版本:", sc.__version__)
print("✅ torch版本:", torch.__version__)
print("✅ scvi版本:", scvi.__version__)
```

### 6.2 XPU验证

> **注意**: 在Intel XPU环境中，由于torchtext与XPU版PyTorch存在兼容性问题，直接导入`scgpt.utils`可能会失败。请使用以下方法绕过：

```python
import torch
import importlib.util

# 方法1: 直接导入device_utils模块（绕过torchtext问题）
spec = importlib.util.spec_from_file_location("device_utils", "scgpt/utils/device_utils.py")
device_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device_utils)

get_device = device_utils.get_device
is_xpu_available = device_utils.is_xpu_available

# 检测设备
device = get_device("auto")
print(f"✅ 自动检测设备: {device}")

# 验证XPU支持
print(f"✅ XPU可用: {is_xpu_available()}")

# 测试张量操作
x = torch.randn(10, 10).to(device)
y = x @ x
print(f"✅ 张量运算成功，结果设备: {y.device}")
```

### 6.3 完整功能测试

```python
import torch
import importlib.util

# 直接导入device_utils模块（绕过torchtext兼容性问题）
spec = importlib.util.spec_from_file_location("device_utils", "scgpt/utils/device_utils.py")
device_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device_utils)

get_device = device_utils.get_device
is_xpu_available = device_utils.is_xpu_available
AutocastConfig = device_utils.AutocastConfig
GradScalerAdapter = device_utils.GradScalerAdapter
optimize_model_for_device = device_utils.optimize_model_for_device

# 1. 设备检测
device = get_device("auto")
print(f"1. 设备检测: {device}")

# 2. 自动精度配置
autocast = AutocastConfig(device, enabled=True)
print(f"2. 自动精度配置: 已启用")

# 3. 梯度缩放器
scaler = GradScalerAdapter(device, enabled=True)
print(f"3. 梯度缩放器: {scaler.is_enabled()}")

# 4. 模型优化
model = torch.nn.Linear(10, 5).to(device)
optimized_model = optimize_model_for_device(model, device)
print(f"4. 模型优化: 成功")

# 5. 训练测试
optimizer = torch.optim.Adam(model.parameters())
x = torch.randn(4, 10).to(device)
y = model(x)
loss = y.sum()
loss.backward()
optimizer.step()
print(f"5. 训练循环: 成功")

print("\n🎉 所有环境验证通过！")
```

---

## 7. 常见问题

### 7.1 torchtext与Intel XPU兼容性问题

**问题**: 导入scGPT时出现torchtext原生扩展错误

**解决方案**:
```python
# 方法1: 直接导入device_utils模块
import importlib.util
spec = importlib.util.spec_from_file_location("device_utils", "scgpt/utils/device_utils.py")
device_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device_utils)

# 方法2: 安装兼容版本
pip install torchtext==0.16.0 --no-deps
```

### 7.2 Flash Attention安装失败

**问题**: 安装flash-attn时出现编译错误

**解决方案**:
```bash
# 确保安装了CUDA工具包
nvcc --version

# 使用预编译版本
pip install flash-attn --no-build-isolation

# 或跳过Flash Attention（使用标准PyTorch Attention）
pip install scgpt
```

### 7.3 XPU设备检测失败

**问题**: `is_xpu_available()` 返回False

**解决方案**:
```bash
# 检查Intel GPU驱动
ls /dev/dri

# 设置环境变量
export ONEAPI_ROOT=/opt/intel/oneapi
source $ONEAPI_ROOT/setvars.sh

# 验证PyTorch XPU版本
python -c "import torch; print(torch.__version__)"
```

---

## 8. 依赖版本矩阵

| Python版本 | PyTorch版本 | CUDA版本 | 推荐配置 |
| :--- | :--- | :--- | :--- |
| 3.7 | 1.13.0 | 11.7 | ⚠️ 不推荐 |
| 3.8 | 2.0.0+ | 11.8+ | ✅ 推荐 |
| 3.9 | 2.0.0+ | 11.8+ | ✅ 推荐 |
| 3.10 | 2.1.0+ | 12.0+ | ✅ 推荐 |
| 3.8-3.10 | 2.12.0+xpu | Intel XPU | ✅ Intel平台推荐 |

---

## 附录：requirements.txt 完整列表

```txt
# 核心依赖
pandas>=1.3.5
scvi-tools>=0.16.0,<1.0
scanpy>=1.9.1
torch>=1.13.0
torchtext>=0.14.0
numba>=0.55.1
scikit-misc>=0.1.4
umap-learn>=0.5.3
leidenalg>=0.8.10
datasets>=2.3.0
typing-extensions>=4.2.0
scib>=1.0.3
cell-gears<0.0.3
orbax<0.1.8

# 可选依赖
flash-attn<1.0.5
wandb>=0.12.3
torch-geometric>=2.3.0
plotly>=5.3.1
```