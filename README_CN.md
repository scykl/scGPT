# scGPT - 中文版文档

这是 **scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI** 的官方代码库。

[![预印本](https://img.shields.io/badge/preprint-available-brightgreen)](https://www.biorxiv.org/content/10.1101/2023.04.30.538439) &nbsp;
[![文档](https://img.shields.io/badge/docs-available-brightgreen)](https://scgpt.readthedocs.io/en/latest/) &nbsp;
[![PyPI版本](https://badge.fury.io/py/scgpt.svg)](https://badge.fury.io/py/scgpt) &nbsp;
[![下载量](https://pepy.tech/badge/scgpt)](https://pepy.tech/project/scgpt) &nbsp;
![Web应用](https://img.shields.io/website?url=https%3A%2F%2Fscgpthub.org&up_color=chartreuse%20&logo=gotomeeting&logoColor=%23FFB3FF&label=WebApp&labelColor=%2300CBFF) &nbsp;
[![许可证](https://img.shields.io/badge/license-MIT-blue)](https://github.com/username/repo/blob/main/LICENSE)

**!更新**: 我们发布了多个新的预训练scGPT检查点。请参见 [预训练scGPT检查点](#预训练-scgpt-检查点) 部分了解详细信息。

**[2024.02.26]** 我们在 [integrate-huggingface-model](https://github.com/bowang-lab/scGPT/tree/integrate-huggingface-model) 分支提供了使用HuggingFace运行预训练工作流的初步支持。我们将进行进一步测试并很快合并到主分支。

**[2023.12.31]** 零样本应用的新教程现已推出！请在 [tutorials/zero-shot](tutorials/zero-shot) 目录中找到它们。我们还提供了一个新的持续预训练模型检查点，用于细胞嵌入相关任务。请参阅 [notebook](tutorials/zero-shot/Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb) 了解更多详情。

**[2023.11.07]** 应许多用户要求，我们现在将flash-attention设为可选依赖项。预训练权重可以使用相同的 [load_pretrained](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/utils/util.py#L304) 函数在pytorch CPU、GPU和flash-attn后端上加载，`load_pretrained(target_model, torch.load("path_to_ckpt.pt"))`。示例用法也在 [这里](https://github.com/bowang-lab/scGPT/blob/f6097112fe5175cd4e221890ed2e2b1815f54010/scgpt/tasks/cell_emb.py#L258)。

**[2023.09.05]** 我们发布了一项新功能，用于将样本参考映射到自定义参考数据集或CellXGene收集的数百万个细胞！借助 [faiss](https://github.com/facebookresearch/faiss) 库，我们实现了出色的时间和内存效率。超过3300万个细胞的索引仅占用不到1GB内存，相似性搜索在GPU上处理10,000个查询细胞仅需不到 **1秒**。请参见 [参考映射教程](https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_Reference_Mapping.ipynb) 了解更多详情。

## 在线应用

scGPT现在也可在以下在线应用中使用，您只需浏览器即可开始！

- 使用云GPU运行 [参考映射应用](https://app.superbio.ai/apps/299?id=6548f339a9ed6f6e5560b07d)、[细胞注释应用](https://app.superbio.ai/apps/274?id=64d205cb980ff714de831ee0) 和 [GRN推断应用](https://app.superbio.ai/apps/270?id=64b804fb823bc93b64c10a76)。感谢 [Superbio.ai](https://app.superbio.ai/) 团队帮助创建和托管这些交互式工具。

## 安装

scGPT适用于 Python >= 3.7.13 和 R >= 3.6.1。请确保在安装前安装了正确版本的Python和R。

scGPT已在PyPI上发布。要安装scGPT，请运行以下命令：

```bash
pip install scgpt "flash-attn<1.0.5"  # 可选，推荐
# 截至2023.09，pip安装可能无法与新版本的google orbax包一起运行，如果遇到相关问题，请改用以下命令：
# pip install scgpt "flash-attn<1.0.5" "orbax<0.1.8"
```

[可选] 我们建议使用 [wandb](https://wandb.ai/) 进行日志记录和可视化。

```bash
pip install wandb
```

**注意**: `flash-attn` 依赖通常需要特定的GPU和CUDA版本。如果遇到任何问题，请参阅 [flash-attn](https://github.com/HazyResearch/flash-attention/tree/main) 仓库获取安装说明。目前（2023年5月），由于报告了安装新版本flash-attn的各种问题，我们建议使用CUDA 11.7和flash-attn<1.0.5。

## Intel XPU支持

本项目已适配Intel XPU平台。如果您使用Intel专版PyTorch（如 `torch==2.12.0+xpu`），scGPT会自动检测并使用XPU设备进行加速计算。

### 检查XPU可用性

```python
from scgpt.utils import get_device, is_xpu_available

# 自动检测设备（XPU > CUDA > CPU）
device = get_device("auto")
print(f"使用设备: {device}")

# 检查XPU是否可用
print(f"XPU可用: {is_xpu_available()}")
```

### XPU使用示例

```python
from scgpt.utils import get_device, AutocastConfig, GradScalerAdapter
import torch

# 获取设备
device = get_device("auto")  # 将自动选择XPU（如果可用）

# 创建模型并移动到设备
model = YourModel().to(device)

# 设置混合精度训练（XPU优化）
autocast_config = AutocastConfig(device, enabled=True)
scaler = GradScalerAdapter(device, enabled=True)

# 训练循环
for batch in dataloader:
    inputs = batch.to(device)
    
    with autocast_config.autocast_context():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 预训练scGPT模型库

以下是预训练模型列表。请找到下载检查点文件夹的链接。我们建议默认使用 `whole-human` 模型进行大多数应用。如果您的微调数据集与器官特定模型的训练数据共享相似的细胞类型上下文，这些模型通常也能表现出有竞争力的性能。每个检查点文件夹中提供了一个配对的词汇表文件，将基因名称映射到ID。如果需要ENSEMBL ID，请在 [gene_info.csv](https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv) 中找到转换。

| 模型名称 | 描述 | 下载链接 |
| :--- | :--- | :--- |
| whole-human（推荐） | 在3300万正常人类细胞上预训练 | [链接](https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing) |
| continual pretrained | 用于零样本细胞嵌入相关任务 | [链接](https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing) |
| brain | 在1320万脑细胞上预训练 | [链接](https://drive.google.com/drive/folders/1vf1ijfQSk7rGdDGpBntR5bi5g6gNt-Gx?usp=sharing) |
| blood | 在1030万血液和骨髓细胞上预训练 | [链接](https://drive.google.com/drive/folders/1kkug5C7NjvXIwQGGaGoqXTk_Lb_pDrBU?usp=sharing) |
| heart | 在180万心脏细胞上预训练 | [链接](https://drive.google.com/drive/folders/1GcgXrd7apn6y4Ze_iSCncskX3UsWPY2r?usp=sharing) |
| lung | 在210万肺细胞上预训练 | [链接](https://drive.google.com/drive/folders/16A1DJ30PT6bodt4bWLa4hpS7gbWZQFBG?usp=sharing) |
| kidney | 在81.4万肾细胞上预训练 | [链接](https://drive.google.com/drive/folders/1S-1AR65DF120kNFpEbWCvRHPhpkGK3kK?usp=sharing) |
| pan-cancer | 在570万多种癌症类型细胞上预训练 | [链接](https://drive.google.com/drive/folders/13QzLHilYUd0v3HTwa_9n4G4yEF-hdkqa?usp=sharing) |

## 微调scGPT用于scRNA-seq整合

请参见我们在 [examples/finetune_integration.py](examples/finetune_integration.py) 中的示例代码。默认情况下，脚本假设scGPT检查点文件夹存储在 `examples/save` 目录中。

## 使用示例

### 基础使用

```python
import torch
import scanpy as sc
from scgpt import TransformerModel, CellEmbedding

# 加载数据
adata = sc.read_h5ad("data.h5ad")

# 加载预训练模型
model = TransformerModel.load_pretrained("path/to/checkpoint")

# 获取细胞嵌入
embedding = CellEmbedding(model)
adata.obsm["X_scgpt"] = embedding.get_embeddings(adata)

# 下游分析
sc.pp.neighbors(adata, use_rep="X_scgpt")
sc.tl.umap(adata)
sc.tl.leiden(adata)

# 可视化
sc.pl.umap(adata, color="leiden")
```

### 参考映射

```python
from scgpt.tasks import ReferenceMapping

# 初始化映射器
mapper = ReferenceMapping(
    reference_data="path/to/reference.h5ad",
    model="path/to/scgpt_checkpoint"
)

# 加载查询数据
query_adata = sc.read_h5ad("query.h5ad")

# 执行映射
results = mapper.map(query_adata)

# 获取映射结果
print(results["cell_type_predictions"])
print(results["confidence_scores"])
```

## 待办事项

- [x] 上传预训练模型检查点
- [x] 发布到PyPI
- [ ] 提供带有生成注意力掩码的预训练代码
- [ ] 多组学整合、细胞类型注释、扰动预测、细胞生成的微调示例
- [x] 基因调控网络分析示例代码
- [x] Readthedocs文档网站
- [x] 升级到PyTorch 2.0
- [x] 在更大数据集上进行新的预训练
- [x] 参考映射示例
- [ ] 发布到HuggingFace模型库

## 贡献

我们非常欢迎对scGPT的贡献。如果您有任何想法或bug修复，请提交pull request。我们也欢迎您在使用scGPT时遇到的任何问题。

## 致谢

我们衷心感谢以下开源项目的作者：

- [flash-attention](https://github.com/HazyResearch/flash-attention)
- [scanpy](https://github.com/scverse/scanpy)
- [scvi-tools](https://github.com/scverse/scvi-tools)
- [scib](https://github.com/theislab/scib)
- [datasets](https://github.com/huggingface/datasets)
- [transformers](https://github.com/huggingface/transformers)

## 引用scGPT

```bibtex
@article{cui2023scGPT,
title={scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI},
author={Cui, Haotian and Wang, Chloe and Maan, Hassaan and Pang, Kuan and Luo, Fengning and Wang, Bo},
journal={bioRxiv},
year={2023},
publisher={Cold Spring Harbor Laboratory}
}
```