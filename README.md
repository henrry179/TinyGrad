# TinyGrad 深度学习框架问题解决项目

<p align="center">
  <img src="https://img.shields.io/badge/任务数量-23个-blue.svg" alt="任务数量">
  <img src="https://img.shields.io/badge/总奖金-$11,500-brightgreen.svg" alt="总奖金">
  <img src="https://img.shields.io/badge/任务类型-6种-orange.svg" alt="任务类型">
</p>

本项目旨在解决 [TinyGrad](https://github.com/tinygrad/tinygrad) 深度学习框架的一系列悬赏任务，按照官方悬赏列表逐一攻克技术难题。

## 项目概述

TinyGrad 是一个轻量级的深度学习框架，本项目专注于解决其核心功能和技术难题。通过系统性地解决这些悬赏任务，我们旨在提升 TinyGrad 的性能、兼容性和功能完整性。

<h3>项目统计</h3>

| 统计项 | 数值 |
|--------|------|
| 总任务数量 | 23个 |
| 总奖金金额 | $11,500 |
| 任务类型 | Torch后端优化、性能提升、新功能开发、Bug修复、硬件支持、MLPerf竞赛等 |
| 奖金范围 | $100 - $2,000 |

## 项目结构

```
TinyGrad_Bounty_Solutions/
├── 01_move_ops_off_cpu/           # 将特定操作从CPU迁移到其他设备
├── 02_hlb_cifar10_torch_backend/  # 使hlb-CIFAR10在tiny torch后端工作
├── 03_beautiful_mnist_torch_compile/ # 使beautiful MNIST在torch.compile下工作
├── 04_gpt_fast_amd_backend/       # 使GPT Fast在AMD后端工作
├── 05_multi_gpu_training_torch/   # 多GPU训练在torch后端
├── 06_2x_matching_engine_speed/   # 2倍匹配引擎速度优化
├── 07_4x_matching_engine_speed/   # 4倍匹配引擎速度优化
├── 08_8x_matching_engine_speed/   # 8倍匹配引擎速度优化
├── 09_fpr_nvidia_tensor_cores/    # FPR NVIDIA张量核心优化
├── 10_amd_llvm_bert_mlperf/       # AMD LLVM BERT MLPerf优化
├── 11_z3_index_validation/        # Z3索引验证
├── 12_fast_olmoe_m3_max/          # Fast OLMoE M3 Max优化
├── 13_mlperf_retinanet/           # MLPerf RetinaNet优化
├── 14_webgpu_export_model/        # WebGPU模型导出
├── 15_buffer_gc_cpu_viz/          # 缓冲区GC CPU可视化
├── 16_llama4_scout_tinybox/       # LLaMA4 Scout TinyBox优化
├── 17_multi_machine_training/     # 多机训练
├── 18_rdna4_tensor_cores/         # RDNA4张量核心优化
├── 19_cpugraph_clang_llvm/        # CPU图Clang LLVM优化
├── 20_5090_support_nv_backend/    # 5090支持NV后端
├── 21_fix_avg_pool3d_linearizer/  # 修复AvgPool3D线性化器
├── 22_fix_linearizer_failure_53/  # 修复线性化器失败53
├── 23_fix_metadata_data_parallel_resnet/ # 修复元数据数据并行ResNet
├── CONTRIBUTING.md                 # 贡献指南
├── LICENSE                         # MIT许可证
└── README.md                       # 项目文档
```

<h2>悬赏任务详细列表</h2>

以下是对 TinyGrad 官方悬赏任务的完整列表，按照任务类型和奖金进行了分类整理：

### 🚀 高价值任务 ($1,000-$2,000)

这些任务具有最高的奖金价值，通常涉及重大性能优化或新功能开发。

| 任务编号 | 简短描述 | 类型 | 奖金 |
| :--- | :--- | :--- | :--- |
| 06-08 | 匹配引擎速度优化 (2x/4x/8x提升) | 速度 | $1,000 |
| 14 | WebGPU export_model 实现 | 重构 | $2,000 |
| 13 | 训练 RetinaNet (MLPerf提交可获双倍奖金) | MLPerf | $1,000 |
| 17 | 多机分布式训练扩展性优化 | 功能 | $1,000 |

### 🔧 中高难度任务 ($300-$500)

这些任务通常涉及后端优化或重要功能实现。

| 任务编号 | 简短描述 | 类型 | 奖金 |
| :--- | :--- | :--- | :--- |
| 03 | beautiful_mnist_torch 使用 torch.compile | Torch | $300 |
| 04 | gpt-fast 在 AMD 后端性能优化 | Torch | $400 |
| 05 | 多 GPU 训练在 tiny torch 后端工作 | Torch | $500 |
| 09 | NVIDIA 张量核心 FPR 支持 | 功能 | $300 |
| 10 | AMD LLVM BERT MLPerf 性能优化 | 功能 | $250 |
| 12 | Apple M3 Max OLMoE 性能优化 | 升级 | $300 |
| 16 | Llama 4 Scout 在 tinybox 上性能目标 | 功能 | $500 |

### ⚙️ 中等难度任务 ($150-$300)

这些任务涉及功能实现或兼容性修复。

| 任务编号 | 简短描述 | 类型 | 奖金 |
| :--- | :--- | :--- | :--- |
| 01 | CPU操作迁移到其他设备 | Torch | $200 |
| 02 | hlb-CIFAR10 在 Torch 后端支持 | Torch | $200 |
| 11 | Z3索引验证工具 | 功能 | $300 |
| 15 | CPU缓冲区GC修复 | Bug 修复 | $200 |
| 18 | RDNA4 张量核心优化 | 功能 | $150 |
| 19 | CPU图 Clang/LLVM 兼容性 | 功能 | $200 |
| 20 | RTX 5090 NV 后端支持 | 功能 | $300 |

### 🐛 修复和维护任务 ($100-$200)

这些任务专注于修复现有问题和完善测试。

| 任务编号 | 简短描述 | 类型 | 奖金 |
| :--- | :--- | :--- | :--- |
| 21 | 修复 AvgPool3D 线性化器 | Bug 修复 | $200 |
| 22 | 修复线性化器失败案例53 | Bug 修复 | $100 |
| 23 | 修复数据并行 ResNet 元数据问题 | Bug 修复 | $100 |

## 任务详情

### Torch 后端优化任务 ($200-$500)

- **01**: CPU操作迁移到GPU ($200)
- **02**: CIFAR10在Torch后端支持 ($200)
- **03**: MNIST在torch.compile下工作 ($300)
- **04**: GPT在AMD后端优化 ($400)
- **05**: Torch后端多GPU训练 ($500)

### 性能优化任务 ($250-$1,000)

- **06-08**: 匹配引擎速度优化 (2x/4x/8x提升，每项$1,000)
- **09**: NVIDIA张量核心FPR优化 ($300)
- **10**: AMD LLVM BERT性能优化 ($250)
- **12**: Apple M3 Max OLMoE优化 ($300)
- **16**: TinyBox LLaMA4性能目标 ($500)
- **17**: 多机分布式训练扩展性 ($1,000)

### 硬件支持任务 ($150-$300)

- **09**: NVIDIA张量核心FPR支持 ($300)
- **18**: RDNA4张量核心优化 ($150)
- **19**: CPU图Clang/LLVM兼容性 ($200)
- **20**: RTX 5090 NV后端支持 ($300)

### 新功能开发任务 ($300-$2,000)

- **11**: Z3索引验证工具 ($300)
- **14**: WebGPU模型导出 ($2,000)

### MLPerf 竞赛任务 ($1,000+)

- **13**: RetinaNet训练优化 (MLPerf提交可获双倍奖金)

### Bug 修复任务 ($100-$200)

- **15**: CPU缓冲区GC修复 ($200)
- **21**: 修复AvgPool3D线性化器 ($200)
- **22**: 修复线性化器失败案例53 ($100)
- **23**: 修复数据并行ResNet元数据问题 ($100)

## 🚀 如何开始

### 选择任务

1. **浏览任务列表** - 查看任务列表，选择您感兴趣的领域
2. **阅读详细要求** - 进入对应目录下的`README.md`了解具体要求
3. **评估优先级** - 根据奖金金额和难度评估优先级

### 开发流程

```
graph LR
A[Fork本仓库] --> B[选择任务目录]
B --> C[开展工作]
C --> D[编写测试]
D --> E[提交Pull Request]
```

1. **Fork本仓库**
2. **在对应任务目录中开展工作**
3. **确保代码符合TinyGrad项目的代码规范**
4. **提交Pull Request**

### 代码规范

<div class="highlight">

✅ **必须遵循的规范：**
- 遵循PEP 8 Python代码风格
- 为所有新功能添加测试用例
- 更新相关文档
- 确保向后兼容性

</div>

### 测试要求

所有解决方案必须包含:

- ✅ 单元测试
- ✅ 性能基准测试（如适用）
- ✅ 集成测试（如适用）

## 💰 悬赏奖励

每个任务都有相应的奖金奖励，总计 **$11,500** 的悬赏金额：

```
title 任务奖金分布
"高价值任务" : 5000
"中高难度任务" : 2550
"中等难度任务" : 1600
"修复和维护任务" : 600
"其他" : 1750
```

- **高价值任务**: $1,000-$2,000 (匹配引擎速度提升、分布式训练、WebGPU导出、MLPerf竞赛)
- **中高难度任务**: $300-$500 (Torch后端优化、硬件支持、性能优化)
- **中等难度任务**: $150-$300 (功能实现、兼容性修复)
- **修复和维护任务**: $100-$200 (Bug修复、测试完善)

## 📄 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 👥 贡献者

欢迎所有对TinyGrad感兴趣的开发者参与！无论您是经验丰富的深度学习工程师，还是刚刚入门的学生，我们都欢迎您的贡献。

<a href="https://github.com/henrry179/TinyGrad/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=henrry179/TinyGrad" />
</a>

---

<p align="center">
  <strong>本项目由AI助手自动生成项目结构，所有任务文档均已就绪。您可以立即开始解决感兴趣的任务！</strong>
</p>
