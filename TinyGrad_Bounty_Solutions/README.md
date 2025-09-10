# TinyGrad 深度学习框架问题解决项目

本项目旨在解决TinyGrad深度学习框架的一系列悬赏任务，按照官方悬赏列表逐一攻克技术难题。

## 项目概述

TinyGrad是一个轻量级的深度学习框架，本项目专注于解决其核心功能和技术难题。通过系统性地解决这些悬赏任务，我们旨在提升TinyGrad的性能、兼容性和功能完整性。

**项目统计：**

- 总任务数量：23个
- 总奖金金额：$11,500
- 任务类型：Torch后端优化、性能提升、新功能开发、Bug修复、硬件支持、MLPerf竞赛等
- 奖金范围：$100 - $2,000

## 项目结构

```bash
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

## 悬赏任务详细列表

以下是TinyGrad官方悬赏任务的完整列表，按照图片中的顺序排列：

| 任务编号 | 简短描述 | 类型 | 奖金/价值 | 详细解释 |
| :--- | :--- | :--- | :--- | :--- |
| 01 | 在 torch_backend 中将 topk, _index_put_impl_, index_tensor, masked_select 和 randperm_generator 操作从 CPU 上移走 | Torch | $200 | 这是关于 TinyGrad 的 Torch 后端功能的优化任务，目标是将这些原本在 CPU 上执行的操作转移到其他设备（如 GPU）上执行，以提升性能。 |
| 02 | hlb-CIFAR10 示例在使用微型的 (tiny) torch 后端时能够工作 | Torch | $200 | 这是一个功能实现任务，确保名为 `hlb-CIFAR10` 的模型训练示例能够在使用轻量级 Torch 后端的情况下正常运行。 |
| 03 | beautiful_mnist_torch 使用 torch.compile，且 TinyJIT 在设置 TINY_BACKEND=1 时能工作（参见 test_complie.py） | Torch | $300 | 这是一个关于编译优化的任务，目标是让 `beautiful_mnist_torch` 示例能利用 PyTorch 的 `torch.compile` 和 TinyGrad 的 TinyJIT 编译器，并在特定后端设置下正常工作。 |
| 04 | 使用我们的 torch 后端（配合编译和 beam 搜索）时，gpt-fast 性能超过 AMD 默认后端 | Torch | $400 | 这是一个性能竞赛任务，目标是让 TinyGrad 的 Torch 后端在运行 `gpt-fast` 时，其速度能够超越在 AMD 默认后端上的运行速度。 |
| 05 | 多 GPU 训练在微型 tiny torch 后端上能够工作 | Torch | $500 | 这是一个功能实现任务，旨在让 TinyGrad 的 Torch 后端支持使用多个 GPU 进行模型训练。 |
| 06 | 匹配引擎速度提升 2 倍（从 400ms 到 200ms）external_benchmark_schedule.py | 速度 | $1,000 | 这是一个性能优化任务，目标是将调度器（或类似组件）的运行速度在基准测试中提升一倍。 |
| 07 | 匹配引擎速度提升 4 倍（从 200ms 到 100ms） | 速度 | $1,000 | 这是上一个速度优化任务的更高目标，要求速度再提升一倍。 |
| 08 | 匹配引擎速度提升 8 倍（从 100ms 到 50ms） | 速度 | $1,000 | 这是速度优化任务的最高目标，要求最终速度达到初始的 8 倍。 |
| 09 | 在 NVIDIA（使用张量核心 Tensor Cores）上支持 FPR（可能指低精度格式，如 FP16） | 功能 | $300 | 这是一个功能支持任务，旨在让 TinyGrad 能够利用 NVIDIA GPU 的张量核心来运行低精度计算（如半精度浮点数）。 |
| 10 | AMD LLVM 在 BERT MLPerf 中内核时间快 5%+（并且我们将其用于提交） | 功能 | $250 | 这是一个性能优化任务，针对 AMD 平台，要求使用 LLVM 编译器后端时，运行 BERT 模型的核心计算时间比基线提升至少 5%，以便用于 MLPerf 竞赛提交。 |
| 11 | 完整的 z3 索引（包括 mask/掩码）验证，通过一个标志控制 | 功能 | $300 | 这是一个调试或验证功能任务，目标是实现一个完整的、基于 z3 求解器的索引验证工具（用于确保张量索引计算的正确性），该功能可以通过开关标志启用。 |
| 12 | 在 tinygrad 中实现快速的 OLMoE（在 M3 Max 上达到理论最大值的 50%+） | 升级 | $300 | 这是一个性能优化任务，要求针对 OLMoE（一种模型）进行优化，使其在 Apple M3 Max 芯片上的运行效率达到硬件理论峰值性能的一半以上。 |
| 13 | (miperf) 训练 RetinaNet（如果进入 MLPerf 则奖金翻倍！） | MLPerf | $1,000 | 这是一个 MLPerf 竞赛任务，目标是实现或优化 RetinaNet 模型的训练，使其符合 MLPerf 标准。成功提交到 MLPerf 可以获得双倍奖金。 |
| 14 | 在 tinygrad 主分支中实现 WebGPU export_model（代码清晰且有文档记录 + 使用文档，$2k 奖金由贡献者自行分配） | 重构 | $2,000 | 这是一个代码重构和功能添加任务，要求实现一个将模型导出为 WebGPU 格式的功能，并且代码质量要高，文档要齐全。奖金较高且可由多位贡献者分享。 |
| 15 | 确保缓冲区在 CPU 上和使用 VIZ 时能被垃圾回收 (GCed)，并有良好的测试 | Bug 修复 | $200 | 这是一个修复内存管理问题的任务，需要确保在 CPU 计算和可能的内存可视化 (VIZ) 调试场景中，临时缓冲区能够被正确释放，防止内存泄漏，并添加测试用例。 |
| 16 | Llama 4 Scout 在 tinybox 上以 100+ tok/s 的速度运行 | 功能 | $500 | 这是一个性能目标任务，要求未来的 Llama 4 Scout 模型能够在被称为 "tinybox" 的硬件上达到每秒生成超过 100 个 token 的推理速度。 |
| 17 | 在 12 个 GPU 上进行多机训练 ResNet 或 BERT，相比 6 个 GPU 至少获得 175% 的速度提升 | 功能 | $1,000 | 这是一个分布式训练功能和性能任务，要求实现或优化多机多 GPU 训练，并且扩展性要好（使用 12 个 GPU 的速度要比使用 6 个 GPU 快 75% 以上）。 |
| 18 | RDNA4 张量核心，修复 gfx1201（9070XT）的张量核心支持 | 功能 | $150 | 这是一个硬件支持任务，针对 AMD 的 RDNA4 架构（特别是 gfx1201，如 RX 9070XT GPU），修复或实现对其张量核心功能的支持。 |
| 19 | CPUGraph 在使用 CLANG+LLVM 时能够工作 | 功能 | $200 | 这是一个编译兼容性任务，确保 CPUGraph 功能在使用 Clang 编译器和 LLVM 后端时能够正常工作。 |
| 20 | 在 NV 后端中支持 5090 | 功能 | $300 | 这是一个硬件支持任务，要求为未来可能发布的 NVIDIA RTX 5090 GPU 添加支持到 TinyGrad 的 NVIDIA 后端中。 |
| 21 | 通过修复线性器中的问题（而不是通过变通方法）来修复 TestOps.test_avg_pool3d_failure | Bug 修复 | $200 | 这是一个修复具体测试失败的任务，要求从根本上修复三维平均池化操作在线性器中的问题，而不是采用临时规避措施。 |
| 22 | 修复 TestLinearizerFailures.test_failure_53 | Bug 修复 | $100 | 这是一个修复具体测试失败的任务，要求修复线性器测试套件中编号为 53 的失败用例。 |
| 23 | 修复 'DEBUG=2 LLVM=1 python3 test/test_multitensor.py TestMultiTensor.test_data_parallel_resnet_train_step' 中的元数据 + 添加测试 | Bug 修复 | $100 | 这是一个修复具体测试失败的任务，要求修复在特定调试和编译设置下数据并行训练 ResNet 时的元数据错误，并为此添加测试用例以防止回归。 |

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

## 如何开始

### 选择任务

1. 查看任务列表，选择您感兴趣的领域
2. 阅读对应目录下的`README.md`了解具体要求
3. 根据奖金金额和难度评估优先级

### 开发流程

1. Fork本仓库
2. 在对应任务目录中开展工作
3. 确保代码符合TinyGrad项目的代码规范
4. 提交Pull Request

### 代码规范

- 遵循PEP 8 Python代码风格
- 为所有新功能添加测试用例
- 更新相关文档
- 确保向后兼容性

### 测试要求

所有解决方案必须包含:

- 单元测试
- 性能基准测试（如适用）
- 集成测试（如适用）

## 悬赏奖励

每个任务都有相应的奖金奖励，总计$11,500的悬赏金额：

- **高价值任务**: $1,000-$2,000 (匹配引擎速度提升、分布式训练、WebGPU导出、MLPerf竞赛)
- **中高难度任务**: $300-$500 (Torch后端优化、硬件支持、性能优化)
- **中等难度任务**: $150-$300 (功能实现、兼容性修复)
- **修复和维护任务**: $100-$200 (Bug修复、测试完善)

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献者

欢迎所有对TinyGrad感兴趣的开发者参与！无论您是经验丰富的深度学习工程师，还是刚刚入门的学生，我们都欢迎您的贡献。

---

_本项目由AI助手自动生成项目结构，所有任务文档均已就绪。您可以立即开始解决感兴趣的任务！_
