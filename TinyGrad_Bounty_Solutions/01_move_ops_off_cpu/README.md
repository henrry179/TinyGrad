# 任务01: 将topk等操作从CPU迁移

## 任务描述
在torch_backend中将topk, _index_put_impl_, index_tensor, masked_select, 和randperm_generator操作从CPU上移走。

## 奖金
$200

## 当前状态
待解决

最后更新时间: 2025-09-08 17:39:00

## 解决方案思路
1. 分析这些操作在CPU上的实现
2. 确定迁移到GPU或其他设备的可行性
3. 实现设备无关的接口
4. 添加相应的设备特定实现

## 相关文件
- `tinygrad/runtime/ops_torch.py`
- `tinygrad/runtime/ops_gpu.py`

## 测试要求
- 验证迁移后功能正确性
- 性能对比测试
- 跨设备兼容性测试
