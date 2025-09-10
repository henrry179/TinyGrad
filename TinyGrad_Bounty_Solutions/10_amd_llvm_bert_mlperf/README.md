# 任务10: AMD LLVM BERT MLPerf优化

## 任务描述
为AMD GPU使用LLVM优化BERT模型的MLPerf性能。

## 奖金
$250

## 当前状态
待解决

最后更新时间: 2025-09-08 17:39:00

## 解决方案思路
1. 分析AMD GPU架构的特性
2. 使用LLVM优化BERT计算图
3. 实现MLPerf特定的优化
4. 验证性能提升

## 相关文件
- `tinygrad/runtime/ops_amd.py`
- `tinygrad/models/bert.py`

## 测试要求
- MLPerf基准测试性能提升
- AMD GPU兼容性验证
- 内存效率优化
