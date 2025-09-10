#!/usr/bin/env python3
"""
Buffer GC功能示例
演示CPU Buffer的垃圾回收和可视化功能
"""

import gc
import time
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinygrad.runtime.ops_cpu import CPUBuffer, cpu_allocator
from tinygrad.viz.buffer_visualizer import buffer_visualizer
from tinygrad.memory.buffer_manager import buffer_manager

def basic_example():
    """基本使用示例"""
    print("=== 基本Buffer使用示例 ===")
    
    # 创建Buffer
    buffer = CPUBuffer(1000)
    print(f"创建了大小为 {buffer.size} 的Buffer")
    
    # 使用Buffer
    data = buffer.data
    data[0] = 1.0
    data[1] = 2.0
    print(f"Buffer数据: {data[:3]}...")
    
    # 删除引用
    del buffer
    print("删除了Buffer引用")
    
    # 强制GC
    gc.collect()
    print("执行了垃圾回收\n")


def allocator_example():
    """分配器使用示例"""
    print("=== Buffer分配器示例 ===")
    
    # 获取统计信息
    initial_stats = cpu_allocator.get_stats()
    print(f"初始统计: {initial_stats}")
    
    # 分配多个Buffer
    buffers = []
    for i in range(5):
        buffer = cpu_allocator.alloc(100)
        buffers.append(buffer)
        print(f"分配了第 {i+1} 个Buffer")
    
    # 检查统计信息
    mid_stats = cpu_allocator.get_stats()
    print(f"分配后统计: {mid_stats}")
    
    # 释放Buffer
    for buffer in buffers:
        cpu_allocator.free(buffer)
        
    # 强制GC
    cpu_allocator.collect()
    
    # 最终统计
    final_stats = cpu_allocator.get_stats()
    print(f"最终统计: {final_stats}")
    print()


def visualization_example():
    """可视化示例"""
    print("=== Buffer可视化示例 ===")
    
    # 启用可视化
    buffer_visualizer.enable()
    print("启用了Buffer可视化")
    
    # 创建并跟踪Buffer
    buffer1 = CPUBuffer(500)
    buffer_visualizer.track_buffer(buffer1, "example_buffer_1")
    
    buffer2 = CPUBuffer(300)
    buffer_visualizer.track_buffer(buffer2, "example_buffer_2")
    
    print(f"当前跟踪的Buffer数量: {buffer_visualizer.get_active_buffer_count()}")
    
    # 查看Buffer信息
    buffer_info = buffer_visualizer.get_buffer_info()
    print(f"Buffer信息: {list(buffer_info.keys())}")
    
    # 删除一个Buffer
    del buffer1
    gc.collect()
    
    print(f"删除一个Buffer后，跟踪的Buffer数量: {buffer_visualizer.get_active_buffer_count()}")
    
    # 导出可视化数据
    viz_data = buffer_visualizer.export_to_json()
    print("可视化数据导出完成")
    
    # 清理
    buffer_visualizer.disable()
    print()


def manager_example():
    """管理器使用示例"""
    print("=== Buffer管理器示例 ===")
    
    # 使用Buffer管理器分配Buffer
    buffer = buffer_manager.allocate_buffer(250, name="managed_buffer")
    print(f"通过管理器分配了Buffer，大小: {buffer.size}")
    
    # 查看统计信息
    stats = buffer_manager.get_memory_stats()
    print(f"内存统计: {stats}")
    
    # 查看生命周期信息
    lifecycles = buffer_manager.get_buffer_lifecycles()
    print(f"Buffer生命周期信息数量: {len(lifecycles)}")
    
    # 删除Buffer
    del buffer
    buffer_manager.force_gc_collect()
    
    # 清理过期元数据
    cleaned = buffer_manager.cleanup_expired_metadata()
    print(f"清理了 {cleaned} 个过期元数据")
    
    # 最终统计
    final_stats = buffer_manager.get_memory_stats()
    print(f"最终统计: {final_stats}")
    print()


def performance_example():
    """性能测试示例"""
    print("=== 性能测试示例 ===")
    
    # 测试大量Buffer分配和回收的性能
    start_time = time.time()
    
    # 分配大量Buffer
    buffers = []
    for i in range(1000):
        buffer = CPUBuffer(10)
        buffers.append(buffer)
    
    alloc_time = time.time() - start_time
    print(f"分配1000个Buffer耗时: {alloc_time:.4f}秒")
    
    # 删除并回收
    del buffers
    gc_start = time.time()
    collected = gc.collect()
    gc_time = time.time() - gc_start
    
    print(f"垃圾回收耗时: {gc_time:.4f}秒，回收了 {collected} 个对象")
    print()


def main():
    """主函数"""
    print("TinyGrad CPU Buffer GC 示例程序\n")
    
    basic_example()
    allocator_example()
    visualization_example()
    manager_example()
    performance_example()
    
    print("所有示例执行完成！")


if __name__ == "__main__":
    main()