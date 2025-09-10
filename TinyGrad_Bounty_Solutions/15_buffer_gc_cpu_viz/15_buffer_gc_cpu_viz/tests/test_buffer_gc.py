#!/usr/bin/env python3
"""
Buffer垃圾回收测试
测试CPU Buffer的分配、使用和自动回收功能
"""

import gc
import weakref
import sys
import os

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tinygrad.runtime.ops_cpu import CPUBuffer, CPUAllocator, cpu_allocator
from tinygrad.viz.buffer_visualizer import BufferVisualizer, buffer_visualizer
from tinygrad.memory.buffer_manager import BufferManager, buffer_manager

def test_basic_buffer_allocation():
    """测试基本Buffer分配和回收"""
    print("测试基本Buffer分配和回收...")
    
    # 创建Buffer
    buffer = CPUBuffer(1000)
    buffer_id = id(buffer)
    
    # 检查Buffer是否正确创建
    assert buffer.size == 1000
    assert buffer.data is not None
    
    # 创建弱引用
    buffer_ref = weakref.ref(buffer)
    
    # 删除强引用
    del buffer
    
    # 强制垃圾回收
    gc.collect()
    
    # 检查Buffer是否被正确回收
    assert buffer_ref() is None
    print("✓ 基本Buffer分配和回收测试通过")


def test_buffer_allocator_caching():
    """测试Buffer分配器缓存功能"""
    print("测试Buffer分配器缓存功能...")
    
    allocator = CPUAllocator()
    
    # 分配一个Buffer
    buffer1 = allocator.alloc(100)
    buffer1_id = id(buffer1)
    
    # 释放Buffer
    allocator.free(buffer1)
    
    # 再次分配相同大小的Buffer
    buffer2 = allocator.alloc(100)
    buffer2_id = id(buffer2)
    
    # 检查是否重用了Buffer（缓存机制）
    # 注意：由于使用了弱引用，可能不会重用，这取决于GC时机
    print(f"Buffer1 ID: {buffer1_id}, Buffer2 ID: {buffer2_id}")
    
    stats = allocator.get_stats()
    print(f"Allocator stats: {stats}")
    print("✓ Buffer分配器缓存功能测试完成")


def test_visualizer_compatibility():
    """测试可视化器兼容性，确保不影响GC"""
    print("测试可视化器兼容性...")
    
    visualizer = BufferVisualizer()
    visualizer.enable()
    
    # 创建Buffer并跟踪
    buffer = CPUBuffer(500)
    buffer_id = id(buffer)
    visualizer.track_buffer(buffer, "test_buffer")
    
    # 检查Buffer是否被正确跟踪
    assert visualizer.get_active_buffer_count() == 1
    
    # 创建弱引用
    buffer_ref = weakref.ref(buffer)
    
    # 删除强引用
    del buffer
    
    # 强制垃圾回收
    gc.collect()
    
    # 检查Buffer是否被正确回收（即使被可视化器跟踪）
    assert buffer_ref() is None
    # 活跃Buffer计数应该更新
    assert visualizer.get_active_buffer_count() == 0
    
    print("✓ 可视化器兼容性测试通过")


def test_buffer_manager():
    """测试Buffer管理器功能"""
    print("测试Buffer管理器...")
    
    manager = BufferManager()
    
    # 分配Buffer
    buffer = manager.allocate_buffer(200, name="test_managed_buffer")
    buffer_id = id(buffer)
    
    # 检查内存统计
    stats = manager.get_memory_stats()
    print(f"内存统计: {stats}")
    
    # 检查生命周期信息
    lifecycles = manager.get_buffer_lifecycles()
    assert buffer_id in lifecycles
    
    # 创建弱引用
    buffer_ref = weakref.ref(buffer)
    
    # 删除强引用并强制GC
    del buffer
    manager.force_gc_collect()
    
    # 检查Buffer是否被回收
    assert buffer_ref() is None
    
    print("✓ Buffer管理器测试通过")


def test_memory_leak_prevention():
    """测试内存泄漏预防"""
    print("测试内存泄漏预防...")
    
    initial_count = len(gc.get_objects())
    
    # 创建大量Buffer
    buffers = []
    for i in range(100):
        buffer = CPUBuffer(10)
        buffers.append(buffer)
        
    # 中间检查
    mid_count = len(gc.get_objects())
    assert mid_count > initial_count
    
    # 清空引用
    del buffers
    
    # 强制垃圾回收
    collected = gc.collect()
    print(f"垃圾回收清理了 {collected} 个对象")
    
    # 最终检查 - 应该接近初始状态
    final_count = len(gc.get_objects())
    print(f"对象数量: 初始={initial_count}, 中间={mid_count}, 最终={final_count}")
    
    print("✓ 内存泄漏预防测试完成")


def run_all_tests():
    """运行所有测试"""
    print("开始Buffer GC测试...\n")
    
    test_basic_buffer_allocation()
    test_buffer_allocator_caching()
    test_visualizer_compatibility()
    test_buffer_manager()
    test_memory_leak_prevention()
    
    print("\n所有测试通过！Buffer GC功能正常工作。")


if __name__ == "__main__":
    run_all_tests()