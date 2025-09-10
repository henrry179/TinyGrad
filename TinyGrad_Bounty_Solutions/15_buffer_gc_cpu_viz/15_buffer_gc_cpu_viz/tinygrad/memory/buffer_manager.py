# Buffer管理器模块
import weakref
import gc
from typing import Dict, Optional, Any
from ..runtime.ops_cpu import CPUBuffer, cpu_allocator
from ..viz.buffer_visualizer import buffer_visualizer

class BufferManager:
    """
    高级Buffer管理器，协调分配、回收和可视化
    """
    
    def __init__(self):
        self._allocator = cpu_allocator
        self._visualizer = buffer_visualizer
        self._buffer_metadata: Dict[int, Dict[str, Any]] = {}
        
    def allocate_buffer(self, size: int, dtype=None, name: str = "") -> CPUBuffer:
        """
        分配Buffer并可选地进行跟踪
        """
        # 使用CPU分配器分配Buffer
        buffer = self._allocator.alloc(size, dtype)
        
        # 添加元数据
        buffer_id = id(buffer)
        self._buffer_metadata[buffer_id] = {
            'name': name,
            'size': size,
            'dtype': str(dtype) if dtype else "default",
            'allocation_time': self._get_timestamp()
        }
        
        # 如果启用了可视化，则跟踪Buffer
        if self._visualizer.is_enabled():
            self._visualizer.track_buffer(buffer, name)
            
        return buffer
        
    def release_buffer(self, buffer: CPUBuffer):
        """
        释放Buffer
        """
        buffer_id = id(buffer)
        # 从元数据中移除
        if buffer_id in self._buffer_metadata:
            self._buffer_metadata[buffer_id]['release_time'] = self._get_timestamp()
            
        # 让分配器处理释放
        self._allocator.free(buffer)
        
    def force_gc_collect(self):
        """
        强制进行垃圾回收
        """
        # 执行垃圾回收
        collected = gc.collect()
        # 也让分配器执行清理
        self._allocator.collect()
        return collected
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息
        """
        stats = self._allocator.get_stats()
        stats['active_buffers'] = self._allocator.get_memory_usage()
        stats['tracked_metadata'] = len(self._buffer_metadata)
        return stats
        
    def _get_timestamp(self) -> float:
        """
        获取当前时间戳
        """
        import time
        return time.time()
        
    def cleanup_expired_metadata(self):
        """
        清理已过期的元数据（对应已回收的Buffer）
        """
        # 获取当前活跃Buffer的ID集合
        active_buffer_ids = set()
        
        # 通过可视化器获取活跃Buffer（如果启用）
        if self._visualizer.is_enabled():
            for buffer in list(self._visualizer._tracked_buffers.keys()):
                active_buffer_ids.add(id(buffer))
                
        # 清理不在活跃列表中的元数据
        expired_ids = []
        for buffer_id in self._buffer_metadata:
            if buffer_id not in active_buffer_ids:
                expired_ids.append(buffer_id)
                
        for buffer_id in expired_ids:
            del self._buffer_metadata[buffer_id]
            
        return len(expired_ids)
        
    def get_buffer_lifecycles(self) -> Dict[int, Dict[str, Any]]:
        """
        获取Buffer生命周期信息
        """
        lifecycles = {}
        for buffer_id, metadata in self._buffer_metadata.items():
            lifecycle = metadata.copy()
            # 计算生命周期（如果已释放）
            if 'release_time' in metadata and 'allocation_time' in metadata:
                lifecycle['lifetime'] = metadata['release_time'] - metadata['allocation_time']
            lifecycles[buffer_id] = lifecycle
        return lifecycles


# 全局Buffer管理器实例
buffer_manager = BufferManager()