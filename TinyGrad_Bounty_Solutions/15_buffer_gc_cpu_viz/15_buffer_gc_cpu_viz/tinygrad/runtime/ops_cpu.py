# CPU Buffer操作和内存管理实现
import ctypes
import numpy as np
import weakref
from typing import Optional, Dict, Any
import gc

class CPUBuffer:
    """
    CPU端Buffer实现，支持自动垃圾回收
    """
    def __init__(self, size: int, dtype=np.float32):
        self.size = size
        self.dtype = dtype
        self._data = np.empty(size, dtype=dtype)
        # 使用弱引用回调跟踪对象销毁
        self._finalizer = weakref.finalize(self, self._cleanup, id(self))
        # 用于可视化追踪的弱引用
        self._viz_refs = weakref.WeakSet()
        
    def _cleanup(self, obj_id):
        """Buffer被销毁时的清理回调"""
        # 实际的清理工作
        self._data = None
        # 可以在这里添加日志或其他清理工作
        
    def as_buffer(self) -> memoryview:
        """返回Buffer的memoryview视图"""
        return memoryview(self._data)
        
    def copyin(self, src: memoryview):
        """将数据拷贝到Buffer中"""
        np.copyto(self._data, np.frombuffer(src, dtype=self.dtype))
        
    def copyout(self, dest: memoryview):
        """从Buffer中拷贝数据到目标内存"""
        np.copyto(np.frombuffer(dest, dtype=self.dtype), self._data)
        
    def add_to_visualization(self, viz_component):
        """将Buffer添加到可视化组件中，使用弱引用避免影响GC"""
        self._viz_refs.add(viz_component)
        
    @property
    def data(self):
        """获取数据访问"""
        return self._data
        
    def __del__(self):
        """析构函数"""
        # 确保数据被清理
        if hasattr(self, '_data') and self._data is not None:
            self._data = None


class CPUAllocator:
    """
    CPU内存分配器，实现内存池和垃圾回收优化
    """
    def __init__(self):
        self._buffer_cache: Dict[tuple, weakref.WeakValueDictionary] = {}
        self._active_buffers = weakref.WeakSet()
        self._stats = {
            'allocated': 0,
            'reused': 0,
            'freed': 0
        }
        
    def _get_cache_key(self, size: int, dtype) -> tuple:
        """生成Buffer缓存键"""
        return (size, str(dtype))
        
    def alloc(self, size: int, dtype=np.float32) -> CPUBuffer:
        """分配Buffer"""
        cache_key = self._get_cache_key(size, dtype)
        
        # 初始化缓存字典
        if cache_key not in self._buffer_cache:
            self._buffer_cache[cache_key] = weakref.WeakValueDictionary()
            
        # 查找可用的缓存Buffer
        buffer_cache = self._buffer_cache[cache_key]
        for buffer_id, buffer in list(buffer_cache.items()):
            if buffer is not None:
                self._stats['reused'] += 1
                self._active_buffers.add(buffer)
                return buffer
                
        # 创建新Buffer
        buffer = CPUBuffer(size, dtype)
        buffer_cache[id(buffer)] = buffer
        self._active_buffers.add(buffer)
        self._stats['allocated'] += 1
        return buffer
        
    def free(self, buffer: CPUBuffer):
        """释放Buffer（在实际应用中可能由GC自动处理）"""
        self._active_buffers.discard(buffer)
        self._stats['freed'] += 1
        
    def collect(self):
        """强制进行垃圾回收"""
        gc.collect()
        
    def get_stats(self) -> Dict[str, int]:
        """获取内存分配统计信息"""
        return self._stats.copy()
        
    def get_memory_usage(self) -> int:
        """获取当前活跃Buffer的数量"""
        return len(self._active_buffers)


# 全局CPU分配器实例
cpu_allocator = CPUAllocator()