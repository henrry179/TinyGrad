# Buffer可视化模块
import weakref
import json
from typing import Dict, List, Any, Optional
from ..runtime.ops_cpu import CPUBuffer

class BufferVisualizer:
    """
    Buffer可视化类，用于跟踪和可视化Buffer的生命周期
    设计为不会阻止Buffer被垃圾回收
    """
    
    def __init__(self):
        # 使用弱引用集合跟踪Buffer，避免影响GC
        self._tracked_buffers = weakref.WeakKeyDictionary()
        self._buffer_events: Dict[int, List[Dict[str, Any]]] = {}
        self._enabled = False
        
    def enable(self):
        """启用可视化"""
        self._enabled = True
        
    def disable(self):
        """禁用可视化"""
        self._enabled = False
        
    def is_enabled(self) -> bool:
        """检查是否启用可视化"""
        return self._enabled
        
    def track_buffer(self, buffer: CPUBuffer, name: str = ""):
        """
        跟踪Buffer，但使用弱引用避免阻止GC
        """
        if not self._enabled:
            return
            
        buffer_id = id(buffer)
        # 使用弱引用跟踪Buffer，不会影响GC
        self._tracked_buffers[buffer] = {
            'name': name,
            'size': buffer.size,
            'dtype': str(buffer.dtype),
            'created_at': self._get_timestamp()
        }
        
        # 记录创建事件
        if buffer_id not in self._buffer_events:
            self._buffer_events[buffer_id] = []
            
        self._buffer_events[buffer_id].append({
            'event': 'created',
            'timestamp': self._get_timestamp(),
            'name': name
        })
        
        # 注册Buffer销毁回调
        weakref.finalize(buffer, self._on_buffer_destroyed, buffer_id)
        
    def _on_buffer_destroyed(self, buffer_id: int):
        """Buffer被销毁时的回调"""
        if buffer_id in self._buffer_events:
            self._buffer_events[buffer_id].append({
                'event': 'destroyed',
                'timestamp': self._get_timestamp()
            })
            
    def _get_timestamp(self) -> float:
        """获取当前时间戳"""
        import time
        return time.time()
        
    def get_buffer_info(self) -> Dict[int, Dict[str, Any]]:
        """
        获取当前跟踪的Buffer信息
        注意：由于使用了弱引用，已经被GC的Buffer不会出现在结果中
        """
        result = {}
        # 由于使用了WeakKeyDictionary，已经被回收的Buffer会自动消失
        for buffer, info in self._tracked_buffers.items():
            buffer_id = id(buffer)
            result[buffer_id] = info.copy()
            if buffer_id in self._buffer_events:
                result[buffer_id]['events'] = self._buffer_events[buffer_id].copy()
        return result
        
    def get_active_buffer_count(self) -> int:
        """获取当前活跃的Buffer数量"""
        return len(self._tracked_buffers)
        
    def get_event_log(self) -> Dict[int, List[Dict[str, Any]]]:
        """获取Buffer事件日志"""
        # 只返回仍然存在的Buffer的事件
        result = {}
        for buffer_id in list(self._buffer_events.keys()):
            # 检查Buffer是否仍然存在
            buffer_exists = any(id(buf) == buffer_id for buf in self._tracked_buffers.keys())
            if buffer_exists or any(event['event'] == 'destroyed' for event in self._buffer_events[buffer_id]):
                result[buffer_id] = self._buffer_events[buffer_id].copy()
        return result
        
    def export_to_json(self) -> str:
        """将可视化数据导出为JSON格式"""
        data = {
            'buffers': self.get_buffer_info(),
            'events': self.get_event_log(),
            'enabled': self._enabled
        }
        return json.dumps(data, indent=2)
        
    def clear(self):
        """清空所有跟踪数据"""
        self._tracked_buffers.clear()
        self._buffer_events.clear()


# 全局Buffer可视化实例
buffer_visualizer = BufferVisualizer()