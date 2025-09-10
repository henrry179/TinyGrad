#!/usr/bin/env python3
"""
将所有Kimi配置文件移动到Kimiconfig文件夹
"""

import os
import shutil
from pathlib import Path

def move_kimi_files():
    """移动所有Kimi相关文件到Kimiconfig文件夹"""

    # 源目录和目标目录
    source_dir = Path("15_buffer_gc_cpu_viz")
    target_dir = Path("Kimiconfig")

    # 确保目标目录存在
    target_dir.mkdir(exist_ok=True)

    # 需要移动的文件列表
    kimi_files = [
        "kimi_config.py",
        "cursor_integration.py",
        "cursor_kimi_config.json",
        "cursor_settings.json",
        "cursor_kimi_setup.py",
        "cursor_test.py",
        "example_usage.py",
        "model_test.py",
        "quick_test.py",
        "test_config.py",
        "CURSOR_KIMI_GUIDE.md",
        "CURSOR_SETUP.md",
        "README.md"
    ]

    print("🚀 开始移动Kimi配置文件...")
    print(f"源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print("-" * 50)

    moved_count = 0
    skipped_count = 0

    for filename in kimi_files:
        source_file = source_dir / filename
        target_file = target_dir / filename

        if source_file.exists():
            try:
                # 复制文件
                shutil.copy2(source_file, target_file)
                print(f"✅ 复制成功: {filename}")
                moved_count += 1
            except Exception as e:
                print(f"❌ 复制失败: {filename} - {e}")
        else:
            print(f"⚠️  文件不存在: {filename}")
            skipped_count += 1

    print("-" * 50)
    print(f"📊 移动统计:")
    print(f"   成功复制: {moved_count} 个文件")
    print(f"   跳过文件: {skipped_count} 个文件")
    print("🎉 Kimi配置文件移动完成!"

if __name__ == "__main__":
    move_kimi_files()