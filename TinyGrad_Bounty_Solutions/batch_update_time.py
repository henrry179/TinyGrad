#!/usr/bin/env python3
"""
批量更新项目中所有文件的时间信息
"""

import os
import datetime
import glob

def update_file_time(file_path, old_time, new_time):
    """更新单个文件中的时间信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if old_time in content:
            new_content = content.replace(old_time, new_time)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"❌ 更新文件失败: {file_path} - {e}")
        return False

def main():
    # 获取当前时间
    now = datetime.datetime.now()
    current_time_str = now.strftime('%Y-%m-%d %H:%M:%S')
    current_time_chinese = now.strftime('%Y年%m月%d日 %H时%M分')

    print("🚀 开始批量更新时间信息")
    print(f"当前时间: {current_time_str}")
    print(f"中文格式: {current_time_chinese}")
    print("-" * 50)

    # 需要更新的时间格式
    replacements = [
        ("2025-09-08 17:39:00", current_time_str),
        ("2024年", current_time_chinese),
        ("YYYY-MM-DD", now.strftime('%Y-%m-%d'))
    ]

    # 查找所有需要更新的文件
    file_patterns = [
        "**/*.md",
        "**/*.txt",
        "**/README*",
        "**/*.py"  # 包含可能的注释或文档字符串
    ]

    total_updated = 0
    total_files = 0

    for pattern in file_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            if os.path.isfile(file_path):
                total_files += 1
                file_updated = False

                for old_time, new_time in replacements:
                    if update_file_time(file_path, old_time, new_time):
                        file_updated = True

                if file_updated:
                    total_updated += 1
                    print(f"✅ 已更新: {file_path}")

    print("-" * 50)
    print(f"📊 更新统计:")
    print(f"   总文件数: {total_files}")
    print(f"   已更新文件数: {total_updated}")
    print("🎉 时间更新完成!")

if __name__ == "__main__":
    main()
