#!/usr/bin/env python3
"""
Cursor IDE Kimi 模型设置验证脚本
用于验证 Cursor 中的 Kimi 配置是否正确
"""

import os
import json
import sys
from pathlib import Path

def check_cursor_settings():
    """检查Cursor设置"""
    print("🔍 检查Cursor设置...")

    # 检查项目中的设置文件
    settings_files = [
        ".cursor/settings.json",
        ".vscode/settings.json",
        "cursor_settings.json"
    ]

    for settings_file in settings_files:
        if os.path.exists(settings_file):
            print(f"✅ 找到设置文件: {settings_file}")
            try:
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                print(f"   配置内容: {json.dumps(settings, indent=2, ensure_ascii=False)}")
            except Exception as e:
                print(f"   ❌ 读取失败: {e}")
        else:
            print(f"❌ 未找到设置文件: {settings_file}")

def check_environment_variables():
    """检查环境变量"""
    print("\n🔍 检查环境变量...")

    env_vars = [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "KIMI_API_KEY",
        "KIMI_BASE_URL",
        "KIMI_DEFAULT_MODEL"
    ]

    for var in env_vars:
        value = os.getenv(var)
        if value:
            if "API_KEY" in var:
                print(f"✅ {var}: {value[:20]}...")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: 未设置")

def check_configuration_files():
    """检查配置文件"""
    print("\n🔍 检查配置文件...")

    config_files = [
        "kimi_config.py",
        "cursor_kimi_config.json",
        ".cursor.env",
        ".env"
    ]

    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ 找到配置文件: {config_file}")
            if config_file.endswith('.py'):
                print("   类型: Python配置文件")
            elif config_file.endswith('.json'):
                print("   类型: JSON配置文件")
            elif config_file.endswith('.env'):
                print("   类型: 环境变量文件")
        else:
            print(f"❌ 未找到配置文件: {config_file}")

def generate_cursor_settings():
    """生成Cursor设置文件"""
    print("\n🔧 生成Cursor设置文件...")

    settings = {
        "ai": {
            "enabled": True,
            "provider": "custom",
            "model": "moonshot-v1-8k",
            "apiKey": "sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf",
            "baseUrl": "https://api.moonshot.cn/v1",
            "temperature": 0.7,
            "maxTokens": 2048
        }
    }

    # 创建 .cursor 目录
    cursor_dir = Path(".cursor")
    cursor_dir.mkdir(exist_ok=True)

    # 写入设置文件
    settings_file = cursor_dir / "settings.json"
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    print(f"✅ 已生成设置文件: {settings_file}")
    print("请重启Cursor以应用设置")

def test_kimi_connection():
    """测试Kimi连接"""
    print("\n🧪 测试Kimi连接...")

    try:
        from kimi_config import chat_with_kimi

        response = chat_with_kimi("Hello, this is a test from Cursor setup verification")
        print("✅ 连接成功!")
        print(f"回复: {response[:100]}...")

        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def show_setup_instructions():
    """显示设置说明"""
    print("\n📖 Cursor Kimi 设置说明")
    print("="*50)
    print("请按照以下步骤在Cursor中配置Kimi:")
    print()
    print("方法1: 使用项目设置文件")
    print("1. 复制 cursor_settings.json 到 .cursor/settings.json")
    print("2. 重启Cursor")
    print()
    print("方法2: 使用环境变量")
    print("1. 复制 .cursor.env 为 .env")
    print("2. Cursor会自动加载环境变量")
    print()
    print("方法3: 手动设置")
    print("1. 打开Cursor设置 (Ctrl/Cmd + ,)")
    print("2. 搜索 'AI' 或 'OpenAI'")
    print("3. 配置以下参数:")
    print("   - API Key: sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf")
    print("   - Base URL: https://api.moonshot.cn/v1")
    print("   - Model: moonshot-v1-8k")
    print()
    print("方法4: 使用VS Code兼容设置")
    print("1. 复制 cursor_settings.json 为 .vscode/settings.json")
    print("2. 如果Cursor基于VS Code，这将生效")

def main():
    """主函数"""
    print("🚀 Cursor Kimi 设置验证工具")
    print("="*50)

    # 执行各项检查
    check_configuration_files()
    check_environment_variables()
    check_cursor_settings()

    # 测试连接
    connection_ok = test_kimi_connection()

    # 显示设置说明
    show_setup_instructions()

    # 如果连接失败，提供解决方案
    if not connection_ok:
        print("\n🔧 自动生成Cursor设置文件...")
        generate_cursor_settings()

    print("\n✅ 设置验证完成!")
    print("请按照上述说明配置Cursor，然后重启IDE")

if __name__ == "__main__":
    main()
