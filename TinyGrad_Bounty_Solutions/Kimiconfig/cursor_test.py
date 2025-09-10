#!/usr/bin/env python3
"""
Cursor IDE 中 Kimi API 测试脚本
专门为 Cursor 环境优化的测试和使用脚本
"""

import os
import sys
from pathlib import Path

# 添加当前目录到 Python 路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from kimi_config import client, DEFAULT_MODEL, chat_with_kimi
except ImportError as e:
    print(f"❌ 导入配置失败: {e}")
    print("请确保 kimi_config.py 文件存在且正确配置")
    sys.exit(1)

def test_kimi_connection():
    """测试 Kimi API 连接"""
    print("🔗 测试 Kimi API 连接...")
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "你好，请确认连接正常。"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        print(f"✅ 连接成功! 回复: {result}")
        return True
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False

def interactive_chat():
    """交互式聊天界面"""
    print("\n🤖 Kimi 交互式聊天 (输入 'exit' 退出)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("👋 再见!")
                break

            if not user_input:
                continue

            print("Kimi: ", end="", flush=True)

            # 使用流式输出
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个专业的AI助手，请用中文回复。"},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

            print()  # 换行

        except KeyboardInterrupt:
            print("\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

def show_menu():
    """显示菜单"""
    print("\n" + "="*50)
    print("🎯 Cursor Kimi API 测试工具")
    print("="*50)
    print("1. 测试API连接")
    print("2. 交互式聊天")
    print("3. 快速测试消息")
    print("4. 查看配置信息")
    print("5. 退出")
    print("="*50)

def quick_test():
    """快速测试"""
    print("\n🧪 快速测试...")

    test_messages = [
        "请介绍一下你自己",
        "解释什么是机器学习",
        "写一个Python Hello World程序"
    ]

    for i, msg in enumerate(test_messages, 1):
        print(f"\n测试 {i}: {msg}")
        try:
            response = chat_with_kimi(msg, model=DEFAULT_MODEL)
            print(f"回复: {response[:100]}{'...' if len(response) > 100 else ''}")
        except Exception as e:
            print(f"❌ 错误: {e}")

def show_config():
    """显示配置信息"""
    print("\n📋 配置信息:")
    print(f"模型: {DEFAULT_MODEL}")
    print(f"API Key: {client.api_key[:20]}..." if client.api_key else "API Key: 未设置")
    print(f"Base URL: {client.base_url}")
    print(f"Python 版本: {sys.version}")

def main():
    """主函数"""
    print("🚀 启动 Cursor Kimi API 测试工具")

    # 首先测试连接
    if not test_kimi_connection():
        print("⚠️  API连接失败，请检查配置")
        return

    while True:
        show_menu()
        try:
            choice = input("请选择 (1-5): ").strip()

            if choice == "1":
                test_kimi_connection()
            elif choice == "2":
                interactive_chat()
            elif choice == "3":
                quick_test()
            elif choice == "4":
                show_config()
            elif choice == "5":
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请重新输入")

        except KeyboardInterrupt:
            print("\n👋 用户中断，再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()
