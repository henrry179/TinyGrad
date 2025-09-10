#!/usr/bin/env python3
"""
快速测试 Kimi API 配置
在 Cursor 中运行此脚本快速验证配置是否正确
"""

from kimi_config import chat_with_kimi

def main():
    print("🧪 快速测试 Kimi k2-0905 配置")
    print("=" * 40)

    try:
        # 测试1: 基本连接
        print("测试1: 基本连接...")
        response = chat_with_kimi("请回复'配置成功'来确认连接正常")
        print(f"✅ 成功: {response}")

        # 测试2: 代码相关问题
        print("\n测试2: 代码相关问题...")
        code_response = chat_with_kimi("用Python写一个计算斐波那契数列的函数")
        print(f"✅ 代码回复: {code_response[:200]}...")

        print("\n🎉 所有测试通过！Kimi API 配置成功！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查 API Key 和网络连接")

if __name__ == "__main__":
    main()
