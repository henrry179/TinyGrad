#!/usr/bin/env python3
"""
Kimi API 配置测试脚本
"""

from kimi_config import client, DEFAULT_MODEL

def test_config():
    print("=== Kimi API 配置测试 ===\n")

    # 检查配置
    print("1. 检查配置信息：")
    print(f"   默认模型: {DEFAULT_MODEL}")
    print(f"   API Key 前缀: {client.api_key[:20]}..." if client.api_key else "   API Key: 未设置")
    print(f"   Base URL: {client.base_url}")
    print()

    # 检查API连接（如果API Key已设置）
    if client.api_key and not client.api_key.startswith("sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf"):
        print("2. 测试API连接：")
        try:
            # 发送一个简单的测试请求
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "user", "content": "你好，请回复'API连接成功'"}
                ],
                max_tokens=50
            )
            result = response.choices[0].message.content.strip()
            print(f"   ✅ API连接成功!")
            print(f"   回复: {result}")
        except Exception as e:
            print(f"   ❌ API连接失败: {e}")
    else:
        print("2. API Key检查：")
        print("   ⚠️  请先在 kimi_config.py 中设置正确的 API Key")
        print("   当前使用的是示例API Key，无法进行真实连接测试")

    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_config()
