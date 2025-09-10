#!/usr/bin/env python3
"""
测试不同 Kimi 模型名称
"""

from kimi_config import client, AVAILABLE_MODELS

def test_model(model_name, display_name):
    """测试指定模型"""
    print(f"🧪 测试模型: {display_name} ({model_name})")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "你好，请回复'测试成功'"}
            ],
            max_tokens=50,
            temperature=0.3
        )
        result = response.choices[0].message.content.strip()
        print(f"✅ 成功: {result}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False

def main():
    print("🔍 测试不同 Kimi 模型名称")
    print("=" * 50)

    working_models = []

    for display_name, model_name in AVAILABLE_MODELS.items():
        success = test_model(model_name, display_name)
        if success:
            working_models.append((display_name, model_name))
        print()

    print("📊 测试结果:")
    print(f"总共测试了 {len(AVAILABLE_MODELS)} 个模型")
    print(f"成功 {len(working_models)} 个模型")

    if working_models:
        print("\n✅ 可用的模型:")
        for display_name, model_name in working_models:
            print(f"  - {display_name}: {model_name}")
    else:
        print("\n❌ 没有找到可用的模型")
        print("建议:")
        print("1. 检查 API Key 是否正确")
        print("2. 确认账户是否有访问权限")
        print("3. 查看 Kimi 官方文档获取最新模型名称")

if __name__ == "__main__":
    main()
