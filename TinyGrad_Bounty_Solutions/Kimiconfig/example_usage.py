#!/usr/bin/env python3
"""
Kimi API 使用示例
"""

from kimi_config import chat_with_kimi, client, DEFAULT_MODEL

def main():
    print("=== Kimi k2-0905 API 使用示例 ===\n")

    # 示例1：使用封装函数
    print("示例1：使用封装函数")
    message = "你好，请介绍一下你自己。"
    print(f"用户：{message}")
    response = chat_with_kimi(message)
    print(f"Kimi：{response}\n")

    # 示例2：直接使用client对象
    print("示例2：直接使用client对象")
    try:
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的AI助手。"},
                {"role": "user", "content": "请解释什么是机器学习？"}
            ],
            temperature=0.3,  # 降低随机性，获得更确定的回答
            max_tokens=1024,
            stream=False
        )
        print(f"用户：请解释什么是机器学习？")
        print(f"Kimi：{response.choices[0].message.content}\n")
    except Exception as e:
        print(f"发生错误：{e}\n")

    # 示例3：流式输出
    print("示例3：流式输出")
    try:
        stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "user", "content": "用几句话描述人工智能的发展历程。"}
            ],
            temperature=0.7,
            max_tokens=512,
            stream=True
        )

        print("用户：用几句话描述人工智能的发展历程。")
        print("Kimi：", end="", flush=True)

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)

        print("\n")
    except Exception as e:
        print(f"发生错误：{e}\n")

if __name__ == "__main__":
    main()
