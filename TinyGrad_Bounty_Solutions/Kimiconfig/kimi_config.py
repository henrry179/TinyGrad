from openai import OpenAI

# Kimi API 配置
client = OpenAI(
    api_key="sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf",  # Kimi API Key 已配置
    base_url="https://api.moonshot.cn/v1",
)

# 默认模型配置
DEFAULT_MODEL = "moonshot-v1-8k"  # Kimi的通用模型名称

# 可用的模型列表
AVAILABLE_MODELS = {
    "kimi-v1": "moonshot-v1-8k",
    "kimi-v1-32k": "moonshot-v1-32k",
    "kimi-v1-128k": "moonshot-v1-128k",
    "kimi-k2-0905": "kimi-k2-0905",  # 如果API支持的话
}

# 使用示例
def chat_with_kimi(message, model=DEFAULT_MODEL, temperature=0.7, max_tokens=2048):
    """
    与 Kimi 模型对话的示例函数

    Args:
        message (str): 用户输入的消息
        model (str): 使用的模型名称，默认为 moonshot-v1-8k
        temperature (float): 随机性参数 (0.0-2.0)
        max_tokens (int): 最大token数

    Returns:
        str: 模型的回复
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "rate_limit" in error_msg.lower():
            return f"⚠️ 请求过于频繁，请稍后再试。错误详情: {error_msg}"
        elif "not found" in error_msg.lower():
            return f"❌ 模型 '{model}' 不可用。错误详情: {error_msg}"
        else:
            return f"❌ 发生错误: {error_msg}"

def chat_with_kimi_stream(message, model=DEFAULT_MODEL, temperature=0.7, max_tokens=2048):
    """
    流式对话函数

    Args:
        message (str): 用户输入的消息
        model (str): 使用的模型名称
        temperature (float): 随机性参数
        max_tokens (int): 最大token数

    Yields:
        str: 流式输出的内容片段
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"❌ 发生错误: {str(e)}"

# 配置说明：
# 1. API Key 已配置完成
# 2. base_url 已经设置为 Kimi 的专用端点 (https://api.moonshot.cn/v1)
# 3. DEFAULT_MODEL 设置为 moonshot-v1-8k (已测试可用的模型)
# 4. 支持的模型: moonshot-v1-8k, moonshot-v1-32k, moonshot-v1-128k
# 5. chat_with_kimi 函数提供了使用示例，可以直接调用
# 6. chat_with_kimi_stream 函数支持流式输出
# 7. 已添加错误处理和速率限制提示
