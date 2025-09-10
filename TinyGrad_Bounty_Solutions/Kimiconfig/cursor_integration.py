#!/usr/bin/env python3
"""
Cursor IDE 集成示例
展示如何在 Cursor 中有效地使用 Kimi API
"""

import time
from kimi_config import chat_with_kimi, chat_with_kimi_stream, DEFAULT_MODEL

class CursorKimiHelper:
    """Cursor 中 Kimi AI 助手类"""

    def __init__(self):
        self.model = DEFAULT_MODEL
        self.last_request_time = 0
        self.min_interval = 1.0  # 最小请求间隔(秒)

    def _rate_limit_wait(self):
        """速率限制等待"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            print(f"⏳ 速率限制: 等待 {wait_time:.1f} 秒...")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def explain_code(self, code, language="python"):
        """解释代码"""
        self._rate_limit_wait()
        prompt = f"""请解释以下{language}代码的功能和原理：

```python
{code}
```

请从以下几个方面进行说明：
1. 这段代码的整体功能
2. 关键步骤和逻辑
3. 可能的优化建议
4. 使用场景"""

        return chat_with_kimi(prompt, temperature=0.3)

    def debug_code(self, code, error_message=""):
        """调试代码"""
        self._rate_limit_wait()
        prompt = f"""请帮助调试以下代码：

```python
{code}
```

错误信息: {error_message}

请提供:
1. 错误原因分析
2. 修复建议
3. 改进后的代码"""

        return chat_with_kimi(prompt, temperature=0.2)

    def refactor_code(self, code, requirements=""):
        """重构代码"""
        self._rate_limit_wait()
        prompt = f"""请重构以下代码使其更加优雅和高效:

```python
{code}
```

要求: {requirements or '提高可读性、性能和维护性'}

请提供重构后的代码和说明。"""

        return chat_with_kimi(prompt, temperature=0.4)

    def generate_tests(self, code, test_type="unit"):
        """生成测试代码"""
        self._rate_limit_wait()
        prompt = f"""请为以下代码生成{test_type}测试:

```python
{code}
```

请提供完整的测试代码，包括:
1. 测试框架使用
2. 测试用例覆盖
3. 边界条件测试"""

        return chat_with_kimi(prompt, temperature=0.3)

    def ask_question(self, question, context=""):
        """问答助手"""
        self._rate_limit_wait()
        if context:
            prompt = f"基于以下上下文回答问题:\n\n上下文: {context}\n\n问题: {question}"
        else:
            prompt = question

        return chat_with_kimi(prompt, temperature=0.7)

    def stream_explanation(self, topic):
        """流式解释"""
        self._rate_limit_wait()
        prompt = f"请详细解释{topic}的概念和应用"

        print(f"🤖 正在解释: {topic}")
        print("-" * 50)

        for chunk in chat_with_kimi_stream(prompt):
            print(chunk, end="", flush=True)

        print("\n" + "-" * 50)

# 创建全局助手实例
kimi_helper = CursorKimiHelper()

def demo_explain_code():
    """演示代码解释功能"""
    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""
    print("🔍 代码解释示例:")
    result = kimi_helper.explain_code(sample_code)
    print(result)

def demo_debug_code():
    """演示代码调试功能"""
    buggy_code = """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
"""
    print("\n🐛 代码调试示例:")
    result = kimi_helper.debug_code(buggy_code, "ZeroDivisionError: division by zero")
    print(result)

def demo_stream():
    """演示流式输出"""
    print("\n🌊 流式输出示例:")
    kimi_helper.stream_explanation("机器学习")

def main():
    """主演示函数"""
    print("🚀 Cursor Kimi 集成演示")
    print("=" * 50)

    try:
        # 演示不同功能
        demo_explain_code()
        print("\n" + "="*50)

        demo_debug_code()
        print("\n" + "="*50)

        # 流式演示（可选，避免速率限制）
        user_input = input("\n是否演示流式输出？(y/N): ").strip().lower()
        if user_input == 'y':
            demo_stream()

        print("\n🎉 演示完成！")
        print("你可以在 Cursor 中使用这些功能来:")
        print("• 解释代码: kimi_helper.explain_code(code)")
        print("• 调试代码: kimi_helper.debug_code(code, error)")
        print("• 重构代码: kimi_helper.refactor_code(code)")
        print("• 生成测试: kimi_helper.generate_tests(code)")
        print("• 问答助手: kimi_helper.ask_question(question)")

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")

if __name__ == "__main__":
    main()
