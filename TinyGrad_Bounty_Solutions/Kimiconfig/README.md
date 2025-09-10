# Cursor IDE 中 Kimi API 配置指南

## 🎯 快速开始

你的 Kimi API 已经在 Cursor 中配置完成！现在可以直接使用。

### 1. 基本使用

```python
# 在 Cursor 中新建 Python 文件并运行
from kimi_config import chat_with_kimi

# 简单对话
response = chat_with_kimi("你好，请介绍一下你自己")
print(response)

# 代码解释
code = "def hello(): return 'world'"
explanation = chat_with_kimi(f"解释这段代码: {code}")
print(explanation)
```

### 2. 高级功能

```python
# 流式输出
from kimi_config import chat_with_kimi_stream

for chunk in chat_with_kimi_stream("解释什么是机器学习"):
    print(chunk, end="", flush=True)
```

### 3. Cursor 集成助手

```python
from cursor_integration import kimi_helper

# 代码解释
result = kimi_helper.explain_code("your_code_here")
print(result)

# 代码调试
result = kimi_helper.debug_code("buggy_code", "error_message")
print(result)

# 代码重构
result = kimi_helper.refactor_code("old_code")
print(result)
```

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `kimi_config.py` | 主要的API配置文件和函数 |
| `cursor_integration.py` | Cursor专用集成助手类 |
| `cursor_test.py` | 完整的测试和交互界面 |
| `quick_test.py` | 快速配置验证 |
| `model_test.py` | 模型可用性测试 |
| `cursor_kimi_config.json` | Cursor配置文件 |
| `.cursor.env` | 环境变量配置 |
| `CURSOR_SETUP.md` | 详细配置指南 |

## ⚙️ 配置状态

✅ **已完成配置:**
- API Key: 已设置
- Base URL: https://api.moonshot.cn/v1
- 默认模型: moonshot-v1-8k (已测试可用)
- 错误处理: 已添加速率限制和异常处理

## 🚀 在 Cursor 中的使用

### 方法一：直接运行脚本

1. 打开 Cursor
2. 在终端中运行: `py cursor_test.py`
3. 选择需要的功能

### 方法二：在代码中使用

1. 在任何 Python 文件中导入配置
2. 调用相关函数
3. 查看结果

### 方法三：环境变量配置

1. 将 `.cursor.env` 重命名为 `.env`
2. 确保在项目根目录
3. Cursor 会自动加载

## 💡 使用技巧

1. **避免频繁请求**: API 有速率限制，请注意调用频率
2. **错误处理**: 所有函数都包含完善的错误处理
3. **流式输出**: 对于长文本建议使用流式函数
4. **参数调整**: 可以调整 temperature 和 max_tokens 参数

## 🐛 故障排除

### 常见问题

**问题**: 请求过于频繁
**解决**: 等待几秒钟后再试，API 有速率限制

**问题**: 模型不可用
**解决**: 使用 `moonshot-v1-8k` 模型，已确认可用

**问题**: 网络连接问题
**解决**: 检查网络连接和防火墙设置

## 📞 获取帮助

如果遇到问题，请检查：
1. API Key 是否正确
2. 网络连接是否正常
3. 是否超过了速率限制

## 🎉 享受使用！

现在你可以在 Cursor 中畅快地使用 Kimi AI 了！

---

*配置完成时间: 2025年09月08日 17时39分*
*模型版本: moonshot-v1-8k*
*API 状态: ✅ 已验证可用*