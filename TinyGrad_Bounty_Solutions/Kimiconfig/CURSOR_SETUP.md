# Cursor IDE 中 Kimi k2-0905 配置指南

## 📋 概述

本指南将帮助你在 Cursor IDE 中完整配置和使用 Kimi k2-0905 AI模型。

## 🔧 配置步骤

### 1. 环境准备

确保你已经安装了必要的依赖：

```bash
# 安装 OpenAI Python 包
py -m pip install openai
```

### 2. 文件结构

项目中的关键配置文件：

```
├── kimi_config.py          # 主要的API配置文件
├── cursor_kimi_config.json # Cursor专用配置
├── .cursor.env            # 环境变量配置
├── cursor_test.py         # Cursor测试脚本
└── CURSOR_SETUP.md        # 本说明文件
```

### 3. 在Cursor中配置

#### 方法一：使用环境变量文件

1. 将 `.cursor.env` 文件重命名为 `.env`
2. 确保 `.env` 文件位于项目根目录
3. Cursor 会自动加载环境变量

#### 方法二：直接在代码中使用

在你的Python文件中导入配置：

```python
from kimi_config import chat_with_kimi, client, DEFAULT_MODEL

# 简单使用
response = chat_with_kimi("你好！")
print(response)

# 高级使用
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{"role": "user", "content": "解释什么是AI"}],
    temperature=0.7,
    max_tokens=1000
)
```

#### 方法三：Cursor设置面板配置

如果Cursor支持自定义AI提供商配置：

1. 打开 Cursor 设置 (Cmd/Ctrl + ,)
2. 查找 AI 或 OpenAI 相关设置
3. 配置以下参数：
   - **API Key**: `sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf`
   - **Base URL**: `https://api.moonshot.cn/v1`
   - **Model**: `kimi-k2-0905`

## 🧪 测试配置

### 运行完整测试

```bash
# 在Cursor的终端中运行
py cursor_test.py
```

这将启动交互式测试菜单，包括：
- ✅ API连接测试
- 💬 交互式聊天
- 🧪 快速功能测试
- 📋 配置信息查看

### 快速测试

```python
# 在Cursor中新建Python文件并运行
from kimi_config import chat_with_kimi

# 测试基本功能
result = chat_with_kimi("你好，请介绍一下自己")
print(result)
```

## 🎯 在Cursor中的使用场景

### 1. 代码补全和建议

Cursor 会使用配置的Kimi模型来提供智能代码补全。

### 2. AI聊天界面

在Cursor中打开AI聊天面板，与Kimi进行对话。

### 3. 代码解释和重构

选中代码后，可以让Kimi解释或重构代码。

### 4. 调试助手

在调试时获取Kimi的建议和解决方案。

## ⚙️ 高级配置

### 自定义参数

在 `kimi_config.py` 中可以调整：

```python
# 调整温度参数 (0.0-2.0，越高越随机)
temperature=0.7

# 调整最大token数
max_tokens=2048

# 启用流式输出
stream=True
```

### 多模型配置

如果需要使用不同模型，可以扩展配置：

```python
MODELS = {
    "kimi-k2-0905": "kimi-k2-0905",
    "kimi-k1-5": "kimi-k1-5",
    "kimi-k1": "kimi-k1"
}

def chat_with_model(message, model_name="kimi-k2-0905"):
    return chat_with_kimi(message, MODELS.get(model_name, DEFAULT_MODEL))
```

## 🔍 故障排除

### 常见问题

1. **ModuleNotFoundError**
   ```bash
   py -m pip install openai
   ```

2. **API Key错误**
   - 检查 `kimi_config.py` 中的API Key是否正确
   - 确保API Key没有过期

3. **网络连接问题**
   - 检查网络连接
   - 确认防火墙设置

4. **模型不可用**
   - 检查模型名称是否正确
   - 确认账户权限

### 调试模式

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 更多资源

- [Kimi 官方文档](https://platform.moonshot.cn/)
- [Cursor 官方文档](https://cursor.sh/docs)
- [OpenAI Python SDK](https://github.com/openai/openai-python)

## 🚀 快速开始

1. 运行测试：`py cursor_test.py`
2. 选择选项1测试连接
3. 选择选项2开始交互式聊天
4. 在代码中使用：`from kimi_config import chat_with_kimi`

享受在Cursor中使用Kimi AI的强大功能！ 🤖✨
