# 🖥️ Cursor IDE 中配置 Kimi 模型的完整指南

## 🎯 前言

本指南将详细介绍如何在 Cursor IDE 中配置 Kimi AI 模型，支持代码补全、聊天、代码解释等多种功能。

## 📋 配置前的准备

确保你已经：
- ✅ 安装了 Python 和 OpenAI 包
- ✅ 获取了 Kimi API Key: `sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf`
- ✅ 确认模型 `moonshot-v1-8k` 可用

## 🔧 配置方法

### 方法一：通过 Cursor 设置面板（推荐）

#### 步骤：

1. **打开 Cursor 设置**
   - Windows/Linux: `Ctrl + ,`
   - macOS: `Cmd + ,`

2. **搜索 AI 设置**
   - 在搜索框中输入 "AI" 或 "OpenAI"
   - 查找 AI 相关的设置选项

3. **配置参数**
   ```
   AI Provider: Custom / OpenAI Compatible
   API Key: sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf
   Base URL: https://api.moonshot.cn/v1
   Model: moonshot-v1-8k
   Temperature: 0.7
   Max Tokens: 2048
   ```

4. **启用功能**
   - ✅ Enable AI Autocomplete
   - ✅ Enable Inline Suggestions
   - ✅ Enable AI Chat
   - ✅ Enable Code Actions

5. **保存并重启**
   - 保存设置
   - 重启 Cursor IDE

### 方法二：使用项目配置文件

#### 步骤：

1. **创建 `.cursor` 目录**
   ```bash
   mkdir .cursor
   ```

2. **创建设置文件**
   ```json
   // .cursor/settings.json
   {
     "ai": {
       "enabled": true,
       "provider": "custom",
       "model": "moonshot-v1-8k",
       "apiKey": "sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf",
       "baseUrl": "https://api.moonshot.cn/v1",
       "temperature": 0.7,
       "maxTokens": 2048
     }
   }
   ```

3. **或者使用现有的配置文件**
   ```bash
   # 复制已有的设置文件
   cp cursor_settings.json .cursor/settings.json
   ```

4. **重启 Cursor**

### 方法三：环境变量配置

#### 步骤：

1. **创建环境文件**
   ```bash
   # 将 .cursor.env 重命名为 .env
   mv .cursor.env .env
   ```

2. **确保 `.env` 文件内容**
   ```env
   OPENAI_API_KEY=sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf
   OPENAI_BASE_URL=https://api.moonshot.cn/v1
   OPENAI_MODEL=moonshot-v1-8k
   ```

3. **Cursor 会自动加载**
   - Cursor 通常会自动检测项目根目录的 `.env` 文件

### 方法四：VS Code 兼容设置（如果适用）

如果你的 Cursor 基于 VS Code：

1. **创建 VS Code 设置**
   ```bash
   mkdir .vscode
   cp cursor_settings.json .vscode/settings.json
   ```

2. **或者直接编辑**
   ```json
   // .vscode/settings.json
   {
     "openai.apiKey": "sk-pxLonvrL2WUK5O1IaEbh2x9SkFdcGP3TFRK5yhaQwCV1KGMf",
     "openai.baseUrl": "https://api.moonshot.cn/v1",
     "openai.model": "moonshot-v1-8k"
   }
   ```

## 🧪 验证配置

### 使用验证脚本

```bash
# 运行设置验证脚本
py cursor_kimi_setup.py
```

### 手动验证

1. **打开 AI 聊天面板**
   - 在 Cursor 中查找 AI Chat 或 Chat 图标
   - 发送一条测试消息: "你好，请确认连接正常"

2. **测试代码补全**
   - 创建一个新的 Python 文件
   - 键入代码，查看是否有 AI 建议

3. **测试代码解释**
   - 选中一段代码
   - 右键选择 "AI: Explain" 或类似选项

## 🔍 故障排除

### 常见问题及解决方案

#### 问题1: Cursor 中找不到 AI 设置
**解决方案:**
- 更新到最新版本的 Cursor
- 检查是否有 AI 相关插件需要安装
- 尝试方法2（配置文件）或方法3（环境变量）

#### 问题2: API 连接失败
**解决方案:**
- 确认 API Key 正确
- 检查网络连接
- 验证 Base URL: `https://api.moonshot.cn/v1`
- 确认模型名称: `moonshot-v1-8k`

#### 问题3: 速率限制错误
**解决方案:**
- 减少请求频率
- 等待几秒后再试
- Kimi API 每分钟限制3次请求

#### 问题4: 设置不生效
**解决方案:**
- 重启 Cursor IDE
- 检查设置文件语法是否正确
- 确认文件位置正确

## 🎯 功能使用指南

### 1. AI 聊天
- 快捷键: `Ctrl/Cmd + Shift + L`
- 用于问答、代码解释、头脑风暴

### 2. 代码补全
- 自动触发
- 提供智能代码建议

### 3. 内联建议
- 光标处显示建议
- `Tab` 键接受建议

### 4. 代码操作
- 选中代码右键
- 选择 "AI: Explain" 或 "AI: Refactor"

## 📊 配置状态检查

运行以下命令检查配置状态：

```bash
# 检查配置文件
py cursor_kimi_setup.py

# 快速测试连接
py quick_test.py

# 测试所有模型
py model_test.py
```

## 🚀 高级配置

### 自定义参数

```json
{
  "ai": {
    "temperature": 0.3,        // 更保守的回复 (0.0-1.0)
    "maxTokens": 4096,         // 更长的回复
    "contextWindow": 8192,     // 上下文窗口
    "stream": true            // 启用流式输出
  }
}
```

### 多模型配置

```json
{
  "ai": {
    "models": {
      "kimi-8k": "moonshot-v1-8k",
      "kimi-32k": "moonshot-v1-32k",
      "kimi-128k": "moonshot-v1-128k"
    },
    "defaultModel": "moonshot-v1-8k"
  }
}
```

## 📞 获取帮助

如果配置过程中遇到问题：

1. **检查日志**: Cursor 的开发者工具控制台
2. **验证配置**: 运行 `py cursor_kimi_setup.py`
3. **测试连接**: 运行 `py quick_test.py`
4. **查看文档**: [Cursor AI 文档](https://cursor.sh/docs)

## 🎉 配置成功标志

配置成功后，你应该能够：

- ✅ 在 AI 聊天中与 Kimi 对话
- ✅ 获得智能代码补全建议
- ✅ 使用代码解释功能
- ✅ 进行代码重构和优化
- ✅ 获得调试建议

---

**配置时间**: 2025年09月08日 17时39分
**Cursor 版本**: 推荐最新版本
**Kimi 模型**: moonshot-v1-8k ✅ 已验证可用
