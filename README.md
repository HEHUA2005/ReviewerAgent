# ReviewerAgent

一个基于 A2A (Agent-to-Agent) 协议的学术论文审稿 AI 智能体，专门用于搜索和审阅学术论文。

## 项目简介

ReviewerAgent 是一个智能的学术论文审稿助手，能够：

- **智能搜索**：通过自然语言描述搜索 arXiv 和 Semantic Scholar 上的学术论文
- **PDF 审稿**：直接上传 PDF 文件进行深度分析和评审
- **结构化评分**：基于方法论、新颖性、清晰度和重要性四个维度进行专业评分
- **交互式选择**：搜索结果支持用户确认和选择

## 核心功能

### 1. 论文搜索与审稿
- 支持中英文自然语言查询
- 整合多个学术数据库（arXiv、Semantic Scholar）
- 智能任务路由，自动识别用户意图
- 提供候选论文列表供用户选择

### 2. PDF 文档处理
- 直接上传 PDF 文件进行审稿
- 智能文本提取和结构化分析
- 支持从 URL 下载论文（如 arXiv 链接）
- 自动缓存机制，避免重复下载

### 3. 专业审稿系统
- **方法论评估**：研究设计、实验设置和方法评价
- **新颖性分析**：原创性和对领域的贡献度
- **清晰度评分**：写作质量、组织结构和表达清晰度
- **重要性判断**：对领域的影响力和相关性

### 4. A2A 协议集成
- 完全兼容 Agent-to-Agent 通信协议
- 支持流式响应和实时交互
- 标准化的 JSON-RPC API 接口
- 支持多种输入模式（文本、文件）

## 技术架构

```
ReviewerAgent/
├── agent_executor.py      # 主要的智能体执行器
├── src/
│   ├── llm_client.py      # LLM 客户端集成
│   ├── llm_reviewer.py    # LLM 驱动的审稿引擎
│   ├── review_engine.py   # 核心审稿逻辑
│   ├── paper_search.py    # 学术论文搜索
│   ├── pdf_processor.py   # PDF 文档处理
│   └── output_formatter.py # 输出格式化
├── tests/                 # 测试套件
└── paper/                 # PDF 缓存目录
```

## 快速开始

### 环境要求
- Python 3.10+
- OpenAI API 密钥（或兼容的 API）
- arXiv API 访问
- Semantic Scholar API 密钥（可选）

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/HEHUA2005/ReviewerAgent.git
   cd ReviewerAgent
   ```

2. **安装依赖**
   ```bash
   pip install -e .
   ```

3. **配置环境变量**
   ```bash
   # 创建 .env 文件
   API_KEY="your-openai-api-key"
   LLM_MODEL="gpt-4o"
   AGENT_PORT="9997"
   SEMANTIC_SCHOLAR_API_KEY="your-semantic-scholar-api-key"  # 可选
   ```

### 启动服务

```bash
# 使用启动脚本
./start_agent.sh

# 或直接运行
python -m __main__
```

服务将在 http://localhost:9997 启动。

## 使用示例

### 1. 搜索论文
```bash
curl -X POST http://localhost:9997/api/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "sendMessage",
    "params": {
      "taskId": "task_id",
      "message": {
        "role": "user",
        "parts": [{"text": "找一些关于 Transformer 注意力机制的论文"}]
      }
    },
    "id": 1
  }'
```

### 2. 上传 PDF 审稿
```bash
curl -X POST http://localhost:9997/api/jsonrpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "sendMessage",
    "params": {
      "taskId": "task_id",
      "message": {
        "role": "user",
        "parts": [{
          "file": {
            "mimeType": "application/pdf",
            "data": "BASE64_ENCODED_PDF_DATA"
          }
        }]
      }
    },
    "id": 1
  }'
```


## 审稿标准

每篇论文将基于以下四个维度进行 1-10 分评分：

- **方法论 (Methodology)**：研究设计、实验设置和方法的严谨性
- **新颖性 (Novelty)**：原创性和对学术领域的贡献
- **清晰度 (Clarity)**：写作质量、组织结构和表达的清晰程度
- **重要性 (Significance)**：对领域的影响力和实际意义

## 配置选项

通过环境变量进行配置：

- `API_KEY`：OpenAI API 密钥（必需）
- `LLM_MODEL`：使用的 LLM 模型（默认：gpt-4o）
- `AGENT_PORT`：服务端口（默认：9997）
- `ARXIV_CATEGORIES`：arXiv 搜索类别（默认：cs.AI,cs.LG,cs.CL）
- `SEMANTIC_SCHOLAR_API_KEY`：Semantic Scholar API 密钥（可选）
- `REVIEW_CRITERIA`：审稿标准类型（默认：academic_peer_review）

## 许可证

本项目采用 MIT 许可证。

## 免责声明

这是一个演示项目。在生产环境中使用时，请确保实施适当的安全措施，包括输入验证和凭据的安全处理。