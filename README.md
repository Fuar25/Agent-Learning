# LangChain & LangGraph 学习项目

以 Jupyter Notebook 为载体，系统学习 LangChain 和 LangGraph 框架。

## 目录结构

```
.
├── langchain/          # LangChain 教程 notebooks
├── langgraph/          # LangGraph 教程 notebooks
├── ReAct/              # ReAct 实验代码
├── pyproject.toml      # 统一依赖管理
└── uv.lock
```

## 环境搭建

使用 **uv** 管理 Python 环境和依赖：

```bash
# 安装依赖
uv sync

# 启动 Jupyter
uv run --with jupyter jupyter lab

# 新增依赖
uv add <package>
```

## 模型配置

通过 `langchain-openai` 的 OpenAI 兼容接口接入通义千问（Qwen）API。

API Key 存于环境变量 `Qwen_API_KEY`，不写入代码。

## 学习路线

### LangChain（`langchain/`）

| 章节 | 主题 |
|------|------|
| 01 | 环境搭建与核心概念（Models、Runnable 接口）|
| 02 | Prompt Templates（提示词模板）|
| 03 | LCEL 链式表达式（`\|` 管道组合）|
| 04 | Memory（对话历史与状态管理）|
| 05 | Retrieval / RAG（文档加载、向量存储、检索）|
| 06 | Agents（工具调用与自主决策）|
| 07 | 生产实践（调试、追踪、部署）|

### LangGraph（`langgraph/`）

| 章节 | 主题 |
|------|------|
| 01 | 环境搭建与核心概念（StateGraph、节点、边）|
| 02 | State 设计（TypedDict、Annotated reducer）|
| 03 | 条件分支与循环（conditional_edges、路由函数）|
| 04 | 工具调用 Agent（Tool Node、ReAct 模式）|
| 05 | Human-in-the-Loop（中断、审批、人工干预）|
| 06 | 多 Agent 协作（子图、Supervisor 模式）|
| 07 | 持久化与检查点（MemorySaver、状态恢复）|
| 08 | 生产实践（流式输出、调试、部署）|
