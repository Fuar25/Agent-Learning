# LangChain、LangGraph、LlamaIndex & Neo4j 学习项目

以 Jupyter Notebook 为载体，系统学习 LLM 应用开发全栈：LangChain（基础框架）→ LangGraph（复杂编排）→ LlamaIndex（RAG 专家）→ Neo4j（知识图谱）。

## 目录结构

```
.
├── langchain/          # LangChain 教程 notebooks
├── langgraph/          # LangGraph 教程 notebooks
├── llamaindex/         # LlamaIndex 教程 notebooks（RAG 专项）
├── neo4j/              # Neo4j & 知识图谱教程 notebooks
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

通过 OpenAI 兼容接口接入通义千问（Qwen）API。LangChain/LangGraph/Neo4j 使用 `langchain-openai`，LlamaIndex 使用 `llama-index-llms-openai`。

API Key 存于环境变量（`.env` 文件），不写入代码。

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

### LlamaIndex（`llamaindex/`）

| 章节 | 主题 |
|------|------|
| 01 | 环境搭建与核心概念（LLM、Document、Settings）|
| 02 | 文档加载与索引构建（Reader、Node、VectorStoreIndex）|
| 03 | 查询引擎与响应合成（Retriever、ResponseSynthesizer、Prompts）|
| 04 | 向量存储与持久化（FAISS、StorageContext、Embedding）|
| 05 | 高级检索策略（混合检索、重排序、HyDE、子问题分解）|
| 06 | 对话引擎与多文档 Agent（ChatEngine、RouterQueryEngine、Tools）|
| 07 | 生产实践与完整项目（评估、可观测性、端到端系统）|

### Neo4j / 知识图谱（`neo4j/`）

| 章节 | 主题 |
|------|------|
| 01 | 环境搭建与图数据库核心概念（Docker、节点、关系、属性）|
| 02 | Cypher 查询语言（MATCH、CREATE、MERGE、路径查询）|
| 03 | 数据建模与批量导入（UNWIND、事务、约束索引）|
| 04 | LangChain Neo4j 集成（Text-to-Cypher、Neo4jVector）|
| 05 | LLM 知识图谱自动构建（LLMGraphTransformer）|
| 06 | GraphRAG 图增强检索（向量 + 图混合检索）|
| 07 | 图驱动 Agent 与生产实践（ReAct Agent、LangGraph 工作流）|
