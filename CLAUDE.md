# LangChain & LangGraph 学习项目

## 学习者背景

- **Python 水平**：中级，熟悉常用库（pandas、requests 等），有项目经验
- **LLM 经验**：已系统学习 LangChain，正在学习 LangGraph
- **学习目标**：系统掌握 LangChain 和 LangGraph，具备独立构建 LLM 应用和多步骤 Agent 的能力
- **教学语言**：中文讲解，代码注释可中可英

## 项目结构

- `langchain/` — LangChain 教程 notebooks
- `langgraph/` — LangGraph 教程 notebooks
- `ReAct/` — ReAct 实验代码（独立，不在教程体系内）

## 章节组织约定

- 每章一个 `.ipynb` 文件，命名格式：`NN_章节名称.ipynb`（如 `01_环境搭建与核心概念.ipynb`）
- 章节编号两位补零，按学习顺序递增
- 每章结构固定：学习目标 → 代码示例 → 概念讲解 → 总结 → 下一章预告

## Notebook 编写规范

- **先代码后解释**：每个知识点先给出可运行的代码示例，紧接着用 Markdown 单元格解释"刚才发生了什么"以及"为什么这样设计"
- 每个代码单元格只做一件事，配有简短注释说明意图
- 关键 API 参数用表格列出，便于查阅
- 代码必须可独立运行（从头顺序执行不出错）
- 敏感信息（API Key）通过环境变量或 `getpass` 输入，绝不硬编码

## 技术栈

### 环境管理

使用 **uv** 管理 Python 环境和依赖：

```bash
# 启动 Jupyter
uv run --with jupyter jupyter lab

# 新增依赖
uv add <package>
```

依赖记录在 `pyproject.toml`，锁文件为 `uv.lock`，两者均纳入版本控制。

### 模型

通过 `langchain-openai` 的 OpenAI 兼容接口接入模型（通义千问、GLM 等）。

所有密钥和 API 地址统一存放在项目根目录的 `.env` 文件中：

```
Qwen_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
Qwen_API_KEY=sk-...
GLM_API_BASE=https://open.bigmodel.cn/api/paas/v4/
GLM_API_KEY=...
```

每章开头统一使用以下初始化代码加载环境：

```python
import os
from dotenv import load_dotenv
load_dotenv("../.env")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_base=os.environ["Qwen_API_BASE"],
    openai_api_key=os.environ["Qwen_API_KEY"],
)
```

**绝不硬编码** API Key 或 API Base URL。`.env` 文件已加入 `.gitignore`，不纳入版本控制。

### 依赖包

- `langchain`、`langchain-core`、`langchain-community`（LangChain 生态）
- `langchain-openai`（通过 OpenAI 兼容接口对接 Qwen）
- `langgraph`（LangGraph 核心框架：状态图、节点、边）
- `faiss-cpu`（向量检索，用于 RAG 章节）
- `python-dotenv`（从 `.env` 文件加载环境变量）

## AI 协作约定

在此目录中，Claude 的职责是：

1. **生成新章节**：按上述结构创建 `.ipynb` 文件，内容要循序渐进，承接前一章知识点
2. **修改已有章节**：补充示例、修正错误、更新 API 用法，不破坏既有代码逻辑
3. **版本敏感**：每章开头确认版本；使用任何 API 前，先确认该用法在当前版本中是否仍然有效，避免教授已废弃的写法
4. **不做的事**：不引入当前章节范围外的复杂概念；不在未讲解的前提下直接使用高级特性；不修改章节编号顺序

### 教学风格

- **代码优先**：先展示一段可运行的代码，再解释它做了什么、为什么这样写。学习者偏好从具体到抽象
- **对比教学**：引入新概念时，先展示"不用这个特性怎么写"，再展示"用了之后的改进"，让差异一目了然
- **预判报错**：当代码可能因版本、网络、API 限制等原因出错时，在代码单元格后补充"常见问题"说明，而非等学习者踩坑
- **少说废话**：不堆砌营销话术，专注于"这个东西解决什么问题"和"怎么用"
- **LangGraph 章节与 LangChain 对比**：学习者已有 LangChain 基础，适当说明两者的关系和区别，帮助迁移已有知识

### 每次生成前的检查清单

1. 查看对应子文件夹中已有的 `.ipynb` 文件，确认编号和进度
2. 回顾上一章末尾的"下一章预告"，保持知识连贯
3. 确认本章涉及的 API 在当前版本中的最新用法
