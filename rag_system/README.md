# RAG System — 年报智能问答系统

基于检索增强生成（Retrieval-Augmented Generation）的年报分析问答系统。从 PDF 年报中提取结构化文本，通过混合检索 + 重排序精准召回相关片段，交由大模型生成带引用来源的回答。

## 系统架构

```
年报.pdf
   │
   ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  文本解析     │───▶│  结构化切分       │───▶│  structured_segments │
│  (Marker)    │    │  (Markdown标题)   │    │  .json               │
└──────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                        │
                                          ┌─────────────┴─────────────┐
                                          ▼                           ▼
                              ┌─────────────────────────┐   ┌─────────────────────────┐
                              │ 高级切分（pipeline.py 中  │   │ 保持原样                 │
                              │  用户选择）               │   │ (MarkdownHeadingSplitter)│
                              │ · 递归字符切分            │   │                         │
                              │ · 语义切分               │   │                         │
                              └────────────┬────────────┘   └────────────┬────────────┘
                                           └───────────────┬─────────────┘
                                                           ▼
                          ┌────────────────────────────────┼────────────────────────────────┐
                          ▼                                                               ▼
                 ┌─────────────────┐                                              ┌─────────────────┐
                 │  BM25 关键词检索 │                                              │  Chroma 向量检索  │
                 │  (jieba+rank_bm25)│                                              │  (BGE Embedding) │
                 └────────┬────────┘                                              └────────┬────────┘
                          │                                                           │
                          └──────────────────┬────────────────────────────────────────┘
                                             ▼
                                ┌─────────────────────────┐
                                │  混合检索融合             │
                                │  (加权融合 / RRF 倒数排名) │
                                └────────────┬────────────┘
                                             ▼
                                ┌─────────────────────────┐
                                │  重排序                   │
                                │  (BGE CrossEncoder)      │
                                └────────────┬────────────┘
                                             ▼
                                ┌─────────────────────────┐
                                │  上下文拼接 (Token预算控制)│
                                └────────────┬────────────┘
                                             ▼
                                ┌─────────────────────────┐
                                │  LLM 生成回答 (通义千问)   │
                                └────────────┬────────────┘
                                             ▼
                                     答案 + 引用来源
```

## 目录结构

```
rag_system/
├── pipeline.py                # 统一管线入口：解析 PDF → 选择切分策略 → 问答
│
├── 文本解析/                  # PDF 解析与结构化切分
│   ├── parse_pdf.py           # 主入口：Marker 转换 + 标题归一化 + 结构切片
│   ├── markdown_splitter.py   # MarkdownHeadingSplitter：按标题层级切分，保留标题路径
│   ├── pdf_parser_marker.py   # Marker PDF→Markdown 基础转换
│   ├── pdf_parser_pymupdf.py  # PyMuPDF 文本与表格提取（备选方案）
│   ├── 年报.pdf               # 示例年报文件
│   └── output.md              # 归一化后的 Markdown（用于检查）
│
├── 文本切分/                  # 切分策略对比实验
│   └── advanced_splitting.py  # 递归字符切分 vs 语义切分对比
│
├── 混合检索/                  # 核心检索管线
│   ├── bm25.py                # BM25Retriever：jieba 分词 + BM25 关键词检索
│   ├── chroma.py              # Chroma 向量索引构建与检索（BGE 中文模型）
│   ├── hybrid_retriever.py    # HybridRetriever：双路召回 + 融合（weighted/RRF）
│   ├── rerank.py              # Reranker：BGE CrossEncoder / Cohere API 重排序
│   └── qa_system.py           # QASystem：完整问答管线（召回→重排→拼接→生成）
│
├── 系统日志/                  # 统一日志
│   ├── config.py              # 控制台 + 文件双输出，RotatingFileHandler
│   └── logs/
│       └── rag_system.log     # 运行日志
│
├── 异常处理/                  # 自定义异常体系
│   └── exceptions.py          # RAGException 基类及 6 个子类
│
├── 评估/                      # 问答评估数据集构建
│   ├── generate_qa_pairs.py   # Phase 1：LLM 自动生成候选 QA 对（qwen-plus + instructor）
│   ├── review_qa_pairs.py     # Phase 2：人工审查（交互式 / CSV+Excel），产出最终数据集
│   └── README.md              # 评估模块使用说明
│
├── 测试/                      # 边界情况测试
│   └── edge_case_test.py      # 12 项边界测试用例
│
├── data/
│   └── structured_segments.json   # 解析输出的结构化片段
│
└── chroma_db/                 # Chroma 向量数据库持久化目录
```

## 核心模块说明

### 文本解析

以 `parse_pdf.py` 为入口，处理流程为：

1. **Marker 转换**：将 PDF 转为 Markdown 文本，保留标题层级和目录元数据
2. **标题归一化**（`normalize_heading_levels`）：统一 Marker 输出不规范的标题级别（如 `第X节`→一级、`数字、`→二级、`数字.数字`→三级），并将误标为标题的正文行降级
3. **结构切片**（`MarkdownHeadingSplitter`）：支持两种模式
   - `leaf`：叶节点模式，提取最底层标题区域作为独立片段
   - `level`：层级模式，按指定标题级别切分
4. **页码标注**（`add_page_numbers_to_segments`）：根据 Marker 的 table_of_contents 元数据为每个片段回填页码

每个片段包含：`title_path`（标题路径链）、`content`（正文）、`level`（标题级别）、`page`（页码）。

### 文本切分对比

`advanced_splitting.py` 用于实验对比不同切分策略，也供统一管线 `pipeline.py` 调用：

- **递归字符切分**（`run_recursive_splitter` / `RecursiveCharacterTextSplitter`）：按分隔符优先级递归切分，测试了 chunk_size 500/1000/1500 三组参数
- **语义切分**（`semantic_chunking`）：基于句子向量的余弦相似度，在相似度低于阈值处断句，使用 BGE 中文模型编码
- **集成入口**（`apply_advanced_splitting`）：读取 `structured_segments.json`，按 `method="recursive" | "semantic"` 重新切分，并把每个块基于字符偏移量映射回原始片段的 `title_path` / `level` / `page`，结果保存为 `data/chunks_recursive.json` 或 `data/chunks_semantic.json`，可直接供 QA 系统加载

### 混合检索

系统采用双路召回 + 融合 + 重排序的三阶段检索架构：

**BM25 检索**（`bm25.py`）
- 使用 jieba 中文分词 + `rank_bm25` 库构建关键词索引
- 支持空查询保护、无匹配兜底返回

**向量检索**（`chroma.py`）
- 使用 ChromaDB 持久化存储，嵌入模型为 `BAAI/bge-small-zh-v1.5`
- 支持索引构建（`build_chroma_index`）和增量获取（`get_or_create_collection`）

**融合策略**（`hybrid_retriever.py`）
- `weighted`：加权融合，对两路分数做 Min-Max 归一化后按权重（默认向量 0.6 + BM25 0.4）加权求和
- `rrf`：倒数排名融合（Reciprocal Rank Fusion），按排名计算 `1/(k+rank)` 求和，k=60

**重排序**（`rerank.py`）
- `bge` 后端：使用 `BAAI/bge-reranker-base` CrossEncoder 本地模型对 query-doc 对评分
- `cohere` 后端：调用 Cohere Rerank API（需设置 `COHERE_API_KEY`）
- 支持超长文本截断（500 字符）和质量阈值过滤（`min_score`）

### 问答系统

`qa_system.py` 中的 `QASystem` 类串联完整管线：

1. 混合检索召回候选集（默认 RRF 融合，top_k=20）
2. BGE 重排序精选 top_k=5
3. **Token 预算控制**：根据模型上下文窗口（默认 32K）预留 system prompt + 问题 + 答案空间，按 rerank 分数降序逐篇拼接，超预算即停止
4. 调用通义千问（`qwen-plus`，DashScope 兼容接口）生成回答，temperature=0.1 保证稳定性
5. 返回答案 + 引用来源（标题路径、内容片段、rerank 分数）

### 异常处理

`exceptions.py` 定义了统一的异常体系，所有异常继承 `RAGException` 基类，携带 `code`（机器可读错误码）和 `detail`（详细信息），支持 `to_dict()` 序列化：

| 异常类 | 触发场景 |
|--------|---------|
| `DocumentError` | PDF 损坏、JSON 格式错误、文件不存在 |
| `IndexError_` | 空文档、Chroma 锁、embedding 失败 |
| `RetrievalError` | 向量库查询失败、融合计算异常 |
| `RerankError` | 模型加载失败、API 调用失败 |
| `LLMError` | API 超时、限流、鉴权失败 |
| `ConfigError` | 缺少 API Key、环境变量缺失 |

### 日志系统

`config.py` 提供统一的 `get_logger()` 函数：
- 控制台 + 文件双输出，文件使用 `RotatingFileHandler`（单文件最大 10MB，保留 3 个备份）
- 通过环境变量 `LOG_LEVEL` 控制日志级别（默认 INFO）
- logger 缓存机制避免重复添加 handler

## 快速开始

### 环境依赖

```bash
pip install marker-pdf pymupdf pandas \
            langchain-text-splitters sentence-transformers scikit-learn \
            jieba rank-bm25 chromadb numpy \
            openai python-dotenv tiktoken
# 可选：Cohere 重排序后端
pip install cohere
```

### 环境变量配置

在项目根目录创建 `.env` 文件：

```env
# 通义千问 API 密钥（必需）
qwen_api_key=your_dashscope_api_key

# 日志级别（可选，默认 INFO）
LOG_LEVEL=INFO

# Cohere API 密钥（仅使用 Cohere 重排序时需要）
# COHERE_API_KEY=your_cohere_api_key
```

### 运行流程

**方式一：统一管线（推荐）**

```bash
# 一条命令完成：解析 PDF → 选择切分策略（原样/递归/语义）→ 交互式问答
python pipeline.py
```

运行时会依次：
1. 若 `data/structured_segments.json` 不存在，自动解析 `文本解析/年报.pdf`
2. 交互选择切分策略：
   - `[0]` 保持原样（MarkdownHeadingSplitter 标题结构切分）
   - `[1]` 递归字符切分（可调 `chunk_size` / `chunk_overlap`）
   - `[2]` 语义切分（可调 `threshold`）
3. 按选择生成 `data/chunks_recursive.json` 或 `data/chunks_semantic.json`
4. 构建索引并进入交互式问答（输入 `exit` / `quit` / `退出` 结束）

**方式二：分步运行**

```bash
# 1. 解析 PDF 年报，生成结构化片段
python 文本解析/parse_pdf.py

# 2. 切分策略对比实验（可选）
python 文本切分/advanced_splitting.py

# 3. 运行完整问答系统（默认使用 structured_segments.json）
python 混合检索/qa_system.py

#    也可指定切分后的数据文件：
python 混合检索/qa_system.py --data-json data/chunks_recursive.json

# 4. 运行边界测试
python 测试/edge_case_test.py
```

## 技术栈

| 类别 | 技术 |
|------|------|
| PDF 解析 | Marker、PyMuPDF |
| 文本切分 | LangChain RecursiveCharacterTextSplitter、语义切分 |
| 分词 | jieba |
| 关键词检索 | rank_bm25 (BM25Okapi) |
| 向量数据库 | ChromaDB |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 |
| 重排序模型 | BAAI/bge-reranker-base (CrossEncoder) |
| 大模型 | 通义千问 qwen-plus (DashScope OpenAI 兼容接口) |
| Token 计算 | tiktoken |
| 日志 | logging + RotatingFileHandler |

## 测试

`测试/edge_case_test.py` 包含 12 项边界情况测试：

- 空查询 / 纯空白查询
- 无意义查询（不存在的概念）
- 超长查询（2000 字符）
- 空文档构建 BM25 索引
- 数据文件不存在 / JSON 格式错误
- 融合空结果（weighted 和 RRF）
- Markdown 切片器空文本 / 无标题 / 空内容过滤
- Token 估算函数
- 重排序空候选

```bash
python 测试/edge_case_test.py
```
