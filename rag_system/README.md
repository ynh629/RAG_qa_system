# RAG System — 年报智能问答引擎层

基于检索增强生成（Retrieval-Augmented Generation）的年报分析问答系统。从 PDF 年报中提取结构化文本，通过混合检索 + 重排序精准召回相关片段，交由大模型生成带引用来源的回答。

> 本目录为**引擎层**（可复用 RAG 组件库）；面向用户的集成入口（FastAPI、CLI、QASystem 编排）位于仓库根目录 `app/`。所有路径/模型配置统一由 `app/config.py` 管理。

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
                              │ 高级切分（app/cli.py 中   │   │ 保持原样                 │
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
                                │  LLM 生成回答 (DeepSeek)   │
                                └────────────┬────────────┘
                                             ▼
                                     答案 + 引用来源
```

## 目录结构

```
rag_system/
├── parsing/                    # PDF 解析与结构化切分
│   ├── parse_pdf.py            # 主入口：Marker 转换 + 标题归一化 + 结构切片 + 页码标注
│   ├── markdown_splitter.py    # MarkdownHeadingSplitter：按标题层级切分，保留标题路径
│   ├── pdf_parser_marker.py    # Marker PDF→Markdown 基础转换
│   └── pdf_parser_pymupdf.py   # PyMuPDF 文本与表格提取（备选方案）
│
├── splitting/                  # 文本切分策略
│   └── advanced_splitting.py   # 递归字符切分 vs 语义切分对比 + 映射回原始片段
│
├── retrieval/                  # 混合检索底层组件
│   ├── bm25.py                 # BM25Retriever：jieba 分词 + rank_bm25 关键词检索
│   ├── chroma.py               # Chroma 向量索引构建与检索（BAAI/bge-small-zh-v1.5）
│   ├── hybrid_retriever.py     # HybridRetriever：双路召回 + 融合（weighted / RRF）
│   └── rerank.py               # Reranker：BGE CrossEncoder / Cohere API 重排序
│
├── common/                     # 公共基础
│   ├── logging_config.py       # 统一日志（控制台 + RotatingFileHandler，10MB×3）
│   └── exceptions.py           # RAGException 基类及 6 个子类
│
├── evaluation/                 # 问答评估
│   ├── generate_qa_pairs.py    # Phase 1：LLM 自动生成候选 QA 对（deepseek + instructor）
│   ├── review_qa_pairs.py      # Phase 2：人工审查（交互式 / CSV+Excel）
│   ├── run_eval.py             # RAGAS 5 指标多模式评估与对比
│   ├── ragas_config.py         # RAGAS 的 LLM / Embedding 封装（复用配置中心）
│   └── README.md               # 评估模块使用说明
│
├── tests/                      # 边界测试
│   └── edge_case_test.py       # 16 项边界测试用例
│
├── data/                       # 数据文件（年报.pdf、structured_segments.json、评估数据集等）
├── chroma_db/                  # Chroma 向量数据库持久化目录
└── logs/                       # 运行日志（rag_system.log）
```

## 核心模块说明

### 文本解析（`parsing/`）

以 `parse_pdf.py` 为入口，处理流程为：

1. **Marker 转换**：将 PDF 转为 Markdown 文本，保留标题层级和目录元数据
2. **标题归一化**（`normalize_heading_levels`）：统一 Marker 输出不规范的标题级别（如 `第X节`→一级、`数字、`→二级、`数字.数字`→三级），并将误标为标题的正文行降级
3. **结构切片**（`MarkdownHeadingSplitter`）：支持 `leaf`（叶节点）与 `level`（指定级别）两种模式
4. **页码标注**（`add_page_numbers_to_segments`）：根据 Marker 目录元数据回填页码

每个片段包含：`title_path`（标题路径链）、`content`（正文）、`level`（标题级别）、`page`（页码）。

### 文本切分（`splitting/`）

- **递归字符切分**（`RecursiveCharacterTextSplitter`）：按分隔符优先级递归切分，可调 `chunk_size` / `chunk_overlap`
- **语义切分**（`semantic_chunking`）：基于句子向量余弦相似度断句，可调 `threshold`
- **集成入口**（`apply_advanced_splitting`）：读取 `structured_segments.json` 重新切分，并把每个块基于字符偏移量映射回原始片段的 `title_path` / `level` / `page`

### 混合检索（`retrieval/`）

三阶段检索架构：**双路召回 + 融合 + 重排序**。

- **BM25**：jieba 中文分词 + `rank_bm25` 关键词检索，支持空查询保护、无匹配兜底
- **向量**：ChromaDB 持久化 + `BAAI/bge-small-zh-v1.5` 嵌入，支持索引构建与增量获取（`get_or_create_collection`）
- **融合策略**：`weighted`（分数归一化加权求和，默认向量 0.6 + BM25 0.4）或 `rrf`（倒数排名融合，k=60）
- **重排序**：`bge`（本地 `BAAI/bge-reranker-base` CrossEncoder）或 `cohere`（API）。bge 后端输出经 **sigmoid 归一化到 [0,1]**（原始 logit 不可直接比较），超 512 token 长片段采用 **500 字符滑窗（50 重叠，最多 8 窗）逐窗打分取最大值**，取代旧的头部截断；支持质量阈值 `min_score` 过滤（语义为"相关性概率过半"，默认 0.5）

### 问答系统（`app/qa_system.py`）

`QASystem` 类串联完整管线，位于应用层 `app/`：

1. **多轮记忆**（可选传入 `history`）：裁剪最近几轮历史 + LLM 查询改写（消解指代/补全省略，改写仅用于检索，详见 `app/conversation.py`）
2. 混合检索召回候选集（默认 RRF 融合，top_k=20，使用改写后的完整问题）
3. BGE 重排序精选 top_k=5
4. **Token 预算控制**：按 rerank 分数降序逐篇拼接，超预算即停止（历史 token 一并计入预算）
5. 调用 DeepSeek（默认 `deepseek-chat`，OpenAI 兼容接口）生成回答，temperature=0.1，历史消息注入 LLM messages
6. 返回答案 + 引用来源（标题路径、内容片段、rerank 分数）+ 改写后查询
7. 支持 5 种检索模式：`full` / `vector_only` / `bm25_only` / `hybrid_no_rerank` / `vector_rerank`

### 异常处理（`common/exceptions.py`）

`exceptions.py` 定义了统一异常体系，所有异常继承 `RAGException` 基类，携带 `code`（机器可读错误码）和 `detail`（详细信息），支持 `to_dict()` 序列化：

| 异常类 | 触发场景 |
|--------|---------|
| `DocumentError` | PDF 损坏、JSON 格式错误、文件不存在 |
| `IndexError_` | 空文档、Chroma 锁、embedding 失败 |
| `RetrievalError` | 向量库查询失败、融合计算异常 |
| `RerankError` | 模型加载失败、API 调用失败 |
| `LLMError` | API 超时、限流、鉴权失败 |
| `ConfigError` | 缺少 API Key、环境变量缺失 |

### 日志系统（`common/logging_config.py`）

`get_logger()` 提供控制台 + 文件双输出，文件使用 `RotatingFileHandler`（单文件最大 10MB，保留 3 个备份）；通过环境变量 `LOG_LEVEL` 控制级别（默认 INFO）。

## 快速开始

### 环境依赖

```bash
# 在仓库根目录
pip install -r requirements.txt
# 可选：pip install -e .  让 app / rag_system 全局可导入
```

### 环境变量配置

在仓库根目录创建 `.env`（由 `app/config.py` 统一加载）：

```env
deepseek_api_key=your_deepseek_api_key
# 可选覆盖项：
# LLM_MODEL=deepseek-chat
# LLM_BASE_URL=https://api.deepseek.com/v1
# LOG_LEVEL=INFO
# API_PORT=8000
# MAX_HISTORY_TURNS=5        # 多轮记忆保留轮数
# HISTORY_TOKEN_BUDGET=1500  # 历史注入 LLM 的 token 预算
```

### 运行流程

**方式一：统一 CLI 管线（推荐，位于 app 层）**

```bash
cd ..  # 仓库根目录
python -m app.cli
```

运行时会依次：解析 PDF → 交互选择切分策略（`[0]` 保持原样 / `[1]` 递归 / `[2]` 语义）→ 构建索引 → 交互式问答。

**方式二：分步运行**

```bash
# 1. 解析 PDF 年报，生成结构化片段到 data/
python -m rag_system.parsing.parse_pdf

# 2. 切分策略对比实验（可选）
python -m rag_system.splitting.advanced_splitting

# 3. 运行完整问答系统（默认读取 data/structured_segments.json）
python -m app.qa_system --interactive --data-json rag_system/data/structured_segments.json

# 4. 运行边界测试
python -m rag_system.tests.edge_case_test
```

### 验证脚本（scripts/）

```bash
python scripts/smoke_test.py             # 包导入冒烟检查
python scripts/smoke_test_memory.py      # 多轮记忆模块测试（不调真实 LLM）
python scripts/test_rerank_scoring.py    # 重排打分纯函数测试（sigmoid + 滑窗，14 项）
python scripts/verify_runtime.py         # 数据 + BM25 + 数据库功能验证
python scripts/run_tests.py              # 运行边界测试（结果写入 UTF-8 日志）
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
| 大模型 | DeepSeek（OpenAI 兼容接口） |
| 多轮对话记忆 | 历史裁剪（轮数 + token 双上限）+ LLM 查询改写（`app/conversation.py`） |
| Token 计算 | tiktoken |
| 日志 | logging + RotatingFileHandler |
| 评估 | RAGAS（faithfulness / relevancy / precision / recall / correctness） |

