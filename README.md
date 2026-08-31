# AI 工作区 —  RAG 数据库智能问答系统

基于检索增强生成（RAG）的年报智能问答系统，含完整评估闭环与 Web API 服务。

## 目录结构

```
RAG_qa_system/
├── app/                     # ★ 应用/产品层（面向用户的集成入口）
│   ├── main.py              # 统一启动入口：python -m app.main
│   ├── api.py               # FastAPI 服务（/health /chat /feedback）
│   ├── schemas.py           # API 请求/响应 Pydantic 模型
│   ├── qa_system.py         # QASystem 编排层（召回→重排→生成→引用）
│   ├── conversation.py      # ★ 多轮对话记忆（历史裁剪 + 检索查询改写）
│   ├── models.py / database.py  # SQLAlchemy ORM 与数据库
│   ├── config.py            # ★ 统一配置中心（路径/模型/端口，读 .env）
│   ├── cli.py               # 交互式 CLI 管线：解析 PDF → 切分 → 问答
│   └── streamlit_app.py     # Streamlit 前端：流式聊天 + 上传 + 历史
│
├── rag_system/              # 引擎层（可复用 RAG 组件库）
│   ├── parsing/             # PDF 解析与结构化切分
│   ├── splitting/           # 文本切分（递归 / 语义）
│   ├── retrieval/           # 混合检索（BM25 + Chroma + 融合 + 重排序）
│   ├── common/              # 统一日志 logging_config.py + 异常 exceptions.py
│   ├── evaluation/          # 评估（QA 对生成/审查、RAGAS 多模式评估）
│   ├── tests/               # 边界测试
│   ├── data/                # 数据文件（年报.pdf、结构化片段、评估数据集）
│   ├── chroma_db/           # Chroma 向量库持久化
│   └── logs/                # 运行日志
│
├── docs/                    # 文档与图表
│   └── eval_compare.png     # RAGAS 多模式评估对比图
│
├── scripts/                 # 开发辅助脚本
│   ├── smoke_test.py        # 包导入冒烟检查
│   ├── smoke_test_memory.py # 多轮记忆模块冒烟测试（不调真实 LLM）
│   ├── verify_runtime.py    # 数据 + BM25 + 数据库功能验证
│   └── run_tests.py         # 运行边界测试
│
├── data/                    # SQLite 数据库（对话历史 / 反馈）
├── .streamlit/              # Streamlit 主题配置
│
├── requirements.txt         # 统一依赖清单
├── pyproject.toml           # 项目元数据与打包配置（pip install -e .）
├── docker-compose.yml       # 阿里云 ECS 一键部署编排
├── Dockerfile / .dockerignore
├── .env.example             # 环境变量示例
├── LICENSE
└── .env                     # 密钥与配置（deepseek_api_key 等，不入库）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
# 可选：pip install -e .  让 app / rag_system 全局可导入

# 2. 配置 .env（仓库根目录）
# deepseek_api_key=your_deepseek_api_key
# 可选：LLM_MODEL=deepseek-chat、LLM_BASE_URL=https://api.deepseek.com/v1、LOG_LEVEL=INFO、API_PORT=8000

# 3. 启动 Web API
python -m app.main            # 等价于 uvicorn app.api:app --reload
# 接口文档：http://localhost:8000/docs

# 4. 或使用交互式 CLI 完整管线（解析→切分→问答）
python -m app.cli

# 5. 启动 Streamlit 前端（流式聊天 + 文件上传 + 历史记录）
streamlit run app/streamlit_app.py
# 访问：http://localhost:8501
```

## 常用命令

| 用途 | 命令 |
|------|------|
| 包导入冒烟检查 | `python scripts/smoke_test.py` |
| 多轮记忆模块测试 | `python scripts/smoke_test_memory.py` |
| 运行时功能验证 | `python scripts/verify_runtime.py` |
| 边界测试 | `python scripts/run_tests.py`（或 `python -m rag_system.tests.edge_case_test`） |
| 解析 PDF 年报 | `python -m rag_system.parsing.parse_pdf` |
| 切分对比实验 | `python -m rag_system.splitting.advanced_splitting` |
| 构建评估数据集 | `python -m rag_system.evaluation.generate_qa_pairs` → `review_qa_pairs` |
| RAGAS 多模式评估 | `python -m rag_system.evaluation.run_eval --compare` |
| 容器化部署 | `docker build -t rag-workspace . && docker run -p 8000:8000 -e deepseek_api_key=xxx rag-workspace` |
| 启动 Streamlit 前端 | `streamlit run app/streamlit_app.py`（http://localhost:8501） |

## 多轮对话记忆

支持跨轮次的指代消解与省略补全（如「那净利润呢」→ 结合上一轮理解为「贵州茅台 2023 年净利润是多少」），核心在 `app/conversation.py`：

```
用户问题 ──┬─ 改写分支：LLM 结合历史改写为完整独立问题 ──→ 检索 + 重排序（召回更准）
           │
           └─ 生成分支：原始问题 + 历史消息注入 LLM messages ──→ 回答（自然承接上下文）
```

- **记忆裁剪**（`trim_history`）：双重上限——最近 `MAX_HISTORY_TURNS` 轮（默认 5）+ token 预算 `HISTORY_TOKEN_BUDGET`（默认 1500），超出从最旧丢弃，并剔除 sources/chat_id 等非 LLM 字段；
- **查询改写**（`rewrite_query`）：改写只用于检索，生成仍用原始问题（历史已注入，模型可自行消解指代）；改写调用失败/超时/结果异常时自动回退原始问题，**可用性优先**；
- **两路接入**：Streamlit 前端传入页面会话历史（`streamlit_app.py`）；API 侧 `/chat` 按 `session_id` 从 ChatHistory 表加载最近 5 轮（不传 `session_id` 则保持单轮行为，向后兼容）；
- **成本**：每轮带历史的提问多一次改写调用（≤200 token、15s 超时）；历史 token 计入上下文预算，不会挤爆窗口。

验证：`python scripts/smoke_test_memory.py`（裁剪/降级/改写等 7 项，不调真实 LLM）。

## 技术栈

PDF 解析(Marker/PyMuPDF) · 文本切分(LangChain/语义) · BM25(jieba+rank_bm25) · 向量库(ChromaDB+BGE) · 重排序(BGE CrossEncoder) · 多轮记忆(历史裁剪+查询改写) · LLM(DeepSeek) · API(FastAPI+SQLAlchemy) · Web前端(Streamlit) · 评估(RAGAS) · 日志(logging RotatingFileHandler)

## RAGAS 评估对比

使用 [RAGAS](https://github.com/explodinggradients/ragas) 对 5 种检索模式进行了逐题评估与多模式对比（数据来源：`rag_system/data/eval_results_*.csv`）。

**5 个评估指标**：忠实度（防幻觉）、答案相关性（切题）、上下文精确率（排序质量）、上下文召回率（召回完整性）、答案正确性（事实一致性）。

![RAGAS 多模式检索评估对比](docs/eval_compare.png)

### 各模式指标均值

| 检索模式 | 忠实度 | 答案相关性 | 上下文精确率 | 上下文召回率 | 答案正确性 |
|----------|--------|-----------|-------------|-------------|-----------|
| full（混合检索+重排序） | 0.5909 | **0.8804** | 0.5593 | 0.8333 | 0.6367 |
| vector_rerank（向量+重排序） | 0.5909 | 0.8722 | 0.6042 | 0.8333 | 0.6000 |
| hybrid_no_rerank（混合检索，无重排） | 0.6528 | 0.8471 | 0.5681 | 0.8333 | 0.7067 |
| vector_only（仅向量） | 0.6319 | 0.8306 | 0.6111 | 0.8333 | **0.7582** |
| bm25_only（仅 BM25） | **0.7986** | 0.8613 | **0.6264** | 0.8333 | 0.6999 |

### 结论与建议

- **答案相关性**：完整管线 `full` 最高（0.88），混合检索 + 重排序对"答得准、答得贴合问题"最有利；
- **忠实度 / 上下文精确率**：`bm25_only` 最高，年报中的数值、术语类问答对关键词精确匹配更敏感，BM25 能更精准地锁定来源片段；
- **答案正确性**：`vector_only` 最高（0.76），与语义召回上下文更完整有关；
- **上下文召回率**：5 种模式均为 0.83，说明召回源一致，模式间的差异主要来自排序与筛选策略；
- **建议**：综合体验选 `full`（答案相关性第一、整体均衡）；对纯数值/术语类问答场景，可提高 BM25 权重或将 BM25 作为兜底通道；实际使用中可用本对比图数据按需切换 `retrieval_mode`。

## 阿里云部署（Docker Compose）

在单台阿里云 ECS 上使用 Docker Compose 一键部署（FastAPI 后端 + Streamlit 前端）。

```bash
# 1. 服务器安装 Docker + Compose 插件
# 2. 上传项目源码到服务器（.env 含密钥，不要上传，在服务器上单独创建）
# 3. 配置环境变量
cp .env.example .env
#    编辑 .env，填入 deepseek_api_key
# 4. 构建并启动
docker compose up -d --build
# 5. 查看状态与日志
docker compose ps
docker compose logs -f rag-api
```

- 后端 API 文档：`http://<服务器IP>:8000/docs`
- 前端页面：`http://<服务器IP>:8501`

### 配置说明

| 项 | 说明 |
|----|------|
| `deepseek_api_key` | 必填，DeepSeek API 密钥（[平台](https://platform.deepseek.com/)） |
| `HF_ENDPOINT` | 默认 `https://hf-mirror.com`。国内 ECS 无法直连 HuggingFace，BGE 嵌入/重排模型从该镜像站下载 |
| `MAX_HISTORY_TURNS` | 可选，多轮记忆保留轮数（默认 5，1 轮 = 一问一答） |
| `HISTORY_TOKEN_BUDGET` | 可选，历史注入 LLM 的 token 预算（默认 1500，超出丢最旧） |
| 数据卷 | rag-api / rag-web 各自独立挂载 data、chroma_db、SQLite、logs（避免 ChromaDB/SQLite 并发写锁冲突）；`hf_cache` 卷共享模型缓存 |
| 首启时间 | 首次启动需下载 BGE 模型并重建索引，约 3~10 分钟，健康检查 `start-period=300s` 已覆盖 |

### 阿里云 ECS 注意事项

- **安全组**放行 8000 / 8501 端口（生产建议前面加 Nginx / SLB 只暴露 443，或仅内网访问）。
- 建议 **2 核 4GB 以上**：BGE 重排模型（bge-reranker-base 约 1.1GB）+ 嵌入模型加载约需 1.5GB 内存。
- 若需公网 HTTPS 域名：使用阿里云 SLB/ALB + SSL 证书，转发到 8000 / 8501。
- **默认单 worker**：本应用每个 worker 都会独立加载模型并重建 Chroma 索引，多 worker 会内存翻倍并引发锁冲突。若要扩展，建议先将 Chroma 迁移为 Server 模式、SQLite 迁移为 RDS PostgreSQL（代码已支持 `DATABASE_URL` 非 SQLite 分支）。
- 数据持久化在命名卷中，`docker compose down` 不会删除；确认清理用 `docker compose down -v`。
- 更新知识库数据：`docker compose cp rag_system/data/structured_segments.json rag-api:/workspace/rag_system/data/` 后重启容器（启动时会自动重建索引）。

## 下一步规划

- 流式 SSE 接口（FastAPI 侧）、鉴权与限流
- 向量库增量复用、多文档管理
- pytest 化 + CI、Alembic 迁移、监控指标、评估闭环回灌
- 记忆增强：对话摘要压缩（超长历史蒸馏为摘要）、按 session 的向量记忆库
