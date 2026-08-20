# AI 工作区 — 年报智能问答 RAG 系统

基于检索增强生成（RAG）的年报智能问答系统，含完整评估闭环与 Web API 服务。

## 目录结构

```
python/
├── app/                     # ★ 应用/产品层（面向用户的集成入口）
│   ├── main.py              # 统一启动入口：python -m app.main
│   ├── api.py               # FastAPI 服务（/health /chat /feedback）
│   ├── schemas.py           # API 请求/响应 Pydantic 模型
│   ├── qa_system.py         # QASystem 编排层（召回→重排→生成→引用）
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
├── scripts/                 # 开发辅助脚本
│   ├── smoke_test.py        # 包导入冒烟检查
│   ├── verify_runtime.py    # 数据 + BM25 + 数据库功能验证
│   ├── run_tests.py         # 运行边界测试
│   ├── extract_person_info.py
│   ├── fast.py              # FastAPI 基础示例
│   └── deep/                # DeepSeek 学习示例
│
├── enterprise_tools_pkg/    # 独立企业工具包（摘要/情感/润色/Text-to-SQL）
├── 客服机器人/               # 独立客服机器人示例（多轮对话 + token 裁剪）
│
├── requirements.txt         # 统一依赖清单
├── pyproject.toml           # 项目元数据与打包配置（pip install -e .）
├── Dockerfile / .dockerignore
└── .env                     # 密钥与配置（qwen_api_key 等，不入库）
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
# 可选：pip install -e .  让 app / rag_system 全局可导入

# 2. 配置 .env（仓库根目录）
# qwen_api_key=your_dashscope_api_key
# 可选：LLM_MODEL=qwen-plus、LOG_LEVEL=INFO、API_PORT=8000

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
| 运行时功能验证 | `python scripts/verify_runtime.py` |
| 边界测试 | `python scripts/run_tests.py`（或 `python -m rag_system.tests.edge_case_test`） |
| 解析 PDF 年报 | `python -m rag_system.parsing.parse_pdf` |
| 切分对比实验 | `python -m rag_system.splitting.advanced_splitting` |
| 构建评估数据集 | `python -m rag_system.evaluation.generate_qa_pairs` → `review_qa_pairs` |
| RAGAS 多模式评估 | `python -m rag_system.evaluation.run_eval --compare` |
| 容器化部署 | `docker build -t rag-workspace . && docker run -p 8000:8000 -e qwen_api_key=xxx rag-workspace` |
| 启动 Streamlit 前端 | `streamlit run app/streamlit_app.py`（http://localhost:8501） |

## 技术栈

PDF 解析(Marker/PyMuPDF) · 文本切分(LangChain/语义) · BM25(jieba+rank_bm25) · 向量库(ChromaDB+BGE) · 重排序(BGE CrossEncoder) · LLM(通义千问 DashScope) · API(FastAPI+SQLAlchemy) · Web前端(Streamlit) · 评估(RAGAS) · 日志(logging RotatingFileHandler)

## 下一步规划（未包含在本次重构）

- 流式 SSE 接口（FastAPI 侧）、鉴权与限流、多轮对话增强
- 向量库增量复用、多文档管理
- pytest 化 + CI、Alembic 迁移、监控指标、评估闭环回灌
