# 评估模块 — 从年报构建问答评估数据集

## 流程总览

```
Phase 1: LLM 自动生成候选 QA 对          Phase 2: 人工审查                  最终产出
┌─────────────────────────────┐      ┌──────────────────────┐      ┌────────────────────────┐
│ generate_qa_pairs.py         │ ───▶ │ review_qa_pairs.py    │ ───▶ │ data/qa_pairs_final.json│
│ 读取 chunks_*.json           │      │ 交互式审查 / CSV 审查 │      │ 50 条标准化 QA 对       │
│ qwen-plus + instructor 生成   │      │ 保留/编辑/删除/改属性  │      │ 可直接用于检索与答案评测 │
└─────────────────────────────┘      └──────────────────────┘      └────────────────────────┘
```

## Phase 1：自动生成

```bash
# 全量生成（每块 4 条，约 64 条候选）
python -m rag_system.evaluation.generate_qa_pairs

# 自定义参数
python -m rag_system.evaluation.generate_qa_pairs --per-chunk 5 --start 0 --end 8
```

产出：`data/qa_pairs_raw.json`

每条候选包含：
- `question` / `ground_truth`（标准答案，须从原文提取）
- `question_type`（数值提取/事实检索/归纳概括/风险分析/对比分析）
- `difficulty`（easy/medium/hard）
- `source_sentence`（原文依据句，用于校验答案 grounded）
- `source_chunk_id` / `source_title_path` / `source_page`（来源定位）

## Phase 2：人工审查

### 方式 A：交互式终端审查

```bash
python -m rag_system.evaluation.review_qa_pairs
```

逐条展示，命令：
| 命令 | 含义 |
|------|------|
| `y` / 回车 | 保留 |
| `e` | 编辑问题/答案后保留 |
| `d` | 删除 |
| `t` | 修改问题类型 |
| `g` | 修改难度 |
| `q` | 退出并保存 |

### 方式 B：CSV + Excel 审查

```bash
# 1. 导出 CSV
python -m rag_system.evaluation.review_qa_pairs --export-csv

# 2. Excel 打开 rag_system/data/qa_pairs_review.csv 增删改后保存

# 3. 导入生成最终数据集
python -m rag_system.evaluation.review_qa_pairs --import-csv
```

## 最终数据集格式

```json
[
  {
    "id": "qa_001",
    "question": "2025年服装行业规模以上企业工业增加值同比变化了多少？",
    "ground_truth": "同比下降3.0%",
    "question_type": "数值提取",
    "difficulty": "easy",
    "source_sentence": "2025年1-12月,服装行业规模以上企业工业增加值同比下降3.0%",
    "source_chunk_id": 0,
    "source_title_path": ["第一节 重要提示"],
    "source_page": 2
  }
]
```

## 后续评测用法（扩展）

1. **检索命中率**：用 `question` 检索，检查 `ground_truth` 所在 chunk 是否出现在 top_k
2. **答案质量**：用 QASystem 回答，与 `ground_truth` 做 LLM-as-judge 比对
3. **切分效果对比**：同一批 50 条问题分别跑原样/递归/语义切分，对比检索指标
