# run_eval.py
"""
RAGAS 评估主脚本。
使用 RAGAS 框架对 RAG 系统进行全面评估，包含 5 个核心指标：
  - faithfulness        忠实度：回答是否基于检索上下文（防幻觉）
  - answer_relevancy    答案相关性：回答是否切题
  - context_precision   上下文精确率：相关上下文是否排在前面
  - context_recall      上下文召回率：标准答案信息是否被检索到
  - answer_correctness  答案正确性：回答与标准答案的事实一致性

支持多种检索模式评估：
  - full               完整 RAG（混合检索 + 重排序）
  - vector_only        基础 RAG（仅向量检索，无重排序）
  - bm25_only          仅 BM25 检索
  - hybrid_no_rerank   混合检索无重排序
  - vector_rerank      向量检索 + 重排序

运行方式：
    # 评估单个模式
    python 评估/run_eval.py --mode vector_only

    # 评估所有模式并对比
    python 评估/run_eval.py --compare
"""
import os
import sys
import json
import time
import argparse

# 确保可以导入各模块
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "混合检索"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from 系统日志.config import get_logger
from ragas_config import get_eval_llm, get_eval_embeddings

logger = get_logger(__name__)

# 数据文件路径
EVAL_DATASET_PATH = os.path.join(BASE_DIR, "data", "eval_dataset.json")

# 所有可评估的模式
ALL_MODES = ("full", "vector_only", "bm25_only", "hybrid_no_rerank", "vector_rerank")

# 模式中文名称
MODE_LABELS = {
    "full": "完整RAG(混合+重排序)",
    "vector_only": "基础RAG(仅向量)",
    "bm25_only": "仅BM25",
    "hybrid_no_rerank": "混合检索(无重排序)",
    "vector_rerank": "向量+重排序",
}

# RAGAS 指标
METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
]

METRIC_LABELS = {
    "faithfulness": "忠实度",
    "answer_relevancy": "答案相关性",
    "context_precision": "上下文精确率",
    "context_recall": "上下文召回率",
    "answer_correctness": "答案正确性",
}


def load_eval_dataset(json_path: str) -> list:
    """加载评估数据集（问题 + 标准答案）。"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"评估数据集不存在: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info("加载评估数据集，共 %d 个问题", len(dataset))
    return dataset


# --------------------------------------------------------------------------- #
#  组件构建：按需构建检索组件，避免不必要的模型加载
# --------------------------------------------------------------------------- #
class ComponentBuilder:
    """按需构建并缓存检索组件，多模式共享已构建的组件。"""

    def __init__(self):
        self.segments = None
        self.chroma_coll = None
        self.bm25_retriever = None
        self.hybrid = None
        self.reranker = None

    def get_segments(self):
        if self.segments is None:
            from chroma import load_segments
            data_json = os.path.join(BASE_DIR, "data", "structured_segments.json")
            logger.info("加载结构化片段...")
            self.segments = load_segments(data_json)
        return self.segments

    def get_chroma(self):
        if self.chroma_coll is None:
            from chroma import build_chroma_index
            logger.info("构建 Chroma 向量索引...")
            self.chroma_coll = build_chroma_index(self.get_segments())
        return self.chroma_coll

    def get_bm25(self):
        if self.bm25_retriever is None:
            from bm25 import BM25Retriever
            logger.info("构建 BM25 索引...")
            documents = [seg["content"] for seg in self.get_segments()]
            self.bm25_retriever = BM25Retriever(documents)
        return self.bm25_retriever

    def get_hybrid(self):
        if self.hybrid is None:
            from hybrid_retriever import HybridRetriever
            self.hybrid = HybridRetriever(self.get_chroma(), self.get_bm25())
        return self.hybrid

    def get_reranker(self):
        if self.reranker is None:
            from rerank import Reranker
            logger.info("加载 BGE Reranker 模型...")
            self.reranker = Reranker(backend="bge")
        return self.reranker


def init_qa_system(mode: str, builder: ComponentBuilder = None):
    """
    根据检索模式初始化 QASystem，只构建所需组件。
    """
    from qa_system import QASystem

    if builder is None:
        builder = ComponentBuilder()

    segments = builder.get_segments()

    if mode == "full":
        qa = QASystem(
            hybrid_retriever=builder.get_hybrid(),
            reranker=builder.get_reranker(),
            retrieval_mode=mode,
        )
    elif mode == "vector_only":
        qa = QASystem(
            chroma_collection=builder.get_chroma(),
            retrieval_mode=mode,
        )
    elif mode == "bm25_only":
        qa = QASystem(
            bm25_retriever=builder.get_bm25(),
            segments=segments,
            retrieval_mode=mode,
        )
    elif mode == "hybrid_no_rerank":
        qa = QASystem(
            hybrid_retriever=builder.get_hybrid(),
            retrieval_mode=mode,
        )
    elif mode == "vector_rerank":
        qa = QASystem(
            chroma_collection=builder.get_chroma(),
            reranker=builder.get_reranker(),
            retrieval_mode=mode,
        )
    else:
        raise ValueError(f"未知模式: {mode}")

    logger.info("QASystem 初始化完成 [%s]", mode)
    return qa


# --------------------------------------------------------------------------- #
#  评估流程
# --------------------------------------------------------------------------- #
def run_qa_for_eval(qa, eval_dataset: list) -> list:
    """
    对每个评估问题运行 QA 系统，收集 RAGAS 所需的数据。
    """
    samples = []
    total = len(eval_dataset)

    for i, item in enumerate(eval_dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("[%d/%d] 处理问题: %s", i + 1, total, question[:50])
        start_time = time.time()

        try:
            result = qa.answer(question)
            answer = result.get("answer", "")
            contexts = result.get("retrieved_contexts", [])
        except Exception as e:
            logger.error("问题 '%s' 处理失败，跳过该样本: %s", question[:50], e, exc_info=True)
            continue

        elapsed = time.time() - start_time
        logger.info("  回答耗时: %.1fs, 上下文数: %d", elapsed, len(contexts))

        samples.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    if not samples:
        raise RuntimeError("所有评估问题处理均失败，无有效样本可评估。")
    return samples


def run_ragas_evaluation(samples: list):
    """使用 RAGAS 对收集到的样本进行评估。"""
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )

    hf_dataset = HFDataset.from_list(samples)
    logger.info("HuggingFace Dataset 构建完成，共 %d 条样本", len(hf_dataset))

    logger.info("初始化 RAGAS 评估 LLM 和 Embedding...")
    eval_llm = get_eval_llm()
    eval_embeddings = get_eval_embeddings()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    logger.info("开始 RAGAS 评估（5 个指标，%d 个问题）...", len(samples))
    start_time = time.time()

    result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    elapsed = time.time() - start_time
    logger.info("RAGAS 评估完成，耗时 %.1f 秒", elapsed)
    return result


def get_metric_scores(result_df) -> dict:
    """从 RAGAS 结果 DataFrame 中提取各指标的平均分。"""
    scores = {}
    for metric in METRIC_NAMES:
        if metric in result_df.columns:
            scores[metric] = result_df[metric].mean()
        else:
            scores[metric] = None
    return scores


def print_single_mode_result(result, mode: str):
    """打印单个模式的评估结果并保存 CSV。"""
    result_df = result.to_pandas()
    scores = get_metric_scores(result_df)

    print(f"\n{'=' * 60}")
    print(f"RAGAS 评估结果 [{MODE_LABELS.get(mode, mode)}]")
    print(f"{'=' * 60}")

    print(f"\n{'指标':<20} {'分数':>8}")
    print("-" * 30)
    for metric in METRIC_NAMES:
        score = scores.get(metric)
        label = METRIC_LABELS.get(metric, metric)
        if score is not None:
            print(f"{label:<20} {score:>8.4f}")
    print("-" * 30)

    # 保存到 CSV
    save_path = os.path.join(BASE_DIR, "data", f"eval_results_{mode}.csv")
    save_df = result_df.copy()
    rename_map = {
        "user_input": "question",
        "response": "answer",
        "retrieved_contexts": "contexts",
        "reference": "ground_truth",
    }
    save_df = save_df.rename(
        columns={k: v for k, v in rename_map.items() if k in save_df.columns}
    )
    save_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"评估结果已保存至: {save_path}")


def print_comparison_table(all_scores: dict):
    """打印所有模式的对比表。"""
    print(f"\n{'=' * 80}")
    print("多模式评估对比")
    print(f"{'=' * 80}")

    # 表头
    header = f"{'模式':<28}"
    for metric in METRIC_NAMES:
        label = METRIC_LABELS.get(metric, metric)
        header += f" {label:>10}"
    print(header)
    print("-" * 80)

    # 每行一个模式
    for mode in ALL_MODES:
        if mode not in all_scores:
            continue
        scores = all_scores[mode]
        label = MODE_LABELS.get(mode, mode)
        row = f"{label:<28}"
        for metric in METRIC_NAMES:
            score = scores.get(metric)
            if score is not None:
                row += f" {score:>10.4f}"
            else:
                row += f" {'N/A':>10}"
        print(row)
    print("-" * 80)

    # 找出每个指标的最佳模式
    print("\n各指标最佳模式：")
    for metric in METRIC_NAMES:
        best_mode = None
        best_score = -1
        for mode, scores in all_scores.items():
            s = scores.get(metric)
            if s is not None and s > best_score:
                best_score = s
                best_mode = mode
        if best_mode:
            label = METRIC_LABELS.get(metric, metric)
            mode_label = MODE_LABELS.get(best_mode, best_mode)
            print(f"  {label:<16} {mode_label} ({best_score:.4f})")
    print("=" * 80)


def run_single_mode(mode: str, eval_dataset: list, builder: ComponentBuilder = None):
    """运行单个模式的完整评估流程，返回指标分数字典。"""
    if builder is None:
        builder = ComponentBuilder()

    print(f"\n{'=' * 60}")
    print(f"评估模式: {MODE_LABELS.get(mode, mode)}")
    print(f"{'=' * 60}")

    # 1. 初始化 QA 系统
    qa = init_qa_system(mode, builder)

    # 2. 运行 QA 系统
    print(f"\n阶段 1/2：运行 QA 系统 [{mode}]")
    print("-" * 40)
    samples = run_qa_for_eval(qa, eval_dataset)

    # 3. 运行 RAGAS 评估
    print(f"\n阶段 2/2：运行 RAGAS 评估 [{mode}]")
    print("-" * 40)
    result = run_ragas_evaluation(samples)

    # 4. 打印并保存结果
    print_single_mode_result(result, mode)

    # 5. 返回指标分数
    result_df = result.to_pandas()
    return get_metric_scores(result_df)


def main():
    """评估主流程。"""
    parser = argparse.ArgumentParser(description="RAGAS 评估脚本")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=ALL_MODES,
        help="检索模式（默认: full）",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="运行所有模式并输出对比表",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAGAS 评估流程启动")
    print("=" * 60)

    # 加载评估数据集
    eval_dataset = load_eval_dataset(EVAL_DATASET_PATH)

    if args.compare:
        # 对比模式：运行所有模式
        builder = ComponentBuilder()
        all_scores = {}
        for mode in ALL_MODES:
            try:
                scores = run_single_mode(mode, eval_dataset, builder)
                all_scores[mode] = scores
            except Exception as e:
                logger.error("模式 %s 评估失败: %s", mode, e, exc_info=True)
                print(f"\n[跳过] 模式 {mode} 评估失败: {e}")

        # 输出对比表
        if all_scores:
            print_comparison_table(all_scores)
    else:
        # 单模式评估
        run_single_mode(args.mode, eval_dataset)


if __name__ == "__main__":
    main()
