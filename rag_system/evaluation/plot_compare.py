# plot_compare.py
"""从 eval_results_*.csv 生成 RAGAS 多模式评估对比图（docs/eval_compare.png）。

用法：
    python -m rag_system.evaluation.plot_compare          # 从 data/ 读最新 CSV
    python -m rag_system.evaluation.plot_compare --output docs/eval_compare.png
"""
import os
import sys
import csv
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Windows 控制台 UTF-8 兼容
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(ROOT_DIR, "rag_system", "data")

# 中文字体（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

MODES = [
    ("full", "完整RAG\n(混合+重排序)"),
    ("vector_rerank", "向量+重排序"),
    ("hybrid_no_rerank", "混合检索\n(无重排序)"),
    ("vector_only", "基础RAG\n(仅向量)"),
    ("bm25_only", "仅BM25"),
]
METRICS = [
    ("faithfulness", "忠实度"),
    ("answer_relevancy", "答案相关性"),
    ("context_precision", "上下文精确率"),
    ("context_recall", "上下文召回率"),
    ("answer_correctness", "答案正确性"),
]


def load_mean(mode: str, metric: str) -> float:
    path = os.path.join(DATA_DIR, f"eval_results_{mode}.csv")
    vals = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = row.get(metric, "")
            if v not in ("", None, "nan"):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
    if not vals:
        raise RuntimeError(f"{path} 中无有效 {metric} 数据")
    return sum(vals) / len(vals)


def main():
    parser = argparse.ArgumentParser(description="生成 RAGAS 多模式评估对比图")
    parser.add_argument("--output", default=os.path.join(ROOT_DIR, "docs", "eval_compare.png"))
    args = parser.parse_args()

    # 数据：modes × metrics 均值矩阵
    data = np.array([[load_mean(m, k) for k, _ in METRICS] for m, _ in MODES])
    labels = [label for _, label in MODES]
    metric_names = [name for _, name in METRICS]
    n_modes, n_metrics = data.shape

    # 每个指标的最高分（用于加粗/标注）
    best = data.argmax(axis=0)

    x = np.arange(n_metrics)
    width = 0.8 / n_modes
    colors = ["#4a6fa5", "#3e8fa8", "#94a3b8", "#b06a2c", "#c2585f"]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    for i, label in enumerate(labels):
        bars = ax.bar(x + (i - (n_modes - 1) / 2) * width, data[i], width,
                      label=label.replace("\n", ""), color=colors[i], edgecolor="white", linewidth=0.6)
        for j, b in enumerate(bars):
            v = b.get_height()
            is_best = best[j] == i
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom",
                    fontsize=8.5 if not is_best else 9.5,
                    fontweight="bold" if is_best else "normal",
                    color=colors[i] if is_best else "#5b6172")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("RAGAS 分数", fontsize=11)
    ax.set_title("RAGAS 多模式检索评估对比（重排序修复后 · 2026-09）", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=5, fontsize=9, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 每指标最佳模式标注在图例下方说明
    best_line = " ｜ ".join(
        f"{metric_names[j]}最佳：{labels[best[j]].replace(chr(10), '')}（{data[best[j], j]:.4f}）"
        for j in range(n_metrics)
    )
    fig.text(0.5, 0.005, best_line, ha="center", fontsize=8.5, color="#5b6172")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    print(f"已生成：{args.output}")

    # 终端同步输出均值表，便于核对
    print("\n各模式指标均值（与 CSV 核算一致）：")
    header = "模式".ljust(22) + "".join(n.ljust(12) for n in metric_names)
    print(header)
    print("-" * len(header))
    for i, label in enumerate(labels):
        print(label.replace("\n", "").ljust(24) + "".join(f"{v:.4f}".ljust(12) for v in data[i]))


if __name__ == "__main__":
    main()
