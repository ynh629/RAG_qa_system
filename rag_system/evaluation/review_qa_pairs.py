# review_qa_pairs.py
"""
Phase 2：人工审查 LLM 生成的 QA 对，产出最终评估数据集。

两种审查方式：

A. 交互式终端审查（推荐，适合 50~80 条规模）
    python -m rag_system.evaluation.review_qa_pairs

B. CSV 批量审查（用 Excel 打开编辑后再导入）
    # 导出 CSV
    python -m rag_system.evaluation.review_qa_pairs --export-csv
    # 在 Excel 中增删改后保存，再导入
    python -m rag_system.evaluation.review_qa_pairs --import-csv

最终结果保存为 data/qa_pairs_final.json（带 id 字段，可直接用于评测）。
"""
import argparse
import csv
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW = os.path.join(BASE_DIR, "data", "qa_pairs_raw.json")
DEFAULT_FINAL = os.path.join(BASE_DIR, "data", "qa_pairs_final.json")
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "qa_pairs_review.csv")

QUESTION_TYPES = ["数值提取", "事实检索", "归纳概括", "风险分析", "对比分析"]
DIFFICULTIES = ["easy", "medium", "hard"]


def load_pairs(path: str) -> list:
    """加载 QA 对 JSON。"""
    if not os.path.exists(path):
        print(f"错误: 文件不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pairs(pairs: list, path: str) -> None:
    """保存 QA 对 JSON（自动补齐 id 字段）。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    for idx, p in enumerate(pairs):
        p["id"] = f"qa_{idx + 1:03d}"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"\n已保存 {len(pairs)} 条 → {path}")


def print_pair(idx: int, total: int, p: dict) -> None:
    """格式化展示一条 QA 对。"""
    print("\n" + "=" * 60)
    print(f"[{idx + 1}/{total}] 类型: {p.get('question_type', '?')} | 难度: {p.get('difficulty', '?')}")
    print(f"Q: {p['question']}")
    print(f"A: {p['ground_truth']}")
    src_title = " > ".join(p.get("source_title_path", [])) or "（无标题）"
    page = p.get("source_page")
    print(f"来源: {src_title}" + (f" / 第{page}页" if page else ""))
    src_sentence = p.get("source_sentence") or ""
    if src_sentence:
        preview = src_sentence[:120] + "..." if len(src_sentence) > 120 else src_sentence
        print(f"原文依据: {preview}")


def review_interactive(pairs: list, target: int) -> list:
    """交互式逐条审查。返回保留的 QA 对列表。"""
    final = []
    i = 0
    total = len(pairs)
    print("\n交互审查开始。命令说明：")
    print("  [y] 保留  [e] 编辑  [d] 删除  [t] 改类型  [g] 改难度  [q] 退出")
    while i < total:
        p = pairs[i]
        print_pair(i, total, p)
        remaining = max(0, target - len(final))
        print(f"\n已保留 {len(final)} 条，目标 {target} 条，还需 {remaining} 条")
        cmd = input("操作 [y/e/d/t/g/q]: ").strip().lower()
        if cmd in ("", "y", "yes"):
            final.append(p)
            i += 1
        elif cmd == "e":
            print("（直接回车 = 保持不变）")
            new_q = input("新问题: ").strip()
            new_a = input("新答案: ").strip()
            if new_q:
                p["question"] = new_q
            if new_a:
                p["ground_truth"] = new_a
            final.append(p)
            print("已编辑并保留。")
            i += 1
        elif cmd == "d":
            print("已删除。")
            i += 1
        elif cmd == "t":
            print(f"可选类型: {QUESTION_TYPES}")
            new_t = input("新类型: ").strip()
            if new_t in QUESTION_TYPES:
                p["question_type"] = new_t
                print(f"类型已改为: {new_t}")
            else:
                print("类型无效，未修改。")
        elif cmd == "g":
            print(f"可选难度: {DIFFICULTIES}")
            new_d = input("新难度: ").strip()
            if new_d in DIFFICULTIES:
                p["difficulty"] = new_d
                print(f"难度已改为: {new_d}")
            else:
                print("难度无效，未修改。")
        elif cmd in ("q", "quit", "exit"):
            print("退出审查。")
            break
        else:
            print("未知命令，请输入 y/e/d/t/g/q")
    return final


def export_csv(pairs: list, path: str) -> None:
    """导出为 CSV（Excel 可打开编辑）。"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "ground_truth", "question_type", "difficulty",
                         "source_sentence", "source_chunk_id", "source_page"])
        for idx, p in enumerate(pairs):
            writer.writerow([
                f"qa_{idx + 1:03d}",
                p.get("question", ""),
                p.get("ground_truth", ""),
                p.get("question_type", ""),
                p.get("difficulty", ""),
                p.get("source_sentence", ""),
                p.get("source_chunk_id", ""),
                p.get("source_page", ""),
            ])
    print(f"已导出 {len(pairs)} 条 → {path}")
    print("提示: 在 Excel 中编辑后保存，再运行 --import-csv 导入。删除行 = 删除该 QA 对。")


def import_csv(path: str, raw_path: str = DEFAULT_RAW) -> list:
    """从 CSV 导入为最终 QA 对，并从 raw 文件回填 source_title_path / source_page。"""
    # 加载 raw 文件，建立 id / question 两个索引，用于回填 CSV 中丢失的元数据。
    # 优先按 id 匹配（导出时 id 为 qa_001...，对应 raw 的列表顺序），
    # 这样即使在 Excel 中修改过 question 也能回填来源信息。
    raw_index = {}
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            for i, rp in enumerate(json.load(f)):
                raw_index[rp.get("question", "")] = rp
                raw_index.setdefault(f"qa_{i + 1:03d}", rp)

    pairs = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("question") or not row.get("ground_truth"):
                continue
            raw_meta = raw_index.get((row.get("id") or "").strip(), {})
            if not raw_meta:
                raw_meta = raw_index.get(row["question"], {})
            pairs.append({
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "question_type": row.get("question_type") or "事实检索",
                "difficulty": row.get("difficulty") or "medium",
                "source_sentence": row.get("source_sentence", ""),
                "source_chunk_id": raw_meta.get("source_chunk_id", row.get("source_chunk_id")),
                "source_title_path": raw_meta.get("source_title_path", []),
                "source_page": raw_meta.get("source_page", row.get("source_page")),
            })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Phase 2：人工审查 QA 对")
    parser.add_argument("--raw", default=DEFAULT_RAW, help=f"原始候选 QA 对 JSON（默认: {DEFAULT_RAW}）")
    parser.add_argument("--final", default=DEFAULT_FINAL, help=f"最终输出 JSON（默认: {DEFAULT_FINAL}）")
    parser.add_argument("--csv", default=DEFAULT_CSV, help=f"审查用 CSV（默认: {DEFAULT_CSV}）")
    parser.add_argument("--target", type=int, default=50, help="目标条数（默认 50，仅用于进度提示）")
    parser.add_argument("--export-csv", action="store_true", help="导出 CSV 供 Excel 编辑")
    parser.add_argument("--import-csv", action="store_true", help="从 CSV 导入并生成最终数据集")
    args = parser.parse_args()

    if args.export_csv:
        pairs = load_pairs(args.raw)
        export_csv(pairs, args.csv)
        return

    if args.import_csv:
        pairs = import_csv(args.csv, raw_path=args.raw)
        if not pairs:
            print("错误: CSV 中没有有效数据")
            sys.exit(1)
        print(f"从 CSV 导入 {len(pairs)} 条")
        save_pairs(pairs, args.final)
        return

    # 默认：交互式审查
    pairs = load_pairs(args.raw)
    print(f"加载候选 QA 对 {len(pairs)} 条（目标 {args.target} 条）")
    final = review_interactive(pairs, args.target)
    if final:
        save_pairs(final, args.final)
        from collections import Counter
        print(f"类型分布: {dict(Counter(p['question_type'] for p in final))}")
        print(f"难度分布: {dict(Counter(p['difficulty'] for p in final))}")
    else:
        print("未保留任何 QA 对，未生成最终文件。")


if __name__ == "__main__":
    main()

