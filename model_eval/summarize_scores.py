"""
读取已有逐题评分，汇总模型表现并计算配对 Bootstrap 区间。

运行方式:
    $run = "model_eval/results/test_all_models/temp0.3_length256_repeat1.1"
    python model_eval/summarize_scores.py --scores-file "$run/review_scores.jsonl" --output-file "$run/score_summary.md"
"""

import argparse
import random
from collections import Counter
from itertools import combinations          # 从一组元素中取出所有指定大小的组合
from pathlib import Path
from statistics import mean, quantiles

from check_data import load_jsonl


MODEL_LABELS = {
    "base": "Base",
    "original_lora": "Original LoRA",
    "rank8_5000": "Rank8-5000",
}
SCORE_FIELDS = ("factuality", "coverage", "repetition", "readability")
BOOTSTRAP_REPEATS = 10_000  # 每对模型重复抽样次数
BOOTSTRAP_SEED = 20261122  # 固定种子以复现区间


# ============================================================
# 配对 Bootstrap：按题目重抽同一对模型的分差
# ============================================================

def paired_bootstrap(differences: list[int]) -> tuple[float, float]:
    """返回平均配对分差的 95% 百分位区间。

    输入各样本 模型分数配对差，输出百分位法 Bootstrap 置信区间
    """
    rng = random.Random(BOOTSTRAP_SEED)
    sample_size = len(differences)  # 每次抽取与共同题数相同的题目
    bootstrap_means = [
        mean(rng.choices(differences, k=sample_size))   # 等概率、有放回抽样，放回列表
        for _ in range(BOOTSTRAP_REPEATS)
    ]
    # 40 等分的第一个和最后一个分位点分别对应 2.5% 和 97.5%
    percentiles = quantiles(bootstrap_means, n=40, method="inclusive")      # n=40控制分位数边界；inclusive 使用简单线性插值计算分位数
    return percentiles[0], percentiles[-1]


# ============================================================
# 汇总各模型的单项得分、总分和严重错误
# ============================================================

def summarize_scores(records: list[dict]) -> str:
    """接收读取后的评分，返回各汇总结果，方便直接转为 md 文件的格式"""
    rows_by_model = {model: {} for model in MODEL_LABELS}
    for row in records:
        # model, question_idx = row["model"], row["test_row"]
        # rows = rows_by_model[model]
        # rows[question_idx] = row
        rows_by_model[row["model"]][row["test_row"]] = row
        # rows_by_model:
        # {
        #   "base": {
        #               1: {"test_row": 1, "model": "base", "answer_line": 1, "factuality": 0, "coverage": 1, "repetition": 2, "readability": 2, "severe_error": null, "finish_reason": "eos", "notes": "把部分一线药主要定位为耐药治疗，动物处置条件待核"},
        #               2: {"test_row": 2, "model": "base", "answer_line": 2, "factuality": 2, "coverage": 2, "repetition": 2, "readability": 2, "severe_error": 0, "finish_reason": "eos", "notes": "说明脑震荡、颅内出血与查体影像，指出急症"},
        #               ...
        #           },
        #    ...
        # }

    # 准备写入 md 文件框架
    metrics = [
        "## 各项均分与有效评分数",
        "",
        "每格为均分（有效评分数）。各项满分 2 分，待复核的空值不参与均分。",
        "",
        "| 模型 | 回答数 | 事实正确性 | 要点覆盖 | 重复控制 | 格式可读性 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    distribution = [
        "## 总分分布",
        "",
        "每行统计该模型四项均有分值的回答，四项等权相加，总分为 0–8 分。",
        "",
        "| 模型 | 有效总分数 | 0 分 | 1 分 | 2 分 | 3 分 | 4 分 | 5 分 | 6 分 | 7 分 | 8 分 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    severity = [
        "## 严重错误",
        "",
        "| 模型 | 已标记 | 已判定 | 待复核 | 已标记 / 全部回答 | 已标记 / 已判定 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    totals_by_model = {}    # 汇总分数，用于后续 Bootstrap
    for model, label in MODEL_LABELS.items():   # 逐模型统计
        # # 各项均分与有效评分数
        rows = rows_by_model[model]     # value 为一个模型的 100 条记录
        metrics_cells = []
        for field in SCORE_FIELDS:      # 逐指标统计
            values = [row[field] for row in rows.values() if row[field] is not None]    # 一个模型一个指标的所有分数（只统计有分数的）
            metrics_cells.append(f"{mean(values):.2f}（{len(values)}）" if values else "—（0）")
        metrics.append(f"| {label} | {len(rows)} | " + " | ".join(metrics_cells) + " |")

        # 总分分布
        totals = {}     # 记录每个问题的总分（只统计四个指标全有的样本）
        for question_idx, row in rows.items():
            scores = [row[field] for field in SCORE_FIELDS]     # 一个样本的四个分数
            if None not in scores:      # 只统计四个指标全有的样本
                totals[question_idx] = sum(scores)
        totals_by_model[model] = totals
        counts = Counter(totals.values())       # {0: 0, 1: 0, ...}     # key 为总分（0~8），value 为总分为这个分数的样本数
        bins = " | ".join(str(counts[score]) for score in range(9))
        distribution.append(f"| {label} | {len(totals)} | {bins} |")

        # 严重错误统计
        errors = Counter(row["severe_error"] for row in rows.values())      # key 为 0、1、None   json.loads() 读取 JSON 时会把 `null` 转成 `None`
        flagged = errors[1]                 # 已标记为严重错误
        decided = errors[0] + flagged       # 已判定（可判定是否有严重错误）
        all_rate = f"{flagged / len(rows):.1%}" if rows else "—"            # 已标记 / 全部回答
        decided_rate = f"{flagged / decided:.1%}" if decided else "—"       # 已标记 / 已判定
        severity.append(f"| {label} | {flagged} | {decided} | {errors[None]} | {all_rate} | {decided_rate} |")

    paired = compare_models(totals_by_model)    # 自定义函数：三模型共同可评分题的配对结果 + 配对 Bootstrap 置信区间
    sections = [["# 正式评测评分汇总"], metrics, distribution, severity, paired]    # 转为 md 的结果
    return "\n\n".join("\n".join(section) for section in sections) + "\n"   # 各行用 \n 拼起来，再用 \n\n 连接各节，最后给整个结果加一个换行


# ============================================================
# 仅在三个模型共同可评分的题目上比较总分 + 配对 Bootstrap 置信区间
# ============================================================

def compare_models(totals: dict[str, dict[int, int]]) -> list[str]:
    """输入:
    totals_by_model:
        {
          "base": {
                      1: 5,
                      2: 8,
                      ...
                  },
           ...
        }
    输出：三模型共同可评分题的配对结果 和 配对 Bootstrap 置信区间，便于直接转为 md 的形式
    """
    common = sorted(set.intersection(*(set(scores) for scores in totals.values())))     # 45 道题的题号
    # set.intersection() 是 Python 集合（set）的交集方法
    # set(scores) 取字典的键（题号）
    # *sets：将 sets 中的元素解包，作为独立的位置参数传入函数。
    # 等价于 set.intersection(A, B, C)，也等价于 A.intersection(B, C)
    lines = ["## 三模型共同可评分题的配对结果", "", f"共同题数为 {len(common)}。", ""]
    if not common:
        return lines + ["没有共同可评分题，无法进行配对比较。"]

    lines.extend([
        "胜、平、负均以前一个模型为基准。",
        "",
        "| 比较 | 胜 | 平 | 负 |",
        "|---|---:|---:|---:|",
    ])
    intervals = [
        "",
        "## 配对 Bootstrap 置信区间",
        "",
        f"分差为前一个模型总分减去后一个模型总分。每次有放回抽取 {len(common)} 道题"
        f"并计算平均分差，重复 {BOOTSTRAP_REPEATS:,} 次，随机种子为 {BOOTSTRAP_SEED}。"
        "取平均分差分布的 2.5% 和 97.5% 分位数作为 95% CI。",
        "",
        "| 比较 | 平均配对差值 | 95% CI |",
        "|---|---:|---:|",
    ]
    for first, second in combinations(MODEL_LABELS, 2):     # 两两配对
        comparison = f"{MODEL_LABELS[first]} - {MODEL_LABELS[second]}"
        differences = [totals[first][question_idx] - totals[second][question_idx] for question_idx in common]
        wins = sum(difference > 0 for difference in differences)    # 胜
        ties = differences.count(0)     # 平
        lines.append(f"| {comparison} | {wins} | {ties} | {len(common) - wins - ties} |")
        lower, upper = paired_bootstrap(differences)    # 百分位法 Bootstrap 置信区间
        intervals.append(f"| {comparison} | {mean(differences):.3f} | [{lower:.3f}, {upper:.3f}] |")

    return lines + intervals + [
        "",
        "区间若包含 0 ，则现有数据不足以判断分差方向。区间解释受共同题数和 AI 评分可靠性等的限制。",
    ]


# ============================================================
# 主函数：读入评分文件，输出 Markdown
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="汇总已有逐题评分并计算配对 Bootstrap 区间")
    parser.add_argument("--scores-file", type=Path, required=True, help="逐题评分 JSONL 文件")
    parser.add_argument("--output-file", type=Path, default="model_eval/results/test_all_models/temp0.3_length256_repeat1.1/score_summary.md",
                        help="将汇总结果另存为 Markdown")
    args = parser.parse_args()

    report = summarize_scores(load_jsonl(args.scores_file))
    print(report, end="")
    if args.output_file:
        args.output_file.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
