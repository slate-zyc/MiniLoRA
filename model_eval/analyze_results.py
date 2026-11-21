"""
汇总调试集生成结果，比较重复和截断。

运行方式:
    python model_eval/analyze_results.py --run-dir `
        model_eval/results/dev_original_lora/greedy_length256_repeat1.1 `
        model_eval/results/dev_original_lora/greedy_length512_repeat1.1 `
        model_eval/results/dev_original_lora/greedy_length256_repeat1.0 `
        model_eval/results/dev_original_lora/temp0.3_length256_repeat1.1 `
        model_eval/results/dev_original_lora/temp0.7_length256_repeat1.1

    $run = "model_eval/results/test_all_models/temp0.3_length256_repeat1.1"
    python model_eval/analyze_results.py --run-dir $run --output-file "$run/test_comparison.md"
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from check_data import load_jsonl, normalize_text


# ============================================================
# 统计重复片段
# ============================================================

def repeat_ratio(response: str, n: int) -> float:
    """重复比例 = 重复出现的片段数 / 全部片段数。"""
    text = normalize_text(response)
    grams = [text[i:i + n] for i in range(len(text) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


# ============================================================
# 主函数：读取多个运行目录，输出对照表
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="比较不同生成配置的重复与截断")
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True, help="生成结果目录，可指定多个")      # nargs="+"：至少接收一个值，可以接收多个值；不设置时默认介绍一个值
    parser.add_argument("--output-file", type=Path, default="model_eval/results/dev_original_lora/dev_comparison.md",
                        help="将对照表另存为 Markdown")
    args = parser.parse_args()

    lines = [
        "| 运行 | 模型 | 长度上限 | 惩罚 | 解码方式 | 回答数 | 截断 | 平均重复比例 | 平均输出 tokens |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for directory in args.run_dir:
        info = json.loads((directory / "run_info.json").read_text(encoding="utf-8"))
        if info["status"] != "complete":
            parser.error(f"运行尚未完成: {directory}")
        config = info["config"]
        gen = config["generation"]
        mode = f"采样 {gen['temperature']}" if gen["do_sample"] else "贪心"
        answers = load_jsonl(directory / "answers.jsonl")
        for spec in config["models"]:
            model = spec["name"]
            rows = [row for row in answers if row["model"] == model]
            truncated = sum(row["finish_reason"] == "length" for row in rows)
            n = config["repeat_ngram_size"]     # 连续四字片段用于发现反复出现的文字
            repetition = mean(repeat_ratio(row["response"], n) for row in rows)     # 平均重复比例
            average_tokens = mean(row["output_tokens"] for row in rows)             # 平均输出 tokens 长度
            lines.append(
                f"| {directory.name} | {model} | {gen['max_new_tokens']} | "
                f"{gen['repetition_penalty']} | {mode} | {len(rows)} | "
                f"{truncated}/{len(rows)} | {repetition:.1%} | {average_tokens:.1f} |"
            )
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.output_file:
        args.output_file.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
