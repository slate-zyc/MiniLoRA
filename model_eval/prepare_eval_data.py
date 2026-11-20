"""
整理 data/medical/test.jsonl，生成调试集与正式测试集。

运行方式:
    python model_eval/prepare_eval_data.py
"""

import argparse
import json
import random

from check_data import EVAL_DIR, load_jsonl, normalize_text     # 父路径, jsonl-> list[dict], 文本正则化


# ============================================================
# 整理原始字段：input → question，output → reference_answer
# ============================================================

def clean_text(text: str) -> str:
    """统一换行符，并去掉首尾的空白。"""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def prepare_data(rows: list[dict]) -> list[dict]:
    """保留有效问答，文字相同的问题只保留第一次出现的问答。

    返回：
        [
            {"question": XXX, "reference_answer": XXX},
            {"question": XXX, "reference_answer": XXX},
            ....
        ]
    """
    items = []
    seen = set()
    for row in rows:
        question = clean_text(row["input"])
        reference = clean_text(row["output"])
        key = normalize_text(question)  # 去标点和空白后的文本只用于去重
        if not key or not reference or key in seen:
            continue
        seen.add(key)
        items.append({"question": question, "reference_answer": reference})
    return items


# ============================================================
# 主函数：读取数据 → 固定顺序打乱 → 分别保存
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="从现有医疗测试集划分调试集和正式测试集")
    parser.add_argument("--dev-size", type=int, default=20, help="调试集题数")
    parser.add_argument("--test-size", type=int, default=100, help="正式测试集题数")
    parser.add_argument("--seed", type=int, default=20261119, help="划分种子")
    args = parser.parse_args()

    source = EVAL_DIR.parent / "data" / "medical" / "test.jsonl"
    rows = load_jsonl(source)
    items = prepare_data(rows)
    if min(args.dev_size, args.test_size) < 1 or args.dev_size + args.test_size > len(items):
        parser.error(f"调试集和正式集题数都需大于 0，合计最多 {len(items)} 题。")

    # 一个固定种子的打乱顺序，调试题和正式题取互不重叠的区间。
    random.Random(args.seed).shuffle(items)
    test_end = args.dev_size + args.test_size
    splits = {
        "dev_set.jsonl": items[:args.dev_size],
        "test_set.jsonl": items[args.dev_size:test_end],
    }
    for name, questions in splits.items():
        # open() 只创建文件，不会自动创建目录，这里父目录已经创建，不需要 (EVAL_DIR / name).parent.mkdir(parents=True，exist_ok=True)
        with (EVAL_DIR / name).open("w", encoding="utf-8", newline="\n") as f:  # newline=None 会启用“通用换行符”模式（Universal Newlines），在 Windows 上：\n 会被翻译为 \r\n
            for row in questions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(questions)} 题")
    print(f"原始 {len(rows)} 条，清洗后 {len(items)} 条。")


if __name__ == "__main__":
    main()
