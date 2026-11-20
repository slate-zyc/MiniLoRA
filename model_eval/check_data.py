"""
检查两份评测题集的字段及题目是否重叠。

运行方式:
    python model_eval/check_data.py
"""

import hashlib          # 计算内容指纹
import json
import re
import unicodedata      # Unicode 字符数据库（UCD）
from pathlib import Path


# ============================================================
# 数据读取与文本标准化
# ============================================================

EVAL_DIR = Path(__file__).resolve().parent
FIELDS = {"question", "reference_answer"}


def load_jsonl(path: Path) -> list[dict]:
    """逐行读取 JSONL，忽略空行。"""
    with path.open("r", encoding = "utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_hash(path: Path) -> str:
    """记录原始文件的 SHA-256，用于关联运行结果与数据。

    hashlib.file_digest(...) 是 Python 3.11 新增的便捷 API。
    - 它会从传入的二进制文件对象 f 中读取数据，计算指定算法的摘要。
    - "sha256" 指定使用 SHA-256。
    - 它通常会自动分块读取文件，因此不需要一次性把整个文件读入内存，适合大文件。

    .hexdigest()
    - digest() 返回原始二进制摘要，例如 bytes。
    - hexdigest() 返回十六进制字符串形式，便于保存到日志、JSON、数据库。
    - SHA-256 的 hexdigest() 固定为 64 个小写十六进制字符
    """
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def normalize_text(text: str) -> str:
    """统一全半角，并去掉标点和空白，便于检查重复。

    最终保留的内容
    - Unicode 字母（包括中文、日文、韩文、西里尔文等）；
    - Unicode 数字；
    - 其他一律删除：空格、换行、制表符、标点、符号、下划线、Emoji、零宽字符等。

    先用 NFKC(Normalization Form KC) 统一全半角和兼容字符，再用 casefold 统一大小写(比 str.lower() 更彻底)，最后用 re.sub() 删除标点、空白和下划线，只保留 Unicode 字母和数字。
    优点是 简洁、支持中文、能抹平常见写法差异；缺点是有损归一化，可能把 "1.5" 和 "15"、"A-B" 和 "AB" 误判为重复，且结果可能为空，不适合展示文本或精确语义比较。

    \w：匹配“单词字符”，包括 Unicode 字母、数字和下划线；
    - re 对 str 默认使用 Unicode 模式，所以 \w 不只是 [a-zA-Z0-9_]，而是包含中文、日文、西里尔字母等 Unicode 字母和数字。
    \W：\w 的补集，匹配任何不是单词字符的字符。
    因为 \w 本身包含下划线 _，所以 \W 不会匹配下划线。要删除下划线，必须显式把它加进字符类。
    + 表示前一个字符或子表达式重复 1 次或多次，这里是为了把连续多个要删除的字符合并成一次替换，减少替换次数，也避免生成中间空串。
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", text)


# ============================================================
# 主函数：检查字段、集内重复和跨集重合
# ============================================================

def main():
    dev = load_jsonl(EVAL_DIR / "dev_set.jsonl")
    test = load_jsonl(EVAL_DIR / "test_set.jsonl")

    # 防止 缺少字段或字段内容为空
    for rows in (dev, test):
        assert all(set(row) == FIELDS and all(row[key].strip() for key in FIELDS) for row in rows), "题目字段不完整"

    # 防止 源测试集有重复的题目（一般不会，可省略）
    dev_questions = {normalize_text(row["question"]) for row in dev}
    test_questions = {normalize_text(row["question"]) for row in test}
    assert len(dev_questions) == len(dev) and len(test_questions) == len(test), "发现集内重复题"
    assert not dev_questions & test_questions, "调试集与正式集有重复题"        # & 表示集合交集

    # 防止 源测试集与 训练集或验证集 有重复题目（一般不会，可省略）
    medical_dir = EVAL_DIR.parent / "data" / "medical"
    known_questions = set()
    for name in ("train.jsonl", "valid.jsonl"):
        rows = load_jsonl(medical_dir / name)
        known_questions.update(normalize_text(row["input"]) for row in rows)
    overlap = (dev_questions | test_questions) & known_questions        # `|` 是并集，`&` 是交集

    print(f"调试集 {len(dev)} 题，正式集 {len(test)} 题，集内和跨集均无文字重复。")
    print(f"与训练集或验证集的标准化文字重合: {len(overlap)} 题。")


if __name__ == "__main__":
    main()
