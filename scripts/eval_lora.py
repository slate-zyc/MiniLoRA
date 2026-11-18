"""
模块 6：批量评测 eval_lora.py

目标：批量跑 base vs LoRA 对比，保存结果到 results/base_vs_lora_old.jsonl

运行方式：
    python scripts/eval_lora.py
"""

import argparse
import gc       # Python 内置的 垃圾回收（Garbage Collection） 模块
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer_compare import load_model, generate
from train_lora import load_jsonl
from prepare_medical_sft import write_jsonl


# SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"
#
#
# # def load_model(model_name, adapter_dir=None):
#     """加载模型（复用模块 5）
#
#     adapter_dir=None  → 加载 base 模型
#     adapter_dir="路径" → 加载 base + LoRA adapter
#     """
#     pass
#
#
# def generate(tokenizer, model, question, max_new_tokens=256):
#     """用模型生成回答（复用模块 5）"""
#     pass
#
#
# def load_jsonl(path):
#     """读取 jsonl 文件，返回 list[dict]"""
#     pass
#
#
# def write_jsonl(path, items):
#     """写出 jsonl 文件，ensure_ascii=False 保证中文不被转义"""
#     pass


def main():
    parser = argparse.ArgumentParser(description="批量评测 base vs LoRA")
    parser.add_argument("--model-name", type=str, default=r"D:\LLMProjects\LLMmodels\Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/medical/eval_prompts.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("results/base_vs_lora.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # 步骤 1 - 读取评测问题
    # 步骤 2 - 加载 base 模型，对所有问题生成回答
    #   用 list 存结果: [{"question": ..., "base_answer": ...}, ...]
    # 步骤 3 - 释放显存（del + gc.collect + empty_cache）
    # 步骤 4 - 加载 LoRA 模型，对所有问题生成回答
    #   追加到已有结果: result[idx]["lora_answer"] = ...
    # 步骤 5 - 用 write_jsonl 保存结果到 jsonl

    # ================= 步骤 1：读取评测问题 =================
    print("=" * 60)
    print("Base vs LoRA 推理对比")
    print("=" * 60)

    if not args.eval_file.exists():
        print(f"⚠️ 评测文件不存在: {args.eval_file}")
        print("请先创建 data/medical/eval_prompts.jsonl")
        print('每行一个 JSON: {"question": "你的问题"}')
        return

    print("\n读取评测问题")
    questions = load_jsonl(args.eval_file)
    print(f"  问题加载成功，共 {len(questions)} 个问题")

    # 准备一个列表存储所有结果
    results = []

    # ================= 步骤 2：Base 模型推理 =================
    print("[1/2] 加载 Base 模型...")
    tokenizer_base, model_base = load_model(args.model_name, adapter_dir=None)
    print("  Base 模型加载完成\n")

    print("  Base 模型回答中...")
    for idx, item in enumerate(questions):
        question = item["question"]
        base_response = generate(tokenizer_base, model_base, question, args.max_new_tokens)
        #  将问题和 base 的回答存入字典，添加到结果列表
        results.append(
            {
                "question": question,
                "base_response": base_response,
            }
        )
        print(f"  进度: {idx + 1}/{len(questions)}")

    # ================= 步骤 3：释放显存 =================
    print("\n释放 Base 模型显存...")
    del model_base
    del tokenizer_base  # del 只是删除了 Python 的变量名，并不会立刻释放底层真实的 GPU 显存
    gc.collect()        # 手动强制启动 Python 的垃圾回收机制 (Garbage Collector)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()        # 清空 PyTorch 的 GPU 缓存
    print("  显存已释放\n")

    # ================= 步骤 4：LoRA 模型推理 =================
    print("[2/2] 加载 LoRA 模型...")
    if not args.adapter_dir.exists():
        print(f"  ⚠️ LoRA adapter 不存在: {args.adapter_dir}")
        print("  请先运行训练脚本生成 LoRA adapter")
        return
    tokenizer_lora, model_lora = load_model(args.model_name, args.adapter_dir)
    print("  LoRA 模型加载完成\n")

    print("  LoRA 模型回答:")
    for idx, item in enumerate(questions):
        question = item["question"]
        lora_response = generate(tokenizer_lora, model_lora, question, args.max_new_tokens)
        #  将 LoRA 的回答直接追加到对应的字典中
        results[idx]["lora_response"] = lora_response
        results[idx]["train_samples"] = 5000
        results[idx]["lora_r"] = 8
        results[idx]["max_length"] = 1024
        print(f"  进度: {idx + 1}/{len(questions)}")

    # ================= 步骤 5：保存结果 =================
    print(f"\n保存评测结果到: {args.output_file}")
    # 确保输出目录存在
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_file, results)
    print("  结果保存成功！\n")

    # --- 对比总结 ---
    print("=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"\n共评测了 {len(results)} 个问题。")
    print("打开 JSONL 文件查看具体对比：")
    print(f"  {args.output_file}")
    print("\n观察点:")
    print("  1. LoRA 模型的回答是否更专业？")
    print("  2. LoRA 模型的回答是否更有条理？")
    print("  3. LoRA 模型是否更倾向于建议就医？")


if __name__ == "__main__":
    main()

'''
(E:\conda-envs\minilora) PS D:\LLMProjects\MiniLoRA> python scripts/eval_lora.py
============================================================
Base vs LoRA 推理对比
============================================================

读取评测问题
  问题加载成功，共 10 个问题
[1/2] 加载 Base 模型...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 312.07it/s]
  Base 模型加载完成

  Base 模型回答中...
  进度: 1/10
  进度: 2/10
  进度: 3/10
  进度: 4/10
  进度: 5/10
  进度: 6/10
  进度: 7/10
  进度: 8/10
  进度: 9/10
  进度: 10/10

释放 Base 模型显存...
  显存已释放

[2/2] 加载 LoRA 模型...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 302.66it/s]
E:\conda-envs\minilora\Lib\site-packages\transformers\modeling_utils.py:3303: FutureWarning: `is_gradient_checkpointing` is deprecated and will be removed in v6. Use `is_activation_checkpointing` instead.
  warnings.warn(
  LoRA 模型加载完成

  LoRA 模型回答:
  进度: 1/10
  进度: 2/10
  进度: 3/10
  进度: 4/10
  进度: 5/10
  进度: 6/10
  进度: 7/10
  进度: 8/10
  进度: 9/10
  进度: 10/10

保存评测结果到: results\base_vs_lora.jsonl
  结果保存成功！

============================================================
对比总结
============================================================

共评测了 10 个问题。
打开 JSONL 文件查看具体对比：
  results\base_vs_lora.jsonl

观察点:
  1. LoRA 模型的回答是否更专业？           否。 Base 整体更全面、准确、谨慎；LoRA 个别有具体数值或简洁建议，但多组存在跑题、重复、不严谨或高风险表述，如糖尿病建议“白米面为主食”、失眠提到安眠药、胃痛/贫血原因表述不够准确等。
  2. LoRA 模型的回答是否更有条理？          否。 Base 多数采用清晰分点、层次完整；LoRA 虽也分点，但常见段落堆砌、内容杂糅、条目冗余，整体条理性不如 Base。
  3. LoRA 模型是否更倾向于建议就医？         否，至少不明显。 LoRA 在感冒发烧、儿童发烧、咳嗽等回答中会建议就医，但不少回答直接给建议而缺少就医提示；Base 在多组中明确建议就医、遵医嘱或定期体检，整体并不比 LoRA 弱。
'''