"""
Base vs LoRA 推理对比脚本：比较原始模型和微调模型在医疗问答上的差异

功能:
    1. 加载 base 模型（原始 Qwen），生成回答
    2. 加载 LoRA adapter（微调后的），生成回答
    3. 对比两者输出的差异

运行方式:
    python scripts/infer_compare.py --question "高血压患者日常生活中应该注意什么？"

核心概念:
    - Base 模型: 通用 Qwen，没有针对医疗领域微调
    - LoRA 模型: 同一个 Qwen + 训练好的 LoRA adapter，医疗领域更专业
    - 对比目的: 验证微调是否让模型回答更专业、更有条理
"""

import argparse
from pathlib import Path

import torch
from peft import PeftModel      # 用来把训练好的 adapter 加载到基础模型上
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 系统提示词：定义模型的角色和行为准则
# ============================================================

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"
# 为什么需要 system prompt？
# - 设定模型的角色（医疗助手），让回答风格一致
# - Qwen 是 instruct 模型，会根据 system prompt 调整回答风格
# - 在医疗场景中，"谨慎"很重要：不能过度诊断，要建议就医


# ============================================================
# 函数 1: 加载模型
# ============================================================

def load_model(model_name: str, adapter_dir: Path | None):
    """加载模型，可选择是否挂载 LoRA adapter

    参数:
        model_name: 基础模型路径（如 models/Qwen2.5-1.5B-Instruct）
        adapter_dir: LoRA adapter 目录（如果为 None，加载 base 模型）

    返回:
        (tokenizer, model) 元组

    加载 base 模型 vs LoRA 模型的区别:
    - Base: load_model("models/Qwen2.5-1.5B-Instruct", None)
    - LoRA: load_model("models/Qwen2.5-1.5B-Instruct", Path("outputs/qwen-medical-lora"))
    """
    # --- 步骤 1: 加载 tokenizer ---
    # 如果有 adapter_dir，优先从 adapter 目录加载 tokenizer
    # 因为训练时 tokenizer.save_pretrained() 保存到了 adapter 目录
    # 如果没有 adapter，就从基础模型目录加载
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name, trust_remote_code=False)
    # # Qwen 现在已经有了，可删
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad_token，用 eos 代替

    # --- 步骤 2: 加载基础模型 ---
    # 注意：这里加载的总是同一个基础模型
    # LoRA 的差异在于后续是否挂载 adapter
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,    # 可删
    )

    # --- 步骤 3: 挂载 LoRA adapter（如果有）---
    # PeftModel.from_pretrained 做了什么？
    # 1. 加载 adapter_config.json（读取 LoRA 配置：rank, alpha, target_modules 等）
    # 2. 在基础模型对应层后面插入 LoRA 层（A 和 B 矩阵）
    # 3. 加载 adapter_model.bin（训练好的 LoRA 权重）
    #
    # 关键点：
    # - 基础模型的权重仍然被冻结（不更新）
    # - 只有 LoRA 层的参数被加载
    # - 前向传播时：h = Wx + (alpha/r) * BAx
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)

    # --- 步骤 4: 设置为评估模式 ---
    # model.eval() 的作用：
    # 1. 关闭 dropout（推理时不应该随机丢弃神经元）
    # 2. 关闭 batch norm 的 running stats 更新（虽然因果模型一般不用 BN）
    # 3. 告诉模型"现在是推理阶段，不需要计算梯度"
    model.eval()
    return tokenizer, model


# ============================================================
# 函数 2: 生成回答
# ============================================================

def generate(tokenizer, model, question: str, max_new_tokens: int) -> str:
    """基于问题生成回答

    参数:
        tokenizer: 分词器
        model: 模型（base 或 LoRA）
        question: 用户问题
        max_new_tokens: 最多生成多少新 token

    返回:
        生成的回答字符串

    生成流程:
        构造 messages → tokenize → model.generate() → decode
    """
    # --- 步骤 1: 构造对话消息 ---
    # 推理时只需要 system + user，不需要 assistant
    # 因为 assistant 的回答是要生成的，不是给定的
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请以谨慎、专业、易懂的方式回答下面的医疗健康问题。\n\n问题：{question}"},
    ]

    # --- 步骤 2: 应用 chat template 并 tokenize ---
    # apply_chat_template 会把 messages 转成带特殊标记的 token 序列
    # 例如 Qwen 会生成类似这样的结构：
    # <|im_start|>system\n你是一名...<|im_end|>\n
    # <|im_start|>user\n问题...<|im_end|>\n
    # <|im_start|>assistant\n

    # 这里要注意训练阶段和推理阶段的区别：
    #   训练阶段还需要处理 answer(添加结束标记)，所以先输出列表而不是张量是为了便于与处理后的 answer 合并。即 分别编码 Prompt 和 Answer，再手动构造 Labels
    #   训练阶段由 Trainer 负责把数据一起送到 GPU，且列表是无法调用.to(model.device)的；
    #   而在推理阶段没有 Trainer 帮忙管理，后面直接就调用 model.generate() 了，所以直接把数据送入到 GPU

    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,                 # 暂时不转换为 Token ID，先返回格式化后的文本字符串
    #     add_generation_prompt=True,
    #     # 由于 tokenize=False 返回的是字符串，所以无需 return_tensors 参数
    # )
    # add_special_tokens=False 是为了防止重复添加特殊标记
    # model_inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
    # 分开写是为了方便调试，print(text)，直接观察完整的聊天模板
    # 合并版本
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenizer=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True    # 默认就是True，返回 BatchEncoding 对象，可以像字典一样访问，包含 input_ids 和 attention_mask 等字段
    ).to(model.device)

    # --- 步骤 3: 生成回答 ---
    # model.generate() 参数说明:
    # - max_new_tokens: 最多生成多少个新 token（不包括 prompt）
    # - do_sample=True: 启用采样，让回答更有多样性
    # - temperature=0.7: 控制随机性，越低越确定，越高越随机
    # - top_p=0.9: nucleus sampling，只从累积概率 > 0.9 的 token 中采样
    # - repetition_penalty=1.1: 重复惩罚，>1 降低重复内容
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,      # 不包含输入 Prompt 已有的 Token
        do_sample=True,                     # 启用随机采样
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,             # 重复惩罚系数
    )       # 返回 torch.Size([batch_size, token_len])   token_len 包含输入 Prompt 和新生成的 Token

    # --- 步骤 4: 解码 ---
    # 只解码新生成的部分（去掉 prompt 部分）
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]   # tokenizer.batch_decode()返回的是列表形式，[0]是为了列表变为字符串
    return response


# ============================================================
# 函数 3: 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Base vs LoRA 推理对比")
    parser.add_argument(
        "--model-name",
        type=str,
        default=r"D:\LLMProjects\LLMModels\Qwen2.5-0.5B-Instruct",
        help="基础模型路径",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("outputs/qwen-medical-lora"),
        help="LoRA adapter 目录",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="高血压患者日常生活中应该注意什么？",
        help="要提问的问题",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="最多生成的新 token 数",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Base vs LoRA 推理对比")
    print("=" * 60)
    print(f"\n问题: {args.question}\n")

    # --- 加载 Base 模型 ---
    print("[1/2] 加载 Base 模型...")
    tokenizer_base, model_base = load_model(args.model_name, adapter_dir=None)
    print("  Base 模型加载完成\n")

    # --- Base 模型生成 ---
    print("  Base 模型回答:")
    base_response = generate(tokenizer_base, model_base, args.question, args.max_new_tokens)
    print(f"  {base_response}\n")

    # --- 加载 LoRA 模型 ---
    print("[2/2] 加载 LoRA 模型...")
    if not args.adapter_dir.exists():
        print(f"  ⚠️  LoRA adapter 不存在: {args.adapter_dir}")
        print("  请先运行训练脚本生成 LoRA adapter")
        return
    tokenizer_lora, model_lora = load_model(args.model_name, args.adapter_dir)
    print("  LoRA 模型加载完成\n")

    # --- LoRA 模型生成 ---
    print("  LoRA 模型回答:")
    lora_response = generate(tokenizer_lora, model_lora, args.question, args.max_new_tokens)
    print(f"  {lora_response}\n")

    # --- 对比总结 ---
    print("=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"\nBase 模型输出长度: {len(base_response)} 字符")
    print(f"LoRA 模型输出长度: {len(lora_response)} 字符")
    print("\n观察点:")
    print("  1. LoRA 模型的回答是否更专业？")
    print("  2. LoRA 模型的回答是否更有条理？")
    print("  3. LoRA 模型是否更倾向于建议就医？")


if __name__ == "__main__":
    main()


'''
(E:\conda-envs\minilora) PS D:\LLMProjects\MiniLoRA> python scripts/infer_compare.py --question "高血压患者日常生活中应该注意什么？"
============================================================
Base vs LoRA 推理对比
============================================================

问题: 高血压患者日常生活中应该注意什么？

[1/2] 加载 Base 模型...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 297.91it/s]
  Base 模型加载完成

  Base 模型回答:
  高血压是一种常见的心血管疾病，主要表现为血压持续升高。对于高血压患者来说，日常生活中需要注意以下几点：

1. **饮食控制**：
   - 少盐饮食，避免过多摄入钠盐。
   - 增加蔬菜和水果的摄入量，保证充足的维生素和矿物质。
   - 减少饱和脂肪和反式脂肪的摄入，如油炸食品等。

2. **适量运动**：
   - 每周至少进行150分钟的中等强度有氧运动，比如快走、游泳或骑自行车。
   - 保持适度的身体活动，有助于降低血压。

3. **戒烟限酒**：
   - 戒烟可以有效预防多种心血管疾病。
   - 限制酒精摄入，男性每天不超过两杯，女性每天不超过一杯。

4. **保持正常体重**：
   - 控制体重，维持健康的BMI（身体质量指数）范围。

5. **定期监测血压**：
   - 定期测量血压，了解自己的血压变化情况，及时调整治疗方案。

6. **心理调节**：
   - 学会管理压力，保持良好的心态，避免过度焦虑。

7. **遵医嘱用药**：
   -

[2/2] 加载 LoRA 模型...
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:02<00:00, 133.67it/s]
E:\conda-envs\minilora\Lib\site-packages\transformers\modeling_utils.py:3303: FutureWarning: `is_gradient_checkpointing` is deprecated and will be removed in v6. Use `is_activation_checkpointing` instead.
  warnings.warn(
  LoRA 模型加载完成

  LoRA 模型回答:
  高血压是一种常见病，是心血管疾病的危险因素之一，长期血压升高会导致心脑血管疾病的发生。因此，高血压患者一定要重视控制血压，预防和治疗高血压。那么，在日常生活中应注意哪些呢?高血压患者的饮食方面应注意以下几点：1.低盐低脂饮食，少吃咸 菜、咸鱼等含盐食品;2.多吃蔬菜水果，保证每日摄入的纤维素达到30克以上;3.保持规律的运动，坚持每天半小时以上的有氧运动或中高强度运动;4.避免饮酒，戒烟，因为酒精会加速血压上升，戒烟后血压可下降50%左右，如果再不戒烟，则血压将更难以下降。此外，高血压患者还应注意以下事项：1.合理安排工作学习时间，不要加班到深夜;2.减少精神压力，避免过度紧张和焦虑;3.适当锻炼身体，如打太极拳、健身操、散步、骑自行车、游泳等;4.养成良好的睡眠习惯，每晚7-8小时睡眠，保持良好的睡眠环境。

============================================================
对比总结
============================================================

Base 模型输出长度: 449 字符
LoRA 模型输出长度: 358 字符

观察点:
  1. LoRA 模型的回答是否更专业？               否。 Base 回答更规范、全面；LoRA 虽有一些具体建议，但存在“戒烟后血压可下降50%左右”等不严谨表述。
  2. LoRA 模型的回答是否更有条理？             否。 Base 采用分点编号，结构更清晰；LoRA 基本是段落式叙述。
  3. LoRA 模型是否更倾向于建议就医？           否。 LoRA 回答中没有明确建议就医；Base 提到“遵医嘱用药”，但也不是明确建议就医。
'''