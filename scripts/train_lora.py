"""
LoRA 微调训练脚本：在 Qwen2.5 上用 LoRA 做医疗健康领域 SFT

输入: data/medical/train.jsonl + valid.jsonl（由 prepare_medical_sft.py 生成）
输出: outputs/qwen-medical-lora/（LoRA adapter 权重）

运行方式:
    python scripts/train_lora.py --max-train-samples 50 --max-valid-samples 20 --epochs 1   # 小样本测试
    python scripts/train_lora.py  # 全样本训练

核心流程:
    1. 加载 tokenizer 和模型
    2. 配置 LoRA（冻结原始权重，只训练低秩矩阵）
    3. 加载数据，用 preprocess 转成 token 序列
    4. 用 HuggingFace Trainer 训练
    5. 保存 LoRA adapter
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model       # 定义 LoRA 配置、指定模型任务类别、为原模型添加 LoRA
# 加载因果语言模型、文本与 Token ID 的转换、处理一个 Batch 内的数据填充、封装训练循环、定义训练超参数
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers import PreTrainedTokenizerBase    # 用于显式类型注解

# ============================================================
# 常量定义
# ============================================================

# -100 是 PyTorch CrossEntropyLoss 的 ignore_index（忽略索引）
# 在 labels 中，值为 -100 的位置不参与 loss 计算
# SFT 标准做法：prompt 部分设为 -100（不学复述问题），只对 answer 部分计算 loss
# 模型可以利用 Prompt 作为上下文，但 Prompt 对应的位置不作为监督目标计算 Loss。
# 这叫 "assistant-only loss mask"
IGNORE_INDEX = -100


# ============================================================
# 数据加载
# ============================================================

def load_jsonl(path: Path) -> list[dict]:
    """读取 jsonl 文件，返回 list[dict]

    列表推导式写法，等价于:
        items = []
        for line in f:
            if line.strip():        # strip() 去掉空白后如果不为空
                items.append(json.loads(line))
        return items
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================================================
# 数据格式转换：原始数据 → messages
# ============================================================

def build_messages(example: dict) -> list[dict]:
    """将一条原始数据转成 chat messages 格式

    输入 example 格式（来自 jsonl）:
        {
            "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
            "input": "高血压应该怎么办？",
            "output": "高血压患者应该..."
        }

    输出 messages 格式（3 条 message）:
        [
            {"role": "system",    "content": "你是一名谨慎的医疗健康问答助手。..."},
            {"role": "user",      "content": "请以谨慎、专业、易懂的方式...\n\n问题：高血压应该怎么办？"},
            {"role": "assistant", "content": "高血压患者应该..."},
        ]

    为什么要转成 messages 格式？
    - Qwen 是 chat 模型，训练和推理都用 messages 格式
    - tokenizer.apply_chat_template() 会根据 messages 自动拼接特殊 token
      比如 system 前会有 <|im_start|>system，后面会有 <|im_end|>
    - 不同模型的 chat template 不同，用 messages 格式可以统一处理
    """
    # 拼接 user 内容：instruction + 换行 + "问题：" + input
    user_content = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    if input_text:
        user_content = f"{user_content}\n\n问题：{input_text}"

    return [
        {"role": "system", "content": "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"].strip()},
    ]


# ============================================================
# 数据预处理：messages → token 序列（模型能吃的格式）
# ============================================================

def preprocess(example: dict, tokenizer: PreTrainedTokenizerBase, max_length: int) -> dict:
    """将一条数据转成训练用的 token 序列

    返回:
        {
            "input_ids": [token1, token2, ...],          # prompt tokens + answer tokens
            "attention_mask": [1, 1, 1, ...],            # 全 1（后面 DataCollator 会动态 pad）
            "labels": [-100, -100, ..., ans_tok1, ...]   # prompt 部分 -100，answer 部分真实 id
        }

    这是 SFT 数据处理中最核心的函数，流程如下:
    """
    # --- 步骤 1: 构造 messages ---
    messages = build_messages(example)

    # --- 步骤 2: 分离 prompt 和 answer ---
    # prompt_messages = system + user（前 2 条），用来生成输入
    # answer = assistant 的回复（最后 1 条），是模型要学习生成的目标
    prompt_messages = messages[:-1]
    answer = messages[-1]["content"]

    # --- 步骤 3: tokenize prompt ---
    # apply_chat_template 会把 messages 转成模型特有的 token 序列
    # 比如 Qwen 的模板会生成:
    #   <|im_start|>system\n你是一名...<|im_end|>\n
    #   <|im_start|>user\n问题...<|im_end|>\n
    #   <|im_start|>assistant\n  ← add_generation_prompt=True 会加上这个
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,               # 返回 token ids，而不是字符串
        add_generation_prompt=True,  # 在末尾加上 assistant 开始回答所需的标记，表示接下来应该由 Assistant 继续生成内容，而不是继续生成 User 的问题。
        return_tensors=None,         # 返回 list（方便后续处理） "pt" 返回二维 tensor
    )

    # ⚠️ 踩坑点：apply_chat_template 的返回值类型不固定
    # 有些版本返回 BatchEncoding 对象（有 .input_ids .token_type_ids .attention_mask 属性）
    # 类型: <class 'transformers.tokenization_utils_base.BatchEncoding'>
    # 有些版本返回 list
    # 有些版本返回 tensor
    # 所以要做类型检查和转换
    # hasattr() 用于判断对象是否具有某个属性
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids     # BatchEncoding → 提取 input_ids
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()      # tensor → list

    # --- 步骤 4: tokenize answer ---
    # answer 后面加 eos_token（结束标记，Qwen 是 <|im_end|>），告诉模型"回答到这里结束"
    # add_special_tokens=False：不加 BOS 等特殊标记（prompt 部分已经加过了）
    # 返回的是 BatchEncoding (字典)，取出 "input_ids"
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

    # --- 步骤 5: 拼接 input_ids ---
    # 最终序列 = prompt tokens + answer tokens
    # 模型看到 prompt tokens 后，应该生成 answer tokens
    # Causal Language Model 的 训练方式是 Prompt 和 Answer 拼接成一条 Token 序列，通过 labels 控制哪些位置计算损失
    input_ids = prompt_ids + answer_ids

    # --- 步骤 6: 构造 labels（关键！）---
    # labels 是 SFT 的核心概念:
    #   - prompt 部分: 设为 -100 → CrossEntropyLoss 会忽略这些位置
    #   - answer 部分: 保留真实 token id → 模型在这些位置计算 loss
    #
    # 为什么 prompt 部分不计算 loss？
    #   因为 prompt 是"问题"，模型不需要学习"复述问题"
    #   模型只需要学习"看到问题后怎么回答"
    #
    # 直觉理解:
    #   input_ids: [你, 是, 谁, <|assistant|>, 我, 是, 助, 手]
    #   labels:    [-100, -100, -100, -100, 我, 是, 助, 手]
    #   → 模型只需要学习：当看到 "...<|assistant|>" 后，生成 "我是助手"
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    # 训练时，模型内部会将预测结果与目标 Token 错开一个位置。因此，这里的数据预处理不需要手动把 labels 整体向左移动一位。

    # --- 步骤 7: 截断到 max_length ---
    # 如果序列太长（超过模型最大上下文），截断
    # input_ids 和 labels 必须同步截断，保持一一对应
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    # --- 步骤 8: attention_mask ---
    # 当前所有 token 都是真实内容，所以全设为 1
    # 后面 DataCollatorForSeq2Seq 会动态 pad 短序列，pad 位置的 mask 设为 0
    attention_mask = [1] * len(input_ids)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ============================================================
# 模型加载 + LoRA 配置
# ============================================================

# def main() 开头的 tokenizer/model 加载部分也在讲同一件事
# 这里拆开讲是为了让你理解每一步

# --- 步骤 1: 加载 tokenizer ---
# tokenizer 负责"文本 ↔ token ids"的互相转换
# AutoTokenizer.from_pretrained 会自动下载/加载 tokenizer 配置
# trust_remote_code=True：Qwen 用了自定义 tokenizer 代码，必须开启
# 当前官方 Qwen2.5 Tokenizer 已有 Padding Token，而 Hugging Face 训练接口也在持续更新。
# 项目中关于“必须启用远程代码”“Qwen 没有 PAD Token”等注释，建议以实际模型配置和当前库版本为准，不要全部当成普遍规则。

# --- 步骤 2: 设置 pad_token ---
# 有些模型（如 Qwen）没有定义 pad_token
# pad_token 用于 batch 训练时对齐不同长度的序列
# 一般设为 eos_token（结束标记）即可

# --- 步骤 3: 加载模型 ---
# AutoModelForCausalLM.from_pretrained 自动下载/加载模型权重
# torch_dtype=torch.bfloat16：用 bfloat16 精度（省一半显存，精度损失很小）
#   为什么用 bfloat16 不用 float16？
#   bfloat16 数值范围和 float32 一样，不容易溢出
#   float16 数值范围小，训练时容易出现 inf/nan
# device_map="auto"：自动把模型分配到可用的 GPU 上
#   RTX 3060 有 6GB 显存，Qwen2.5-0.5B 大约需要 1GB（bfloat16）

# --- 步骤 4: 关闭 use_cache ---
# use_cache = True 时，推理会缓存 KV（Key-Value），加速自回归生成
# 但训练时不需要这个缓存（训练是 teacher forcing，一次前向传播）
# 而且开启缓存会多占显存

# --- 步骤 5: 启用输入梯度 ---
# enable_input_require_grads() 让模型输入也参与梯度计算
# 为什么需要？LoRA 层的输入是原始模型层的输出
# 它用于让模型输入 Embedding 的输出参与梯度计算，某些冻结基础模型、使用 PEFT 和梯度检查点的组合会需要这种设置。
# 一般情况下，冻结原始权重不会阻止 LoRA 参数接收梯度。 这行代码并不是所有 LoRA 训练都必须使用的。 这里可删

# --- 步骤 6: 配置 LoRA ---
# LoraConfig 的每个参数含义:
#   task_type=TaskType.CAUSAL_LM: 因果语言模型（GPT 类自回归模型）
#   r=8: rank，低秩矩阵的维度。r 越大参数越多，表达能力越强
#   lora_alpha=16: 缩放因子。实际缩放 = alpha/r = 16/8 = 2
#   lora_dropout=0.05: LoRA 层的 dropout 比率，防止过拟合
#   target_modules: 对哪些层加 LoRA
#     - q_proj, k_proj, v_proj, o_proj: 注意力层的 Q/K/V/Output 投影
#     - gate_proj, up_proj, down_proj: FFN 层的三个投影（SwiGLU）
#   LoRA 公式: h = Wx + (alpha/r) * BAx
#     W: 原始权重（冻结，不训练）
#     A: d×r 矩阵（训练）
#     B: r×d 矩阵（训练）
#     alpha/r: 缩放因子

# --- 步骤 7: get_peft_model ---
# get_peft_model(model, lora_config) 做了什么？
# 1. 冻结原始模型所有参数（requires_grad = False）
# 2. 在 target_modules 指定的层后面插入 LoRA 层（A 和 B 矩阵）
# 3. 返回一个 PeftModel，只有 LoRA 参数可训练
# model.print_trainable_parameters() 打印可训练参数占比
# 预期: ~4.4M 可训练参数 / ~498M 总参数 ≈ 0.88%


# ============================================================
# 训练主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen LoRA on Chinese medical SFT data.")
    # --- 命令行参数 ---
    parser.add_argument("--model-name", type=str, default=r"D:\LLMProjects\LLMModels\Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/medical/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/medical/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--max-length", type=int, default=1024,      # 序列最大长度（token 数）
                        help="超过此长度的序列会被截断")
    parser.add_argument("--epochs", type=float, default=1.0,         # 训练轮数
                        help="1.0 = 看一遍全部数据")
    parser.add_argument("--batch-size", type=int, default=1,         # 每张卡的 batch size
                        help="RTX 3060 6GB 显存，只能设 1")
    parser.add_argument("--grad-accum", type=int, default=8,         # 梯度累积步数
                        help="batch_size=1 + grad_accum=8 等效于 batch_size=8")
    parser.add_argument("--lr", type=float, default=2e-4,            # 学习率
                        help="LoRA 用 2e-4，比全参微调大（因为只调少量参数）")
    parser.add_argument("--max-train-samples", type=int, default=0,  # 限制训练数据量
                        help="0 = 用全部数据，>0 = 只用前 N 条（调试用）")
    parser.add_argument("--max-valid-samples", type=int, default=0,  # 限制验证数据量
                        help="0 = 用全部数据")
    args = parser.parse_args()

    # ============================================================
    # 步骤 1: 加载 tokenizer 和模型
    # ============================================================
    # trust_remote_code=True 表示允许执行模型仓库提供的自定义 Python 代码（现在Qwen2.5已经不需要）。
    # 对于 Transformers 已原生支持的 Qwen2.5，通常可以不启用。只有确实需要自定义代码并信任其来源时才应该开启。
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    # # 现在官方 Qwen2.5-0.5B-Instruct 的 Tokenizer 本身已经定义了 Padding Token，因此通常不会进入这个条件分支。
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad_token，用 eos 代替

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        # bfloat16: 省显存，精度损失小。RTX 30/40 系列支持 bfloat16
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32,
        # device_map="auto": 自动分配到 GPU，如果显存不够会自动卸载到 CPU
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )
    model.config.use_cache = False       # 训练时关闭 KV Cache（省显存，训练不需要）
    model.enable_input_require_grads()   # 让输入张量参与梯度计算（非 LoRA 训练必要。冻结原始权重不会阻止 LoRA 参数接收梯度） 这里可删

    # ============================================================
    # 步骤 2: 配置 LoRA
    # ============================================================
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,    # Causal language modeling 因果语言模型，还有SEQ_CLS文本分类、SEQ_2_SEQ_LM、QUESTION_ANS等
        r=8,                             # rank：低秩矩阵维度
        lora_alpha=16,                   # 缩放因子：16/8 = 2
        lora_dropout=0.05,               # LoRA 分支在训练时使用 5% 的 Dropout，缓解过拟合
        target_modules=[                 # 对哪些线性层加 LoRA
            "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力层 4 个投影
            "gate_proj", "up_proj", "down_proj",      # FFN 层 3 个投影 SwiGLU = (xW1 * Swish(xW3))W2
        ],
    )
    model = get_peft_model(model, lora_config)    # 把 LoRA 挂到模型上 根据配置，将原始模型包装为带 LoRA Adapter 的 PEFT 模型
    model.print_trainable_parameters()            # 打印模型中可训练参数、总参数及其比例（预期 ~0.88%）

    # ============================================================
    # 步骤 3: 加载数据
    # ============================================================
    train_items = load_jsonl(args.train_file)
    valid_items = load_jsonl(args.valid_file)

    # 如果指定了 max_samples，只取前 N 条（调试用，全量训练太慢）
    if args.max_train_samples > 0:
        train_items = train_items[: args.max_train_samples]
    if args.max_valid_samples > 0:
        valid_items = valid_items[: args.max_valid_samples]

    # ============================================================
    # 步骤 4: 构建 Dataset
    # ============================================================
    # Dataset.from_list(): 把 list[dict] 转成 HuggingFace Dataset 对象
    # .map(): 对每条数据调用 preprocess 函数，把原始数据转成 token 序列
    #   num_proc: 可以设多进程加速，但 Windows 上有时有问题，默认 1 就行
    # remove_columns: 删除原始列（instruction/input/output），只保留 token 序列
    #   为什么要删？Trainer 默认会把所有列传给 model.forward()
    #   但 model.forward() 只认识 input_ids/labels/attention_mask，不认识 instruction
    train_dataset = Dataset.from_list(train_items).map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=list(train_items[0].keys()),     # map参数，映射之后删除原始列，只保留新生成的列
    )
    valid_dataset = Dataset.from_list(valid_items).map(
        lambda x: preprocess(x, tokenizer, args.max_length),
        remove_columns=list(valid_items[0].keys()),
    )

    # # 虽然 batch=1，但不同样本的 token 长度不同，某个长样本可能让显存突然升高。
    # lengths = [len(ids) for ids in train_dataset["input_ids"]]
    # print("最长训练输入：", max(lengths))
    # for limit in (256, 512, 768, 1024):
    #     print(f"长度 >= {limit} 的样本数：", sum(n >= limit for n in lengths))

    # ============================================================
    # 步骤 5: 配置 TrainingArguments
    # ============================================================
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),                # checkpoint 保存目录
        num_train_epochs=args.epochs,                   # 训练轮数
        per_device_train_batch_size=args.batch_size,    # 每张卡的训练 batch size
        per_device_eval_batch_size=args.batch_size,     # 每张卡的评估 batch size
        gradient_accumulation_steps=args.grad_accum,    # 梯度累积步数，达到累积次数后执行一次参数更新
        # 梯度累积原理:
        #   batch_size=1, grad_accum=8
        #   每次前向传播 1 条数据，累积 8 次梯度后才更新一次参数
        #   效果等同于 batch_size=8，但只需要 1/8 的显存
        learning_rate=args.lr,                          # 学习率
        # 为什么 LoRA 用 2e-4 而不是常见的 5e-5？
        # 因为 LoRA 只训练少量参数（~0.88%），可以用更大的学习率
        # 全参微调用大学习率会破坏原始权重

        # 梯度检查点通过减少前向传播中需要长期保存的激活，在反向传播时按需重算中间结果，以额外计算换取显存
        activation_checkpointing=True,  # 启用梯度检查点，省显存

        # 定期释放 PyTorch 暂时不用的显存缓存，观察 reserved 指标是否有所下降
        # 官方文档提到会有约 10% 的性能损失，牺牲部分训练时间尝试解决OOM
        torch_empty_cache_steps=10,

        # 下面的“步”通常指优化器更新步，而不是一个 bath_size 算一步
        logging_steps=10,            # 每 10 步打印一次 train loss
        save_steps=100,              # 每 100 步保存一次 checkpoint
        eval_strategy="steps",       # 按步数评估（还有 "epoch" 选项）
        eval_steps=100,              # 每 100 步在验证集上评估一次
        save_strategy="steps",       # 按步数保存
        save_total_limit=2,          # 最多保留 2 个 checkpoint（节省磁盘）

        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),  # CUDA 可用时且支持时开启 bfloat16 混合精度
        fp16=False,                  # 不用 float16（bfloat16 更稳定）
        report_to="none",            # 不上报到 wandb 等平台
        remove_unused_columns=False, # 保留所有列（因为我们自定义了 input_ids/labels）
        # 为什么不删？默认 Trainer 会删除模型 forward 不需要的列
        # 但我们已经手动构建了 input_ids/labels/attention_mask，Dataset.map() 已经删除了原始字段
        # 如果 Trainer 再删一次会出错
    )

    # ============================================================
    # 步骤 6: 创建 Trainer
    # ============================================================
    trainer = Trainer(
        model=model,                # 要训练的模型（已经挂了 LoRA）
        args=training_args,         # 训练参数
        train_dataset=train_dataset,    # 训练数据
        eval_dataset=valid_dataset,     # 验证数据
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),    # 整理批次内数据
        # DataCollatorForSeq2Seq 的作用:
        #   同一批次中不同样本长度不同（比如一个 200 token，一个 300 token）
        #   需要 pad 到本批次中最长的长度
        #   它会:
        #   1. 动态 pad input_ids（用 pad_token_id）
        #   2. 动态 pad labels（用 -100，这样 pad 位置不计算 loss）
        #   3. 动态生成 attention_mask（pad 位置为 0，真实 token 为 1）
        #   为什么用 DataCollatorForSeq2Seq 而不是 DataCollatorForLanguageModeling？
        #   因为 Seq2Seq 版本会正确处理 labels 的 padding（用 -100）
        # 注意：原代码虽然配置了 Batch Size 1，但它仍然需要一个能把样本整理为张量的 Collator。
    )

    # ============================================================
    # 步骤 7: 开始训练
    # ============================================================
    # trainer.train() 做了什么？
    # 1. 把数据按 batch_size 分批（从 Dataset 中读取一个 Batch）
    # 2. 每批: forward → 计算 loss → backward → 累积梯度
    # 3. 每 grad_accum 步: 更新参数（optimizer.step）
    # 4. 每 eval_steps 步: 在验证集上评估 loss
    # 5. 每 save_steps 步: 保存 checkpoint
    # 6. 每 logging_steps 步: 打印 train loss
    # trainer.train()

    # 从这里重新开始统计显存峰值
    torch.cuda.reset_peak_memory_stats()    # 重置 PyTorch 对显存峰值的统计
    try:
        trainer.train()
    finally:
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)      # 真实被数据（Tensor）占用的显存，而不是 PyTorch 向显卡申请的预留缓存
        print(f"训练峰值显存：{peak:.2f} MiB")
        print("当前 allocated：", torch.cuda.memory_allocated() / 1024 ** 2, "MiB")    # 真实被数据（Tensor）占用的显存
        print("当前 reserved：", torch.cuda.memory_reserved() / 1024 ** 2, "MiB")      # PyTorch 向显卡申请的预留缓存
        print(torch.cuda.memory_summary())
        # 峰值高，说明这次计算确实需要较多张量显存
        # reserved 明显高于 allocated，说明存在较多缓存或分配空间

    # ============================================================
    # 步骤 8: 保存最终结果
    # ============================================================
    # save_pretrained 只保存 LoRA adapter 权重（A 和 B 矩阵）
    # 不保存原始模型权重（因为原始权重冻结了，没变过）
    # 所以 adapter 文件很小（几 MB），而原始模型是几百 MB
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # 保存 tokenizer 是为了推理时能直接从 output_dir 加载一切（还需要加载一下基础模型，这里即 Qwen2.5-0.5B-Instruct）
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
