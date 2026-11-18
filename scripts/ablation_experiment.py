"""
LoRA 消融实验：在 Qwen2.5 上对比 rank 和训练数据量

输入: data/medical/train.jsonl + valid.jsonl（由 prepare_medical_sft.py 生成）
输出: outputs/lora-ablation/（LoRA adapter 权重）和 results/lora_ablation_summary.csv

运行方式:
    python scripts/ablation_experiment.py --run-all  # 依次运行预设的 5 组消融实验
    python scripts/ablation_experiment.py --r 4 --max-train-samples 1000 --output-dir outputs/lora-ablation/lora_rank4_1000
    python scripts/ablation_experiment.py --r 8 --max-train-samples 5000 --activation-checkpointing --torch-empty-cache-steps 10 --output-dir outputs/lora-ablation/lora_rank8_5000

预设实验: rank4_1000、rank8_1000、rank16_1000、rank8_200、rank8_5000。
--run-all 时只有 rank8_5000 开启梯度检查点和每 10 步清缓存；其余四组均关闭。
单组运行时两个选项由命令行控制，默认分别为 False 和 None。

核心流程:
    1. 加载 tokenizer 和模型
    2. 配置 LoRA（冻结原始权重，只训练低秩矩阵）
    3. 加载数据，用 preprocess 转成 token 序列
    4. 用 HuggingFace Trainer 训练
    5. 保存 LoRA adapter
"""

import argparse
import csv              # 把每组实验的配置、训练损失和验证损失写入新的 CSV 汇总表。
import subprocess       # 在 --run-all 时，为每组实验启动独立的 Python 进程；上一组结束后释放资源，再运行下一组。
import sys              # 通过 sys.executable 获取当前 Python 解释器路径，确保各组实验使用同一个 Conda 环境。
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model       # 定义 LoRA 配置、指定模型任务类别、为原模型添加 LoRA
# 加载因果语言模型、文本与 Token ID 的转换、处理一个 Batch 内的数据填充、封装训练循环、定义训练超参数
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers import set_seed    # 固定实验随机种子

from train_lora import load_jsonl, build_messages, preprocess

# ============================================================
# 模型加载 + LoRA 配置
# ============================================================

def load_model_and_lora(model_name, r=8, lora_alpha=16, lora_dropout=0.05):
    """加载模型并配置 LoRA

        返回: (tokenizer, lora_model)

        步骤：
        1. 加载 tokenizer
           - AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        2. 加载模型
           - AutoModelForCausalLM.from_pretrained(
               model_name,
               torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
               device_map="auto" if torch.cuda.is_available() else None,
               trust_remote_code=True,
           )

        关闭 use_cache（训练时不需要 KV Cache）
           - model.config.use_cache = False


        4. 配置 LoRA
           - LoraConfig(
               task_type=TaskType.CAUSAL_LM,    # 因果语言模型
               r=r,                               # rank，低秩矩阵维度
               lora_alpha=lora_alpha,             # 缩放因子
               lora_dropout=lora_dropout,         # dropout
               target_modules=[                   # 对哪些层加 LoRA
                   "q_proj", "k_proj", "v_proj", "o_proj",  # 注意力层
                   "gate_proj", "up_proj", "down_proj",      # FFN 层
               ],
           )

        5. 用 get_peft_model 把 LoRA 挂到模型上
           - model = get_peft_model(model, lora_config)

        6. 打印可训练参数
           - model.print_trainable_parameters()

        LoRA 公式: h = Wx + (alpha/r) * BAx
        - W: 原始权重（冻结）
        - A: d×r 矩阵（训练）
        - B: r×d 矩阵（训练）
        - alpha/r: 缩放因子
        """

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 加载基础模型
    cuda_available = torch.cuda.is_available()
    use_bf16 = cuda_available and torch.cuda.is_bf16_supported()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto" if cuda_available else None
    )

    # 训练时关闭 KV Cache
    model.config.use_cache = False

    # 3. 配置 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # 冻结基础模型参数，添加可训练的 LoRA 参数
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数

    return tokenizer, model

# ============================================================
# 训练主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen LoRA rank and training-data ablation experiments.")
    # --- 命令行参数 ---
    parser.add_argument("--model-name", type=str, default=r"D:\LLMProjects\LLMModels\Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=Path("data/medical/train.jsonl"))
    parser.add_argument("--valid-file", type=Path, default=Path("data/medical/valid.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lora_ablation"),
                        help="单组的输出目录；--run-all 时作为各组输出目录的父目录")
    parser.add_argument("--max-length", type=int, default=256,      # 序列最大长度（token 数）
                        help="超过此长度的序列会被截断")
    parser.add_argument("--epochs", type=float, default=1.0,         # 训练轮数
                        help="1.0 = 看一遍全部数据")
    parser.add_argument("--batch-size", type=int, default=1,         # 每张卡的 batch size
                        help="RTX 3060 6GB 显存，只能设 1")
    parser.add_argument("--grad-accum", type=int, default=4,         # 梯度累积步数
                        help="batch_size=1 + grad_accum=4 等效于 batch_size=4")
    parser.add_argument("--lr", type=float, default=2e-4,            # 学习率
                        help="LoRA 用 2e-4，比全参微调大（因为只调少量参数）")
    parser.add_argument("--max-train-samples", type=int, default=0,  # 限制训练数据量
                        help="0 = 用全部数据，>0 = 只用前 N 条（调试用）")
    parser.add_argument("--max-valid-samples", type=int, default=160,  # 限制验证数据量
                        help="0 = 用全部数据")

    # 解决 max_length=1024, max_train_samples=5000, r=8 时的 OOM 问题 max_length 使用256完全用不到
    parser.add_argument("--activation-checkpointing", action="store_true", default=False,
                        help="启用梯度检查点；默认关闭")
    parser.add_argument("--torch-empty-cache-steps", type=int, default=None,
                        help="每 N 个更新步清理显存缓存；默认不定期清理")

    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--summary-file", type=Path,
                        default=Path("results/lora_ablation_summary.csv"))
    parser.add_argument("--run-all", action="store_true", help="依次运行预设的 5 组消融实验")
    args = parser.parse_args()

    if args.r <= 0 or (args.torch_empty_cache_steps is not None and args.torch_empty_cache_steps <= 0):
        parser.error("--r 和 --torch-empty-cache-steps 必须为正整数")

    if args.run_all:
        # 每组启动独立进程，从同一基础模型开始，避免前一组权重或显存状态残留。
        common_args = []
        for name in ("model_name", "train_file", "valid_file", "max_length", "epochs",
                     "batch_size", "grad_accum", "lr", "max_valid_samples", "seed", "summary_file"):
            common_args.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])   # getattr(args, name)：从 args 对象中取出名为 name 的属性值
        for rank, samples in ((4, 1000), (8, 1000), (16, 1000), (8, 200), (8, 5000)):
            experiment = f"lora_rank{rank}_{samples}"
            command = [sys.executable, str(Path(__file__).resolve()), *common_args,
                       "--r", str(rank), "--max-train-samples", str(samples),
                       "--output-dir", str(args.output_dir / experiment)]
            if (rank, samples) == (8, 5000):
                command.extend(["--activation-checkpointing", "--torch-empty-cache-steps", "10"])
            print(f"开始消融实验：{experiment}", flush=True)
            subprocess.run(command, check=True)     # check=True 表示子进程非零退出码会抛 CalledProcessError，从而中断整个消融流程——即一旦某组失败就不继续，保证结果的完整性。
        return

    set_seed(args.seed)

    # 加载 tokenizer、基础模型并配置 LoRA
    tokenizer, model = load_model_and_lora(
        model_name=args.model_name,
        r=args.r,
        lora_alpha=2 * args.r,  # alpha / r = 2
        lora_dropout=0.05,
    )

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

    # 因果语言模型的标签会错开一位，排除截断后没有有效预测目标的样本。
    train_dataset = train_dataset.filter(
        lambda item: any(label != -100 for label in item["labels"][1:])
    )
    valid_dataset = valid_dataset.filter(
        lambda item: any(label != -100 for label in item["labels"][1:])
    )
    if len(train_dataset) == 0 or len(valid_dataset) == 0:
        raise ValueError("过滤后训练集或验证集为空，请检查数据及 --max-length。")
    print(f"有效训练样本：{len(train_dataset)}，有效验证样本：{len(valid_dataset)}")

    # ============================================================
    # 步骤 5: 配置 TrainingArguments
    # ============================================================
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),                # checkpoint 保存目录
        num_train_epochs=args.epochs,                   # 训练轮数
        per_device_train_batch_size=args.batch_size,    # 每张卡的训练 batch size
        per_device_eval_batch_size=args.batch_size,     # 每张卡的评估 batch size
        gradient_accumulation_steps=args.grad_accum,    # 梯度累积步数，达到累积次数后执行一次参数更新

        learning_rate=args.lr,                          # 学习率
        seed=args.seed,
        data_seed=args.seed,
        activation_checkpointing=args.activation_checkpointing,
        torch_empty_cache_steps=args.torch_empty_cache_steps,

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
    train_result = trainer.train()

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
    # 小样本实验可能达不到 eval_steps，训练结束后统一评估最终模型。
    eval_result = trainer.evaluate()
    trainable, total = model.get_nb_trainable_parameters()
    row = {
        "experiment": f"rank{args.r}_{len(train_items)}",
        "rank": args.r,
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "train_loss": f"{train_result.training_loss:.3f}",
        "eval_loss": f"{eval_result['eval_loss']:.3f}",
        "trainable_params_pct": f"{100 * trainable / total:.2f}%",
        "adapter_dir": args.output_dir.as_posix(),      # 把 Path 对象转成字符串，并且强制使用正斜杠 / 作为路径分隔符（即 POSIX 风格）
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not args.summary_file.exists()) or (args.summary_file.stat().st_size == 0)  # 文件不存在或者大小为 0 字节
    # 每完成一组就追加一行；重跑时保留之前的新实验记录。
    # "a"（append 追加模式）  newline=""（换行符不转换）
    with args.summary_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))        # 用字典写 CSV
        if write_header:    # 只有第一组写表头
            writer.writeheader()
        writer.writerow(row)    # 写入数据
    print(f"消融结果已追加到：{args.summary_file}")


if __name__ == "__main__":
    main()

'''

'''