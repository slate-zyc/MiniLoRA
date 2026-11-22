"""
固定配置运行 Base 与两个 LoRA 模型，逐题保存回答。

运行方式:
    # 测试程序是否能正常运行
    python model_eval/run_eval.py --split dev --model original_lora --limit 2 --output-dir model_eval/results/my_dev_2
    # 选择最大生成长度（256、512）：固定贪心解码和重复惩罚 1.1，只改 --max-new-tokens
    python model_eval/run_eval.py --split dev --model original_lora --greedy --max-new-tokens 256 --repetition-penalty 1.1
    python model_eval/run_eval.py --split dev --model original_lora --greedy --max-new-tokens 512 --repetition-penalty 1.1
    # 选重复惩罚（1.0、1.1）：固定 256 和贪心解码，与上一步的结果比较
    python model_eval/run_eval.py --split dev --model original_lora --greedy --max-new-tokens 256 --repetition-penalty 1.0
    # 选解码方式（贪心、采样 0.3、采样 0.7）：固定 256 和 1.1
    python model_eval/run_eval.py --split dev --model original_lora --temperature 0.3 --max-new-tokens 256 --repetition-penalty 1.1
    python model_eval/run_eval.py --split dev --model original_lora --temperature 0.7 --max-new-tokens 256 --repetition-penalty 1.1

    # 正式评测
    python model_eval/run_eval.py --split test
    # - 若再次运行应该重新指定目录，防止结果覆盖
    $run = "model_eval/results/test_all_models/temp0.3_length256_repeat1.1_rerun"
    python model_eval/analyze_results.py --run-dir $run --output-file "$run/test_comparison.md"

输出:
    results/本次运行目录/answers.jsonl
    results/本次运行目录/run_info.json
"""

import argparse
import gc           # 管理 Python 的垃圾回收
import hashlib      # 计算内容指纹
import json
import platform     # 获取 Python 和操作系统的信息
import time
from datetime import datetime
from importlib.metadata import version      # 查询已安装包的版本号
from pathlib import Path
from zoneinfo import ZoneInfo               # 指定时区

import torch
from peft import PeftModel      # 用来把训练好的 LoRA adapter 加载到基础模型上
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, set_seed     # 类型注解，随机种子

from check_data import EVAL_DIR, file_hash, load_jsonl


# ============================================================
# 加载模型：三个模型使用同一个基础模型与 tokenizer
# ============================================================

def load_model(config: dict, adapter_dir: str | None):
    """先加载基础模型，再按需挂载 LoRA。

    adapter_dir: 形如 "../outputs/qwen-medical-lora"
    - ./ 和 ../ 的区别就是：一个留在当前目录，一个先返回上一级目录
    """
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        dtype=getattr(torch, config["dtype"]),   # 固定 bfloat16，避免不同精度干扰比较   getattr(obj, name) 等价于 obj.name，区别是 name 可以是运行时才确定的字符串
        device_map={"": config["device"]},      # 整个模型放在同一张 GPU 上，每次只加载一个  键 ""：空字符串，是 accelerate 库约定的根模块名
        local_files_only=True,                 # 使用项目已有权重，不联网下载模型
        trust_remote_code=False,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, EVAL_DIR / adapter_dir)
    model.eval()  # 关闭训练阶段的 dropout
    return model


# ============================================================
# 生成回答：只输入问题，参考要点不传给模型
# ============================================================

@torch.inference_mode()  # 推理时不构建梯度图，减少显存占用
def generate(tokenizer, model, questions: list[dict], config: dict,
             generation_config: GenerationConfig) -> list[dict]:
    """基于问题生成回答

    返回的键包括 response、input_tokens、output_tokens、finish_reason、batch_seconds

    questions 批量输入时，需要指定分词器 AutoTokenizer.from_pretrained() 参数 padding_side="left"
    - 因为自回归需要使用右边的真实 token, 所以需要在左边打 pad，默认 padding_side="right
    """

    # 步骤 1：构建对话消息，推理只需要 system + user, 不需要 assistant
    conversations = [
        [
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": f"{config['user_prefix']}\n\n问题：{item['question']}"},
        ]
        for item in questions
    ]

    # 步骤 2：应用 chat template 到 tokenize
    inputs = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,                # 直接转成 token ID
        add_generation_prompt=True,   # 在末尾添加 assistant 开始标记
        padding=True,                 # 同一批问题补到相同长度     在 AutoTokenizer.from_pretrained() 通过参数 padding_side="left" 指定补齐方式
        return_tensors="pt",
        return_dict=True,             # 同时返回 input_ids 与 attention_mask, BatchEncoding 对象（一个 dict 子类）
    ).to(config["device"])

    # 步骤 3：生成回答
    start = time.perf_counter()     # perf_counter() 保证不会倒退，即使系统时间被 NTP 校正、用户手动改时间、时区切换，它也只增不减。这是它和 time.time() 最本质的区别。
    output_ids = model.generate(
        **inputs,
        generation_config=generation_config,  # 显式使用 完整 配置，避免继承模型的采样设置  generation_config 是一个 GenerationConfig 对象，承载所有生成超参数
    )       # 返回 torch.Size([batch_size, token_len])token_len 包含输入 Prompt 和新生成的 Token

    # 步骤 4：解码
    new_rows = output_ids[:, inputs.input_ids.shape[1]:].tolist()   # 去掉 prompt 部分；把 PyTorch 张量转成 Python 嵌套列表 list[list[int]]
    batch_seconds = time.perf_counter() - start

    # ESO 的 ID 转为集合，方便后续 O(1) 判断
    # 在 Qwen2.5-0.5B-Instruct 模型的 .|Qwen2.5-0.5B-Instruct\generation_config.json 中可查到有两个 eos_token_id，为列表形式：
    # "eos_token_id": [
    #     151645,       # <|im_end|>        一条聊天消息结束
    #     151643        # <|endoftext|>     文本结束
    #   ]
    eos_ids = generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])      # isinstance(值, 类型) 是 Python 的内置函数，用来判断一个值是不是指定类型，结果是 True 或 False。

    results = []
    for token_ids, input_tokens in zip(new_rows, inputs.attention_mask.sum(dim=1).tolist()):    # inputs.attention_mask.sum(dim=1): 每条真实 prompt 长度（不算 padding），形状 (B,), .tolist() 转成 Python int 列表 list[int]
        # 同一批中先结束的回答会补 PAD，只保留第一个 EOS 及之前的内容。
        eos_index = next((i for i, token in enumerate(token_ids) if token in eos_ids), None)    # 找第一个 EOS 的位置，找不到时返回 None
        if eos_index is not None:
            token_ids = token_ids[:eos_index + 1]       # 截断到第一个 EOS（含 EOS）
        results.append({
            "response": tokenizer.decode(token_ids, skip_special_tokens=True),      # 逐条解码，所以使用 decode 而不是 batch_decode
            "input_tokens": input_tokens,        # 只统计实际输入，不计补位，整数
            "output_tokens": len(token_ids),     # 包含末尾 EOS，不计补位
            "finish_reason": "eos" if eos_index is not None else "length",      # 结束原因，"length" 表示回答过长被截断
            "batch_seconds": round(batch_seconds, 3),  # 同一批的耗时，不是单题延迟
        })
    return results


# ============================================================
# 主函数：读取配置 → 依次生成 → 保存运行信息
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="使用固定配置运行独立模型评测")
    parser.add_argument("--config", type=Path, default=EVAL_DIR / "eval_config.json",
                        help="固定推理配置")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="dev 用于调试，test 用于正式评测")
    parser.add_argument("--limit", type=int, default=None,
                        help="只运行前几题；用于小样本检查，记录为子集运行")
    parser.add_argument("--model", default=None, help="只运行指定模型，例如 original_lora；不选择时运行全部三个模型")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="覆盖每题的生成长度上限")

    # 在 parser 里创建一个互斥参数组，并把这个组存到变量 decoding 里。即不允许 --greedy --temperature 0.3
    decoding = parser.add_mutually_exclusive_group(required=False)  # 默认就是 False, 改为 True 表示必须选一个
    decoding.add_argument("--greedy", action="store_true", help="使用贪心解码")
    decoding.add_argument("--temperature", type=float, default=None,
                          help="指定温度时使用采样，并固定 top_p=0.9")

    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="调试时覆盖重复惩罚，其余生成参数按配置文件固定")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="结果目录")    # 在下文设置默认目录
    args = parser.parse_args()

    # with args.config.open("r", encoding="utf-8") as f:
    #     config = json.load(f)   # json.load() 用于直接读取 JSON 文件对象；json.loads() 用于 逐行 读取 JSONL 文件
    # 效果同上
    config = json.loads(args.config.read_text(encoding="utf-8"))    # read_text() 读取文件返回字符串
    # config 为文件读取结果，后续修改 config 不会影响源文件

    # 准备评测模型
    if args.model is not None:  # 配置 --model 时只保留选择的一个模型，为 None 时三个模型都运行
        config["models"] = [spec for spec in config["models"] if spec["name"] == args.model]
        if not config["models"]:        # 一般不会触发，会在eval_config.json 中提前配置好
            parser.error(f"配置中没有模型 {args.model}")

    # 用命令行参数覆盖本次运行的配置
    generation = config["generation"]
    if args.max_new_tokens is not None:
        generation["max_new_tokens"] = args.max_new_tokens
    if args.repetition_penalty is not None:
        generation["repetition_penalty"] = args.repetition_penalty
    if args.greedy:
        generation.update(do_sample=False, temperature=1.0, top_p=1.0)
    elif args.temperature is not None:
        generation.update(do_sample=True, temperature=args.temperature, top_p=0.9)

    # 准备评测问题
    dataset_path = EVAL_DIR / config["datasets"][args.split]
    all_questions = load_jsonl(dataset_path)
    questions = all_questions[:args.limit]

    # 结果文件保存目录及命名规则
    model_name = args.model or "all_models"
    decoding_name = f"temp{generation['temperature']}" if generation["do_sample"] else "greedy"
    output_dir = (args.output_dir or EVAL_DIR / "results" / f"{args.split}_{model_name}" / f"{decoding_name}_length{generation['max_new_tokens']}_repeat{generation['repetition_penalty']}")
    output_dir.mkdir(parents=True, exist_ok=False)  # 目录已存在时报错。防止覆盖；每次运行单独保存，保留已有回答和评分

    # 统一使用 Qwen2.5-0.5B-Instruct 的分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        local_files_only=True,       # 使用项目已有权重，不联网下载模型
        trust_remote_code=False,     # 可删
        padding_side="left",         # 批量自回归生成需要从右端真实 token 接着写；默认是 right，批量回答时记得修改
    )

    # run_information 写入详细信息
    base_dir = Path(config["base_model"])
    now = datetime.now(ZoneInfo("Asia/Shanghai"))   # ZoneInfo 使用的是 IANA 时区数据库，Asia/Shanghai 代表中国大陆标准时间 UTC+8
    run_info = {
        "run_id": output_dir.name, "created_at": now.isoformat(), "split": args.split,      # 存放运行结果文件名，创建时间，数据集
        "question_count": len(questions), "dataset_count": len(all_questions),              # 实际回答问题数量，数据集问题数量
        "is_full_split": len(questions) == len(all_questions),          # 实际回答数量是否等于数据集问题数量
        "source_config_sha256": file_hash(args.config),         # 原配置文件哈希值，实际修改只会影响内存中的 config，并不会影响磁盘中的 congfig.json
        "effective_config": config,                             # 实际生效配置
        "effective_config_sha256": hashlib.sha256(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            # sort_keys=True 指定使用字典的key进行排序后再转换为 JSON 字符串，防止内容相同但先后顺序不同时哈希值不同
            # （这里的 config 是在同一个源文件上修改的，所以不存在这种情况，可省略；
            # 同样这里只是为了计算哈希，不需考虑可读性，也可省略 indent 参数，默认为 None）
        ).hexdigest(),                                          # 实际生效配置哈希
        # hashlib.sha256() 接收的是 bytes-like object（字节类对象），而不是普通的 Python 字符串 str。
        # 所以需要使用Python内置str类型 自带 的方法 .encode() 将字符串转换为字节（bytes）
        "dataset_sha256": file_hash(dataset_path),                      # 数据集文件哈希值
        "tokenizer_sha256": file_hash(base_dir / "tokenizer.json"),     # 分词器文件哈希值
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),       # 聊天模板哈希值
        "base_weights_sha256": {p.name: file_hash(p) for p in base_dir.glob("*.safetensors")},      # Qwen2.5-0.5B-Instruct 权重文件(文件名以 safetensors 结尾)哈希值   glob() 是 Path 对象的方法，用于按照通配符查找文件
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),       # python 版本， 运行平台（Windows）
            "torch": torch.__version__, "transformers": version("transformers"),        # torch 版本， transformers 库版本
            "peft": version("peft"), "cuda": torch.version.cuda,                        # peft 库版本， cuda 版本
            "gpu": torch.cuda.get_device_name(config["device"]),                        # gpu 型号
        },
        "models": {}, "status": "running",          # 模型详细配置，运行状态（"running"/ "complete"）        后文补充
    }
    info_path = output_dir / "run_info.json"
    # write()：文件对象（file object）的方法。
    # write_text()：pathlib.Path 对象的方法。
    info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")       # 中文正常显示， 控制 JSON 格式化时的缩进空格数量，否则通常是直接一行，文件以换行符结束

    # 记录模型回答 answer.jsonl，同时完善 run_ifo.json
    with (output_dir / "answers.jsonl").open("w", encoding="utf-8") as f:
        for spec in config["models"]:       # --model 后选择模型时只有一个，不选择时有三个
            print(f"\n加载模型: {spec['name']}", flush=True)
            model = load_model(config, spec["adapter_dir"])     # 加载 + 挂载 模型
            gen = config["generation"]             # 设置 model.generate() 的 generation_config 参数
            generation_config = GenerationConfig(
                max_new_tokens=gen["max_new_tokens"],   # 最多生成的新 token 数，不含问题
                do_sample=gen["do_sample"],             # False 取最高概率（greedy），True 按概率抽样
                num_beams=gen["num_beams"],             # 1表示不做束搜索：每一步保留 B 个最优候选序列，最终选累计概率最高的
                repetition_penalty=gen["repetition_penalty"],       # 1.0不加惩罚，大于1.0抑制重复
                no_repeat_ngram_size=gen["no_repeat_ngram_size"],   # 0不强制禁止重复片段
                use_cache=gen["use_cache"],             # 复用历史KV，避免每步重复计算
                temperature=gen["temperature"],         # 越低越偏向高概率 token，贪心时不生效
                top_p=gen["top_p"],                     # 越低越偏向高概率 token，贪心时不生效
                top_k=gen["top_k"],                     # 采样时最多保留的候选数，本轮固定
                eos_token_id=model.generation_config.eos_token_id,      # 保留 Qwen 的结束标记
                pad_token_id=tokenizer.pad_token_id,        # 通过在模型的 tokenizer_config.json 和 tokenizer.json 文件中可以找到
                bos_token_id=tokenizer.bos_token_id,        # 起始标记，tokenizer_config.json 中可以查到这版 Qwen 的分词器没有单独设置 BOS
            )
            model_info = {"generation_config": generation_config.to_dict()}
            if spec["adapter_dir"] is not None:
                adapter_dir = EVAL_DIR / spec["adapter_dir"]    # eg D:\LLMProjects\MiniLoRA\model_eval\..\outputs\qwen-medical-lora 即为 D:\LLMProjects\MiniLoRA\outputs\qwen-medical-lora
                model_info["adapter_config"] = json.loads(
                    (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
                )
                model_info["adapter_sha256"] = file_hash(adapter_dir / "adapter_model.safetensors")     # LoRA 权重参数文件哈希
            run_info["models"][spec["name"]] = model_info       # 完善 run_ifo.json
            info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")   # 重新写入，以更新文件

            # 开始写入 answers.json
            set_seed(config["seed"])  # 各模型在相同种子下开始生成
            batch_size = config["batch_size"]  # 三组模型共用批大小与题目顺序
            for start in range(0, len(questions), batch_size):
                batch_questions = questions[start:start + batch_size]
                results = generate(tokenizer, model, batch_questions, config, generation_config)    # 在这里面会处理 batch_questions，只传给模型问题而不传入参考答案
                for item, result in zip(batch_questions, results):
                    row = {
                        "model": spec["name"], "question": item["question"], **result,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()  # 每批写入，方便查看运行进度
                truncated = sum(item["finish_reason"] == "length" for item in results)      # 统计本批次回答被截断的数量
                print(f"  {start + len(batch_questions)}/{len(questions)} 已完成，本批截断 {truncated} 题", flush=True)

            # 释放模型显存，这里不要释放 tokenizer 的显存，因为 tokenizer 是共有的
            del model
            gc.collect()
            torch.cuda.empty_cache()  # 释放上一组模型显存，再加载下一组

    run_info["status"] = "complete"     # 更新运行状态： "running" -> "complete"
    info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")   # 重新写入，以更新文件
    print(f"\n结果目录: {output_dir}")


if __name__ == "__main__":
    main()
