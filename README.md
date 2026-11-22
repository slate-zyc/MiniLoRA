# MiniLoRA：中文医疗问答微调实验

个人 LLM 微调实践项目。使用 Qwen2.5-0.5B-Instruct 完成中文医疗问答 SFT、Base/LoRA 生成对比、数据质量审计和消融实验，并记录实际运行结果。

目前完成了数据准备、LoRA 训练、10 题生成评测，以及 rank 和训练数据量两组消融对比。项目重点是理解数据、训练配置与模型表现之间的关系。

消融实验完成后，另做了一轮三模型、100 题的正式评测。 [评测思路与步骤](#消融实验后的模型评测)

## 当前结果

10 题生成对比中，Base 的整体表达更清晰，LoRA 出现了更多重复、机械列举和重点偏移。后续优先改进监督数据质量，并测试清洗数据后的训练轮数。

| 实验 | 实际训练数 | train_loss | eval_loss |
|---|---:|---:|---:|
| rank4_1000 | 1000 | 2.734 | 2.683 |
| rank8_1000 | 1000 | 2.732 | 2.678 |
| rank16_1000 | 1000 | 2.740 | 2.673 |
| rank8_200 | 200 | 2.685 | 2.789 |
| rank8_5000 | 4999 | 2.708 | 2.611 |

五组均使用相同的 159 条有效验证数据，固定 alpha/r=2。增加数据量的收益大于本次增加 rank 的收益，rank8_5000 的验证损失最低。

详细记录：[实验结果分析](实验结果分析.md) · [消融 CSV](results/lora_ablation_summary.csv) · [训练日志](results/logs/lora_ablation_20261118.log) · [生成对比](results/base_vs_lora.jsonl)

消融实验后的 100 题正式评测中，`base` 在 45 道共同可评分题上的平均总分最高（5.51/8），`original_lora` 和 `rank8_5000` 分别为 4.73 和 4.04。`original_lora` 截断最少（18/100），`rank8_5000` 的平均重复比例最高（19.4%）。 [完整评测结果](#正式测试结果)

## 项目结构

```text
MiniLoRA/
├── README.md                         # 当前项目说明
├── requirements.txt                  # 当前环境的固定依赖版本
├── .gitignore
├── download_model.py                 # 下载基础模型
├── download_dataset.py               # 下载原始医疗数据
├── GPU_test.py                       # 检查 CUDA、GPU 与 BF16
├── data_quality_audit.ipynb           # 数据长度、空值和截断审计
├── 实验结果分析.md
├── LoRA训练显存不足问题复盘.md
├── scripts/
│   ├── prepare_medical_sft.py         # 按原始 train/valid/test 划分清洗、抽样
│   ├── train_lora.py                  # 单组 LoRA 训练、SFT 预处理
│   ├── infer_compare.py               # 单题 Base/LoRA 生成对比
│   ├── eval_lora.py                   # 批量生成对比，保存 JSONL
│   └── ablation_experiment.py         # 五组消融及 CSV 汇总
├── model_eval/                       # 调试集选配置与正式测试
│   ├── README.md                      # 评测流程、调试记录与评分规则
│   ├── eval_config.json               # 固定推理配置
│   ├── dev_set.jsonl                  # 20 道调试题
│   ├── test_set.jsonl                 # 100 道正式测试题
│   ├── prepare_eval_data.py           # 整理并划分评测数据
│   ├── check_data.py                  # 检查题集重复及与训练、验证集重合
│   ├── run_eval.py                    # 生成回答并记录参数
│   ├── analyze_results.py             # 汇总截断和重复指标
│   └── results/                      # 本地评测结果
├── data/
│   ├── medical_raw/                   # 原始下载数据，本地保存
│   └── medical/
│       ├── train.jsonl               # 5000 条
│       ├── valid.jsonl               # 300 条
│       ├── test.jsonl                # 300 条
│       └── eval_prompts.jsonl         # 固定 10 个评测问题
├── outputs/                          # 本地 adapter 与 checkpoint
│   ├── qwen-medical-lora/             # 10 题评测使用的单组训练模型，也以 original_lora 参与正式评测
│   └── lora_ablation/                 # 本次五组消融模型，其中 rank8_5000 参与正式评测
└── results/
    ├── base_vs_lora.jsonl
    ├── lora_ablation_summary.csv
    └── logs/
        └── lora_ablation_20261118.log
```

模型权重、训练输出、原始数据和处理后的 train/valid/test 文件保存在本地；代码、评测问题、结果表、分析和归档日志保留在项目中。

## 环境

本次运行环境：Windows、Python 3.11.18、RTX 3060 Laptop 6GB、PyTorch 2.6.0+cu124、Transformers 5.23.0、PEFT 0.21.3。其余版本见 [requirements.txt](requirements.txt)。

以下命令在项目根目录的 PowerShell 中执行。新环境可按以下顺序安装：

```powershell
conda create -n minilora python=3.11
conda activate minilora
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python GPU_test.py
```

先安装 CUDA 12.4 对应的 PyTorch wheel，再安装其余固定版本依赖。Notebook 在 PyCharm 或 VS Code 中选择 minilora 内核运行。

## 数据与模型

模型：[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)  
数据：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical)

当前基础模型目录为 `D:\LLMProjects\LLMModels\Qwen2.5-0.5B-Instruct`。更换目录时，更新下载脚本中的 `local_dir`，并在训练、推理和评测命令中传入 `--model-name`。

```powershell
python download_model.py
python download_dataset.py
python scripts/prepare_medical_sft.py --train-size 5000 --valid-size 300 --test-size 300 --seed 20261117
```

数据准备保留原始 train/valid/test 划分，分别清洗、打乱、抽样。SFT 数据包含 `instruction`、`input`、`output` 三个字段；训练时仅对 assistant 回答计算损失。

## 运行实验

### 单组训练

```powershell
python scripts/train_lora.py --max-train-samples 5000 --max-valid-samples 300 --max-length 1024 --epochs 1 --batch-size 1 --grad-accum 8 --output-dir outputs/qwen-medical-lora
```

该脚本使用 rank=8、alpha=16，开启梯度检查点和每 10 步清理显存缓存。显存压力较大时，可将 `--max-length` 调为 512。

### 生成对比

```powershell
python scripts/infer_compare.py --adapter-dir outputs/qwen-medical-lora --question "高血压患者日常生活中应注意什么？"
python scripts/eval_lora.py --adapter-dir outputs/qwen-medical-lora --output-file results/base_vs_lora.jsonl
```

批量评测读取固定的 10 个问题，先运行 Base，再释放显存并运行 LoRA。生成参数为 max_new_tokens=256、temperature=0.7、top_p=0.9、repetition_penalty=1.1。

`eval_lora.py` 中的训练元数据目前固定记录为 5000 条、rank=8、max_length=1024；更换训练配置或 adapter 时，同步更新这三个记录字段。

### 消融实验

```powershell
python scripts/ablation_experiment.py --run-all
```

预设依次运行 rank4_1000、rank8_1000、rank16_1000、rank8_200、rank8_5000。默认配置为 1 轮、max_length=256、batch_size=1、grad_accum=4、learning_rate=2e-4、seed=20261118，alpha 始终为 rank 的两倍。

各组取相同的前 160 条验证数据，过滤无有效监督标签的样本后为 159 条。只有 rank8_5000 开启 `activation_checkpointing=True` 和 `torch_empty_cache_steps=10`；其他组默认关闭。每组使用独立进程，从相同的基础模型开始训练。

每完成一组，脚本向 CSV 追加一行，两个 loss 固定保留三位小数。重复实验可用 `--summary-file` 和 `--output-dir` 指定新的结果表及模型目录。

后续运行如需同时保存终端日志：

```powershell
New-Item -ItemType Directory -Force results/logs | Out-Null
python -u scripts/ablation_experiment.py --run-all 2>&1 |
    Tee-Object -FilePath results/logs/lora_ablation_latest.log
```

本次完整日志已归档至 [results/logs/lora_ablation_20261118.log](results/logs/lora_ablation_20261118.log)。

### 消融实验后的模型评测

消融实验中，`rank8_5000` 的验证损失最低（2.611）。为了看它在实际问答中的表现，将它与基座模型 `base`、10 题生成对比所用的 `original_lora` 一起评测。三组沿用现有权重，在同一份题集和相同的推理设置下比较回答质量。

1. **整理并划分题集。** 从 `data/medical/test.jsonl` 提取问题和参考答案，统一文本格式、去重，再用固定种子划分出 20 道调试题和 100 道正式测试题。按归一化后的题面检查两组互不重叠，并与训练集、验证集比对完全重复的题目。
2. **在调试集上选推理配置。** 用 `original_lora` 对同一批 20 题依次比较生成上限 256 和 512、重复惩罚 1.0 和 1.1、贪心与采样温度 0.3 和 0.7。结合截断、重复和逐题切题情况，选定上限 256 tokens、重复惩罚 1.1、采样温度 0.3。 [调试过程与原始回答](model_eval/README.md)
3. **固定配置运行正式测试。** 三个模型统一使用选定的生成参数、提示词和随机种子，各自回答相同的 100 道正式测试题，共生成 300 条回答。评测脚本保存原始回答和实际运行参数。 [固定推理配置](model_eval/eval_config.json)
4. **统计指标并逐题评分。** 脚本统计触及长度上限的回答数和平均四字片段重复比例。AI 按事实正确性、要点覆盖、重复控制和格式可读性四项给每条回答打 0–2 分，严重错误单独标记，依据不足的项目记为待复核。 [正式评分规则](model_eval/README.md)

#### 正式测试结果

| 模型 | 截断 | 平均重复比例 | 同题平均总分 / 8 | 已标记严重错误 |
|---|---:|---:|---:|---:|
| `base` | 42/100 | 2.5% | 5.51 | 4 |
| `original_lora` | 18/100 | 10.1% | 4.73 | 8 |
| `rank8_5000` | 29/100 | 19.4% | 4.04 | 9 |

总分基于三个模型四项均可评分的 45 道同题。Base 的要点覆盖与同题总分最高。`original_lora` 截断最少，部分短答漏掉核心问题。`rank8_5000` 的循环复述较多。逐题评分、原始回答和典型错例报告保存在本地 `model_eval/results/test_all_models/temp0.3_length256_repeat1.1/`。 [正式测试结果索引](model_eval/README.md)

#### 复现正式推理与自动统计

在项目根目录使用已准备好的题集和模型。复现时将 `$run` 指向新的结果目录。

```powershell
$run = "model_eval/results/test_all_models/temp0.3_length256_repeat1.1_rerun"
python model_eval/run_eval.py --split test --config model_eval/eval_config.json --output-dir $run
python model_eval/analyze_results.py --run-dir $run --output-file "$run/test_comparison.md"
```

## 后续改进

- 高质量数据清洗。数据质量分类，如：

    | 类别 | 示例特征 | 处理建议 |
    | :--- | :--- | :--- |
    | A: 完整回答 | 能解释问题，包含必要信息 | 保留 |
    | B: 简短但有效 | 直接回答药物名称、术语或简单事实 | 人工判断，不能一律删除 |
    | C: 答案碎片 | 只有几个关键词，无法解决问题 | 删除或补充 |
    | D: 模板化回答 | 大量重复的通用句式 | 降权或清理 |
    | E: 问答不匹配 | 问题和答案讨论的内容不一致 | 删除 |
    | F: 重复样本 | 问题和答案完全重复 | 去重 |
    | G: 高风险错误 | 错误的诊断、治疗或用药建议 | 删除或人工复核 |
    | H: 不完整答案 | 句子明显被截断，缺少结尾 | 修复或删除 |

- 对比清洗前后数据，并以 rank=8、alpha=16、max_length=512 测试 1/2/3 轮。
- 按验证损失选择模型，再使用固定问题和相同生成设置对比回答质量。
- 增加随机种子重复，记录均值与波动。
