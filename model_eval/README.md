# MiniLoRA 模型评测

先用现有 `original_lora` 权重在调试集上选择推理配置，再用选定配置在正式测试集上比较三个现有模型。

## 数据从哪里来

题目和参考答案都来自项目原有的 `data/medical/test.jsonl`。整理脚本使用 `input` 和 `output`，统一换行并去掉首尾空白，再按固定种子打乱，从中取互不重叠的两段。

| 文件 | 题数 | 字段 |
|---|---:|---|
| `dev_set.jsonl` | 20 | `question`、`reference_answer` |
| `test_set.jsonl` | 100 | `question`、`reference_answer` |

生成时给模型的是问题及固定提示词。参考答案仅供核验使用。

在项目根目录运行以下命令可重建、检查两份题集：

~~~powershell
conda activate minilora
python model_eval/prepare_eval_data.py
python model_eval/check_data.py
~~~

## 在调试集上选配置

三步使用同一个模型、同一批题、相同提示词和 tokenizer。精度为 bfloat16，每批最多 25 题，随机种子为 20261119。每一步保留上一步的选择，只变动要考察的参数。

| 步骤 | 候选 | 要解决的问题 |
|---|---|---|
| 1. 生成长度 | `max_new_tokens=256`、`512`。贪心解码，`repetition_penalty=1.1` | 正常答案能否说完？如果触顶前已在循环复述，归为重复问题 |
| 2. 重复惩罚 | 固定选出的长度，贪心下比较 `1.0`、`1.1` | 能否减少循环兼顾要点覆盖？复用第 1 步的 `1.1` 结果 |
| 3. 解码方式 | 对照选中的贪心配置，采样分别用 `temperature=0.3`、`0.7`，固定 `top_p=0.9` | 采样是否让回答更切题、易读？复用第 2 步的贪心结果 |

共 5 组。贪心解码每一步选择概率最高的 token。采样按概率抽取，温度越低越偏向高概率的词。本轮固定 `num_beams=1` 和 `top_k=50`。

每组统计全部 20 道题的截断和重复，并抽查同样的问题。先看回答有没有解决问题、有没有明显错误，再比较重复、完整性和可读性。重复比例按归一化文字的连续 4 字片段计算。

## 实验记录与选择

以下记录以 2026-11-21 16:41–16:43 重跑的五组结果为准，逐条按[调试集](dev_set.jsonl)行号对齐。重复比例是每题的连续 4 字片段重复比例，再对 20 题取平均。

| 运行 | 模型 | 长度上限 | 惩罚 | 解码方式 | 回答数 | 截断 | 平均重复比例 | 平均输出 tokens |
|---|---|---:|---:|---|---:|---:|---:|---:|
| [greedy_length256_repeat1.1](results/dev_original_lora/greedy_length256_repeat1.1/answers.jsonl) | original_lora | 256 | 1.1 | 贪心 | 20 | 6/20 | 18.0% | 115.8 |
| [greedy_length512_repeat1.1](results/dev_original_lora/greedy_length512_repeat1.1/answers.jsonl) | original_lora | 512 | 1.1 | 贪心 | 20 | 6/20 | 22.3% | 192.7 |
| [greedy_length256_repeat1.0](results/dev_original_lora/greedy_length256_repeat1.0/answers.jsonl) | original_lora | 256 | 1.0 | 贪心 | 20 | 8/20 | 36.9% | 120.6 |
| [temp0.3_length256_repeat1.1](results/dev_original_lora/temp0.3_length256_repeat1.1/answers.jsonl) | original_lora | 256 | 1.1 | 采样 0.3 | 20 | 3/20 | 8.5% | 80.3 |
| [temp0.7_length256_repeat1.1](results/dev_original_lora/temp0.7_length256_repeat1.1/answers.jsonl) | original_lora | 256 | 1.1 | 采样 0.7 | 20 | 1/20 | 4.9% | 73.5 |


1. **长度选 256。** 两组触顶的是同样 6 题（第 1、5、6、7、10、14 题），其余 14 题的回答逐字相同。这 6 题的 512 版本都从 256 版本的末尾继续生成。胃肠间质瘤和脑静脉血栓影像题循环同一术语，低频神经肌和重症胆管炎题继续堆叠护理措辞，面神经损伤题循环列举颈椎检查，微波治疗前列腺炎题继续列举无关疾病，后段也出现循环。放宽上限后，6 题仍全部截断。
2. **重复惩罚选 1.1。** 在相同长度下，平均重复比例从 36.9% 降到 18.0%，截断从 8 题降到 6 题。原先截断的第 2、3、16、18、20 题自然结束，同时第 1、6、10 题新触顶。例如，第 2 题剖腹产后月经不再循环，第 10 题重症胆管炎却从一句“综合护理”变成反复列举护理术语。
3. **采样温度选 0.3。** 与贪心相比，两个温度都减少了截断和重复。0.7 的自动指标更低，但第 2 题绕开了剖腹产后的月经时间，第 11 题从婴儿排便转去讲辅食且前后给出不同的添加时间，第 19 题转向另一诊断和用药。0.3 在这三题更贴近所问，但第 2 题只笼统给出恢复时间，第 11 题未明确回答排便是否正常，第 16 题也未回答术后是否需要化疗。结合这些例子，选择 0.3。

选定值已写入 [eval_config.json](eval_config.json)：`max_new_tokens=256`、`repetition_penalty=1.1`、`do_sample=true`、`temperature=0.3`、`top_p=0.9`、`top_k=50`，种子 20261119。本结论来自 `original_lora` 在调试集上的表现。

## 复现实验

按第 1 步、第 2 步、第 3 步的顺序执行。选定值已是脚本默认值，复现贪心组时显式传入 `--greedy`。

~~~powershell
python model_eval/run_eval.py --split dev --model original_lora --greedy --max-new-tokens 256 --repetition-penalty 1.1
python model_eval/run_eval.py --split dev --model original_lora --greedy --max-new-tokens 512 --repetition-penalty 1.1
python model_eval/run_eval.py --split dev --model original_lora --greedy --max-new-tokens 256 --repetition-penalty 1.0
python model_eval/run_eval.py --split dev --model original_lora --temperature 0.3 --max-new-tokens 256 --repetition-penalty 1.1 
python model_eval/run_eval.py --split dev --model original_lora --temperature 0.7 --max-new-tokens 256 --repetition-penalty 1.1 
~~~

每次运行目录都包含 `answers.jsonl` 和记录实际参数的 `run_info.json`。将多个目录传给 `analyze_results.py --run-dir` 可以生成对照表。
~~~powershell
python model_eval/analyze_results.py --run-dir `
    model_eval/results/dev_original_lora/greedy_length256_repeat1.1 `
    model_eval/results/dev_original_lora/greedy_length512_repeat1.1 `
    model_eval/results/dev_original_lora/greedy_length256_repeat1.0 `
    model_eval/results/dev_original_lora/temp0.3_length256_repeat1.1 `
    model_eval/results/dev_original_lora/temp0.7_length256_repeat1.1
~~~

## 正式评分规则

正式评测时，先用冻结配置生成回答，再按下面四项逐题评分。每项 0—2 分，严重错误单列。

| 维度 | 2 分 | 1 分 | 0 分 |
|---|---|---|---|
| 事实正确性 | 未发现可确认的错误 | 局部不准确 | 关键结论或行动建议错误 |
| 要点覆盖 | 回答核心问题并覆盖关键要点 | 漏掉部分要点 | 错答、跑题或只复述提问 |
| 重复控制 | 没有明显冗余 | 局部重复 | 连续循环或大段重复 |
| 格式可读性 | 清楚易读 | 结构有局部问题 | 难以读懂或无法使用 |

截断另记一项。危险建议单独标记，保留有问题的原文和判定理由。参考答案有疑问时，记录为待复核。

## 正式测试结果

使用调试集选定的配置，对 `base`、`original_lora` 和 `rank8_5000` 运行同一份 100 题正式测试集。三组共 300 条回答，均已逐题记录。本轮配置见 [run_info.json](results/test_all_models/temp0.3_length256_repeat1.1/run_info.json)。

| 模型 | 截断 | 平均重复比例 | 45 道同题平均总分 / 8 | 已标记严重错误 |
|---|---:|---:|---:|---:|
| base | 42/100 | 2.5% | 5.51 | 4 |
| original_lora | 18/100 | 10.1% | 4.73 | 8 |
| rank8_5000 | 29/100 | 19.4% | 4.04 | 9 |

将三个模型四项均可评分的 45 道同题用于总分对比，依据不足的单项标为待复核。Base 的要点覆盖和同题总分最高；`original_lora` 截断最少，但存在短答漏掉题目核心的情况；`rank8_5000` 的循环复述更多。详见 [正式测试报告](results/test_all_models/temp0.3_length256_repeat1.1/test_review.md)、[原始回答](results/test_all_models/temp0.3_length256_repeat1.1/answers.jsonl)、[逐题评分](results/test_all_models/temp0.3_length256_repeat1.1/review_scores.jsonl)和[自动统计](results/test_all_models/temp0.3_length256_repeat1.1/test_comparison.md)。

在项目根目录用 PowerShell 可复现正式推理和自动统计。

~~~powershell
conda activate minilora
python model_eval/run_eval.py --split test 
$run = "model_eval/results/test_all_models/temp0.3_length256_repeat1.1"
python model_eval/analyze_results.py --run-dir $run --output-file "$run/test_comparison.md"
~~~
