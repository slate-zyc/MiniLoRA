| 运行 | 模型 | 长度上限 | 惩罚 | 解码方式 | 回答数 | 截断 | 平均重复比例 | 平均输出 tokens |
|---|---|---:|---:|---|---:|---:|---:|---:|
| greedy_length256_repeat1.1 | original_lora | 256 | 1.1 | 贪心 | 20 | 6/20 | 18.0% | 115.8 |
| greedy_length512_repeat1.1 | original_lora | 512 | 1.1 | 贪心 | 20 | 6/20 | 22.3% | 192.7 |
| greedy_length256_repeat1.0 | original_lora | 256 | 1.0 | 贪心 | 20 | 8/20 | 36.9% | 120.6 |
| temp0.3_length256_repeat1.1 | original_lora | 256 | 1.1 | 采样 0.3 | 20 | 3/20 | 8.5% | 80.3 |
| temp0.7_length256_repeat1.1 | original_lora | 256 | 1.1 | 采样 0.7 | 20 | 1/20 | 4.9% | 73.5 |
