# LoRA 训练报错排查与解决记录

记录日期：2026-11-18。**结果：增加torch_empty_cache_steps=10后，第三轮完成625/625、epoch=1，并输出LoRA adapter保存信息。**

修改顺序按本次手动操作记录整理；运行数值、错误与完成状态均以附录中的三份原始输出为依据。日志没有给出的底层原因不作推断。

## 1. 问题与观察方式

三轮均执行python scripts/train_lora.py。日志中的共同信息如下：

| 项目 | 三轮日志中的输出 |
| --- | --- |
| 训练 / 验证数据预处理数量 | 5000 / 300 |
| 可训练参数 / 总参数 | 4,399,104 / 498,431,872 |
| 可训练参数比例 | 0.8826% |
| 最长训练输入 | 1024 tokens |
| 达到1024 tokens的训练样本数 | 19 |
| 进度条中的计划更新次数 | 625 |

我在原脚本的数据预处理后打印输入长度统计；在trainer.train()前重置峰值统计，并在finally中打印峰值、当前allocated和reserved。第三轮还打印了memory_summary。

## 2. 手动排查过程

**第一轮：记录原始错误。** 训练在152/625处中断，输出RuntimeError: CUDA error: out of memory。调用栈经过accelerator.backward、loss.backward和autograd，记录为“在反向传播中报告OOM”。

**第二轮：开启梯度检查点后重新运行。** 在TrainingArguments中开启：

~~~python
activation_checkpointing=True,
~~~

这一轮仍在152/625处中断，但错误变为CUBLAS_STATUS_INTERNAL_ERROR。调用栈经过compute_loss、model.forward、k_proj、LoRA和F.linear，记录为“前向矩阵乘法中报告cuBLAS内部错误”。峰值为2855.75 MiB，低于第一轮，但仍未完成训练；这里保留原始错误类型，不将它改写成CUDA OOM。

**补充：如何理解这个cuBLAS错误。**

cuBLAS是NVIDIA提供的GPU线性代数库，负责矩阵乘法等计算。第二轮原始调用栈显示的报错路径可以概括为：

~~~text
LoRA中的F.linear → cuBLAS的cublasGemmEx → CUBLAS_STATUS_INTERNAL_ERROR
~~~

| 日志中的名称 | 含义 |
| --- | --- |
| cuBLAS | 执行GPU线性代数计算的底层库 |
| cublasGemmEx | cuBLAS的矩阵乘法接口 |
| CUDA_R_16BF | 这次调用中出现的BF16数据类型标记 |
| CUBLAS_STATUS_INTERNAL_ERROR | cuBLAS内部某个操作失败，错误码本身未指出本次失败的具体原因 |

因此，这条日志说明“底层矩阵乘法调用报告了内部错误”，不能直接等同于显存不足，也不能据此认定BF16、LoRA或驱动本身存在错误。它给出了报告失败的调用位置，没有直接证明底层根因。

第三轮增加torch_empty_cache_steps=10后成功完成训练，是日志能够确认的处理结果；不能仅凭运行成功，反推出第二轮内部错误必然由某一种底层原因造成。

以上为[官方cuBLAS文档](https://docs.nvidia.com/cuda/archive/12.4.1/cublas/index.html)中的术语说明；本次运行的结果判断仍以原始输出为依据。



**第三轮：保留检查点，增加周期清缓存配置。** 在第二轮配置基础上增加：

~~~python
torch_empty_cache_steps=10,
~~~

重新运行后，日志继续越过原先的152步，最终显示625/625、epoch=1，并输出LoRA adapter saved to outputs\qwen-medical-lora。

本次工作过程是：先记录失败阶段和显存，再开启检查点观察变化，最后增加上述配置并完成整轮训练。

## 3. 三轮实测对照

| 指标 | 第一轮：原始运行 | 第二轮：开启检查点 | 第三轮：再增加周期清缓存 |
| --- | ---: | ---: | ---: |
| 完成更新次数 | 152/625 | 152/625 | 625/625 |
| 运行结果 | CUDA OOM中断 | cuBLAS内部错误中断 | 完成训练并输出保存信息 |
| 峰值allocated（MiB） | 5771.70 | 2855.75 | 2854.12 |
| 打印时allocated（MiB） | 4879.19 | 1028.79 | 1008.92 |
| 打印时reserved（MiB） | 12618.0 | 11600.0 | 4058.0 |
| reserved峰值（MiB） | 未输出 | 未输出 | 8736 |

打印时allocated在表中保留两位小数，完整精度见原文。前两轮在异常退出时打印，第三轮在训练结束时打印；这些快照时点不同，不将它们当成同一阶段的峰值对照。

## 4. 成功运行的直接证据

第三轮日志显示：

| 项目 | 日志记录 |
| --- | --- |
| 最终进度 / epoch | 625/625 / 1 |
| train_runtime | 2218 |
| train_loss | 2.734 |
| 最后一次eval_loss / epoch | 2.681 / 1 |
| CUDA OOMs | 0 |
| cudaMalloc retries | 0 |
| GPU reserved memory：当前 / 峰值 | 4058 MiB / 8736 MiB |
| 保存信息 | LoRA adapter saved to outputs\qwen-medical-lora |

从三轮输出能确认：第二轮的张量峰值低于第一轮，但仍失败；第三轮峰值与第二轮接近，记录的结束reserved较低，并完整完成训练和保存流程。**本次配置下，此前的训练中断问题未再出现。**

日志没有直接给出底层根因，因此不将结果进一步写成已证明某种底层故障；也不据训练完成或loss数值推断生成质量提升。

## 5. 根据实际过程组织面试表达

> 我先在原训练脚本中加入输入长度和显存统计。原始运行在152/625处报告OOM；开启梯度检查点后，峰值从5771.70 MiB降到2855.75 MiB，但仍在152步遇到cuBLAS内部错误。随后我保留检查点，增加torch_empty_cache_steps=10，重新运行完成了625/625、epoch=1，并输出adapter保存信息。最终峰值为2854.12 MiB，结束时reserved为4058 MiB，memory_summary中的CUDA OOM计数为0。这一轮完成了整轮训练；底层原因以日志能够证明的范围为限。

## 6. 三轮完整原始日志

以下内容按原文保留，包含进度条、警告、统计和完整调用栈。折叠展示仅便于阅读，不删改日志内容。


<!-- raw-log:A -->
<details>
<summary>第一轮：原始运行，152步报告CUDA OOM</summary>

~~~text
(E:\conda-envs\minilora) PS D:\LLMProjects\MiniLoRA> python scripts/train_lora.py                                                                                            
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 340.39it/s]
E:\conda-envs\minilora\Lib\site-packages\transformers\modeling_utils.py:3303: FutureWarning: `is_gradient_checkpointing` is deprecated and will be removed in v6. Use `is_activation_checkpointing` instead.
  warnings.warn(
trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1161.74 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:00<00:00, 1482.20 examples/s]
最长训练输入： 1024
长度 >= 256 的样本数： 968
长度 >= 512 的样本数： 303
长度 >= 768 的样本数： 36
长度 >= 1024 的样本数： 19
{'loss': '2.74', 'grad_norm': '1.238', 'learning_rate': '0.0001971', 'epoch': '0.016'}                                                                                                                                                 
{'loss': '2.923', 'grad_norm': '1.168', 'learning_rate': '0.0001939', 'epoch': '0.032'}                                                                                                                                                
{'loss': '2.937', 'grad_norm': '1.227', 'learning_rate': '0.0001907', 'epoch': '0.048'}                                                                                                                                                
{'loss': '2.75', 'grad_norm': '0.8979', 'learning_rate': '0.0001875', 'epoch': '0.064'}                                                                                                                                                
{'loss': '2.823', 'grad_norm': '1.749', 'learning_rate': '0.0001843', 'epoch': '0.08'}                                                                                                                                                 
{'loss': '2.695', 'grad_norm': '1.05', 'learning_rate': '0.0001811', 'epoch': '0.096'}                                                                                                                                                 
{'loss': '2.812', 'grad_norm': '1.341', 'learning_rate': '0.0001779', 'epoch': '0.112'}                                                                                                                                                
{'loss': '2.848', 'grad_norm': '0.8099', 'learning_rate': '0.0001747', 'epoch': '0.128'}                                                                                                                                               
{'loss': '2.595', 'grad_norm': '1.799', 'learning_rate': '0.0001715', 'epoch': '0.144'}                                                                                                                                                
{'loss': '2.828', 'grad_norm': '1.314', 'learning_rate': '0.0001683', 'epoch': '0.16'}                                                                                                                                                 
{'eval_loss': '2.773', 'eval_runtime': '32.03', 'eval_samples_per_second': '9.366', 'eval_steps_per_second': '9.366', 'epoch': '0.16'}                                                                                                 
{'loss': '2.821', 'grad_norm': '1.284', 'learning_rate': '0.0001651', 'epoch': '0.176'}                                                                                                                                                
{'loss': '2.615', 'grad_norm': '0.8257', 'learning_rate': '0.0001619', 'epoch': '0.192'}                                                                                                                                               
{'loss': '2.651', 'grad_norm': '1.508', 'learning_rate': '0.0001587', 'epoch': '0.208'}                                                                                                                                                
{'loss': '2.817', 'grad_norm': '1.335', 'learning_rate': '0.0001555', 'epoch': '0.224'}                                                                                                                                                
{'loss': '2.717', 'grad_norm': '1.196', 'learning_rate': '0.0001523', 'epoch': '0.24'}                                                                                                                                                 
 24%|██████████████████████████████████████████████▋                                                                                                                                                 | 152/625 [06:11<13:39,  1.73s/it] 训练峰值显存：5771.70 MiB
当前 allocated： 4879.19482421875 MiB
当前 reserved： 12618.0 MiB
Traceback (most recent call last):
  File "D:\LLMProjects\MiniLoRA\scripts\train_lora.py", line 452, in <module>
    main()
  File "D:\LLMProjects\MiniLoRA\scripts\train_lora.py", line 430, in main
    trainer.train()
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 1576, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 1660, in _inner_training_loop
    self._run_epoch(
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 1909, in _run_epoch
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 2118, in training_step
    self.accelerator.backward(loss, **kwargs)
  File "E:\conda-envs\minilora\Lib\site-packages\accelerate\accelerator.py", line 2861, in backward
    loss.backward(**kwargs)
  File "E:\conda-envs\minilora\Lib\site-packages\torch\_tensor.py", line 626, in backward
    torch.autograd.backward(
  File "E:\conda-envs\minilora\Lib\site-packages\torch\autograd\__init__.py", line 347, in backward
    _engine_run_backward(
  File "E:\conda-envs\minilora\Lib\site-packages\torch\autograd\graph.py", line 823, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: out of memory
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

~~~

</details>
<!-- /raw-log:A -->


<!-- raw-log:B -->
<details>
<summary>第二轮：开启检查点，152步报告cuBLAS内部错误</summary>

~~~text
(E:\conda-envs\minilora) PS D:\LLMProjects\MiniLoRA> python scripts/train_lora.py
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:01<00:00, 257.05it/s]
E:\conda-envs\minilora\Lib\site-packages\transformers\modeling_utils.py:3303: FutureWarning: `is_gradient_checkpointing` is deprecated and will be removed in v6. Use `is_activation_checkpointing` instead.
  warnings.warn(
trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
Map: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:05<00:00, 993.52 examples/s]
Map: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:00<00:00, 701.38 examples/s]
最长训练输入： 1024
长度 >= 256 的样本数： 968
长度 >= 512 的样本数： 303
长度 >= 768 的样本数： 36
长度 >= 1024 的样本数： 19
{'loss': '2.74', 'grad_norm': '1.229', 'learning_rate': '0.0001971', 'epoch': '0.016'}                                                                                                                                                 
{'loss': '2.926', 'grad_norm': '1.145', 'learning_rate': '0.0001939', 'epoch': '0.032'}                                                                                                                                                
{'loss': '2.94', 'grad_norm': '1.196', 'learning_rate': '0.0001907', 'epoch': '0.048'}                                                                                                                                                 
{'loss': '2.751', 'grad_norm': '0.8929', 'learning_rate': '0.0001875', 'epoch': '0.064'}                                                                                                                                               
{'loss': '2.825', 'grad_norm': '1.617', 'learning_rate': '0.0001843', 'epoch': '0.08'}                                                                                                                                                 
{'loss': '2.694', 'grad_norm': '1.065', 'learning_rate': '0.0001811', 'epoch': '0.096'}                                                                                                                                                
{'loss': '2.815', 'grad_norm': '1.33', 'learning_rate': '0.0001779', 'epoch': '0.112'}                                                                                                                                                 
{'loss': '2.846', 'grad_norm': '0.8172', 'learning_rate': '0.0001747', 'epoch': '0.128'}                                                                                                                                               
{'loss': '2.592', 'grad_norm': '1.734', 'learning_rate': '0.0001715', 'epoch': '0.144'}                                                                                                                                                
{'loss': '2.83', 'grad_norm': '1.349', 'learning_rate': '0.0001683', 'epoch': '0.16'}                                                                                                                                                  
{'eval_loss': '2.773', 'eval_runtime': '29.05', 'eval_samples_per_second': '10.33', 'eval_steps_per_second': '10.33', 'epoch': '0.16'}                                                                                                 
{'loss': '2.823', 'grad_norm': '1.319', 'learning_rate': '0.0001651', 'epoch': '0.176'}                                                                                                                                                
{'loss': '2.615', 'grad_norm': '0.8337', 'learning_rate': '0.0001619', 'epoch': '0.192'}                                                                                                                                               
{'loss': '2.653', 'grad_norm': '1.531', 'learning_rate': '0.0001587', 'epoch': '0.208'}                                                                                                                                                
{'loss': '2.819', 'grad_norm': '1.35', 'learning_rate': '0.0001555', 'epoch': '0.224'}                                                                                                                                                 
{'loss': '2.717', 'grad_norm': '1.196', 'learning_rate': '0.0001523', 'epoch': '0.24'}                                                                                                                                                 
 24%|██████████████████████████████████████████████▋                                                                                                                                                 | 152/625 [09:14<24:29,  3.11s/it] 训练峰值显存：2855.75 MiB
当前 allocated： 1028.79443359375 MiB
当前 reserved： 11600.0 MiB
Traceback (most recent call last):
  File "D:\LLMProjects\MiniLoRA\scripts\train_lora.py", line 452, in <module>
    main()
  File "D:\LLMProjects\MiniLoRA\scripts\train_lora.py", line 430, in main
    trainer.train()
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 1576, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 1660, in _inner_training_loop
    self._run_epoch(
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 1909, in _run_epoch
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 2082, in training_step
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\trainer.py", line 2164, in compute_loss
    outputs = model(**inputs)
              ^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\accelerate\utils\operations.py", line 943, in forward
    return model_forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\accelerate\utils\operations.py", line 931, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\amp\autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\peft\peft_model.py", line 2120, in forward
    return self.base_model(
           ^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\peft\tuners\tuners_utils.py", line 359, in forward
    return self.model(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\utils\generic.py", line 943, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\models\qwen2\modeling_qwen2.py", line 444, in forward
    outputs: BaseModelOutputWithPast = self.model(
                                       ^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\utils\generic.py", line 1081, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\utils\output_capturing.py", line 361, in wrapper
    outputs = func(self, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\models\qwen2\modeling_qwen2.py", line 380, in forward
    hidden_states = decoder_layer(
                    ^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\modeling_layers.py", line 120, in forward_with_checkpointing
    return self._gradient_checkpointing_func(partial(forward, self, **kwargs), *args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\_compile.py", line 32, in inner
    return disable_fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\_dynamo\eval_frame.py", line 745, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\utils\checkpoint.py", line 496, in checkpoint
    ret = function(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\models\qwen2\modeling_qwen2.py", line 274, in forward
    hidden_states, _ = self.self_attn(
                       ^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\transformers\models\qwen2\modeling_qwen2.py", line 199, in forward
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\peft\tuners\lora\layer.py", line 1102, in forward
    result = result + lora_B(lora_A(dropout(x))) * scaling
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\conda-envs\minilora\Lib\site-packages\torch\nn\modules\linear.py", line 125, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasGemmEx( handle, opa, opb, m, n, k, &falpha, a, CUDA_R_16BF, lda, b, CUDA_R_16BF, ldb, &fbeta, c, CUDA_R_16BF, ldc, compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`
 24%|██████████████████████████████████████████████▋  
~~~

</details>
<!-- /raw-log:B -->


<!-- raw-log:C -->
<details>
<summary>第三轮：增加周期清缓存，完成625步</summary>

~~~text
(E:\conda-envs\minilora) PS D:\LLMProjects\MiniLoRA> python scripts/train_lora.py                                                              
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 290/290 [00:00<00:00, 315.47it/s]
E:\conda-envs\minilora\Lib\site-packages\transformers\modeling_utils.py:3303: FutureWarning: `is_gradient_checkpointing` is deprecated and will be removed in v6. Use `is_activation_checkpointing` instead.
  warnings.warn(
trainable params: 4,399,104 || all params: 498,431,872 || trainable%: 0.8826
Map: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:03<00:00, 1498.73 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 300/300 [00:00<00:00, 1606.10 examples/s]
最长训练输入： 1024
长度 >= 256 的样本数： 968
长度 >= 512 的样本数： 303
长度 >= 768 的样本数： 36
长度 >= 1024 的样本数： 19
{'loss': '2.741', 'grad_norm': '1.231', 'learning_rate': '0.0001971', 'epoch': '0.016'}                                                                                                                                                
{'loss': '2.927', 'grad_norm': '1.132', 'learning_rate': '0.0001939', 'epoch': '0.032'}                                                                                                                                                
{'loss': '2.94', 'grad_norm': '1.202', 'learning_rate': '0.0001907', 'epoch': '0.048'}                                                                                                                                                 
{'loss': '2.752', 'grad_norm': '0.9091', 'learning_rate': '0.0001875', 'epoch': '0.064'}                                                                                                                                               
{'loss': '2.823', 'grad_norm': '1.641', 'learning_rate': '0.0001843', 'epoch': '0.08'}                                                                                                                                                 
{'loss': '2.692', 'grad_norm': '1.063', 'learning_rate': '0.0001811', 'epoch': '0.096'}                                                                                                                                                
{'loss': '2.811', 'grad_norm': '1.308', 'learning_rate': '0.0001779', 'epoch': '0.112'}                                                                                                                                                
{'loss': '2.846', 'grad_norm': '0.8129', 'learning_rate': '0.0001747', 'epoch': '0.128'}                                                                                                                                               
{'loss': '2.593', 'grad_norm': '1.766', 'learning_rate': '0.0001715', 'epoch': '0.144'}                                                                                                                                                
{'loss': '2.826', 'grad_norm': '1.359', 'learning_rate': '0.0001683', 'epoch': '0.16'}                                                                                                                                                 
{'eval_loss': '2.775', 'eval_runtime': '30.24', 'eval_samples_per_second': '9.921', 'eval_steps_per_second': '9.921', 'epoch': '0.16'}                                                                                                 
{'loss': '2.825', 'grad_norm': '1.308', 'learning_rate': '0.0001651', 'epoch': '0.176'}                                                                                                                                                
{'loss': '2.618', 'grad_norm': '0.8315', 'learning_rate': '0.0001619', 'epoch': '0.192'}                                                                                                                                               
{'loss': '2.651', 'grad_norm': '1.513', 'learning_rate': '0.0001587', 'epoch': '0.208'}                                                                                                                                                
{'loss': '2.819', 'grad_norm': '1.358', 'learning_rate': '0.0001555', 'epoch': '0.224'}                                                                                                                                                
{'loss': '2.719', 'grad_norm': '1.188', 'learning_rate': '0.0001523', 'epoch': '0.24'}                                                                                                                                                 
{'loss': '2.8', 'grad_norm': '1.06', 'learning_rate': '0.0001491', 'epoch': '0.256'}                                                                                                                                                   
{'loss': '2.688', 'grad_norm': '1.091', 'learning_rate': '0.0001459', 'epoch': '0.272'}                                                                                                                                                
{'loss': '2.655', 'grad_norm': '0.9101', 'learning_rate': '0.0001427', 'epoch': '0.288'}                                                                                                                                               
{'loss': '2.964', 'grad_norm': '1.214', 'learning_rate': '0.0001395', 'epoch': '0.304'}                                                                                                                                                
{'loss': '2.548', 'grad_norm': '1.439', 'learning_rate': '0.0001363', 'epoch': '0.32'}                                                                                                                                                 
{'eval_loss': '2.711', 'eval_runtime': '32.98', 'eval_samples_per_second': '9.097', 'eval_steps_per_second': '9.097', 'epoch': '0.32'}                                                                                                 
{'loss': '2.841', 'grad_norm': '0.8982', 'learning_rate': '0.0001331', 'epoch': '0.336'}                                                                                                                                               
{'loss': '2.726', 'grad_norm': '1.742', 'learning_rate': '0.0001299', 'epoch': '0.352'}                                                                                                                                                
{'loss': '2.727', 'grad_norm': '0.9191', 'learning_rate': '0.0001267', 'epoch': '0.368'}                                                                                                                                               
{'loss': '2.89', 'grad_norm': '0.7601', 'learning_rate': '0.0001235', 'epoch': '0.384'}                                                                                                                                                
{'loss': '2.699', 'grad_norm': '0.8494', 'learning_rate': '0.0001203', 'epoch': '0.4'}                                                                                                                                                 
{'loss': '2.539', 'grad_norm': '2.161', 'learning_rate': '0.0001171', 'epoch': '0.416'}                                                                                                                                                
{'loss': '2.697', 'grad_norm': '1.179', 'learning_rate': '0.0001139', 'epoch': '0.432'}                                                                                                                                                
{'loss': '2.782', 'grad_norm': '0.7545', 'learning_rate': '0.0001107', 'epoch': '0.448'}                                                                                                                                               
{'loss': '2.812', 'grad_norm': '0.9724', 'learning_rate': '0.0001075', 'epoch': '0.464'}                                                                                                                                               
{'loss': '2.826', 'grad_norm': '1.52', 'learning_rate': '0.0001043', 'epoch': '0.48'}                                                                                                                                                  
{'eval_loss': '2.703', 'eval_runtime': '23.07', 'eval_samples_per_second': '13.01', 'eval_steps_per_second': '13.01', 'epoch': '0.48'}                                                                                                 
{'loss': '2.932', 'grad_norm': '0.8648', 'learning_rate': '0.0001011', 'epoch': '0.496'}                                                                                                                                               
{'loss': '2.687', 'grad_norm': '0.7354', 'learning_rate': '9.792e-05', 'epoch': '0.512'}                                                                                                                                               
{'loss': '2.726', 'grad_norm': '1.34', 'learning_rate': '9.472e-05', 'epoch': '0.528'}                                                                                                                                                 
{'loss': '2.538', 'grad_norm': '1.102', 'learning_rate': '9.152e-05', 'epoch': '0.544'}                                                                                                                                                
{'loss': '2.787', 'grad_norm': '0.8983', 'learning_rate': '8.832e-05', 'epoch': '0.56'}                                                                                                                                                
{'loss': '2.718', 'grad_norm': '1.566', 'learning_rate': '8.512e-05', 'epoch': '0.576'}                                                                                                                                                
{'loss': '2.735', 'grad_norm': '0.7125', 'learning_rate': '8.192e-05', 'epoch': '0.592'}                                                                                                                                               
{'loss': '2.705', 'grad_norm': '0.9578', 'learning_rate': '7.872e-05', 'epoch': '0.608'}                                                                                                                                               
{'loss': '2.461', 'grad_norm': '1.035', 'learning_rate': '7.552e-05', 'epoch': '0.624'}                                                                                                                                                
{'loss': '2.75', 'grad_norm': '1.794', 'learning_rate': '7.232e-05', 'epoch': '0.64'}                                                                                                                                                  
{'eval_loss': '2.684', 'eval_runtime': '41.58', 'eval_samples_per_second': '7.215', 'eval_steps_per_second': '7.215', 'epoch': '0.64'}                                                                                                 
{'loss': '2.814', 'grad_norm': '0.8616', 'learning_rate': '6.912e-05', 'epoch': '0.656'}                                                                                                                                               
{'loss': '2.805', 'grad_norm': '0.8877', 'learning_rate': '6.592e-05', 'epoch': '0.672'}                                                                                                                                               
{'loss': '2.81', 'grad_norm': '0.8066', 'learning_rate': '6.272e-05', 'epoch': '0.688'}                                                                                                                                                
{'loss': '2.642', 'grad_norm': '1.169', 'learning_rate': '5.952e-05', 'epoch': '0.704'}                                                                                                                                                
{'loss': '2.657', 'grad_norm': '0.8166', 'learning_rate': '5.632e-05', 'epoch': '0.72'}                                                                                                                                                
{'loss': '2.641', 'grad_norm': '1.232', 'learning_rate': '5.312e-05', 'epoch': '0.736'}                                                                                                                                                
{'loss': '2.604', 'grad_norm': '1.798', 'learning_rate': '4.992e-05', 'epoch': '0.752'}                                                                                                                                                
{'loss': '2.71', 'grad_norm': '1.017', 'learning_rate': '4.672e-05', 'epoch': '0.768'}                                                                                                                                                 
{'loss': '2.774', 'grad_norm': '1.697', 'learning_rate': '4.352e-05', 'epoch': '0.784'}                                                                                                                                                
{'loss': '2.613', 'grad_norm': '1.164', 'learning_rate': '4.032e-05', 'epoch': '0.8'}                                                                                                                                                  
{'eval_loss': '2.708', 'eval_runtime': '21.23', 'eval_samples_per_second': '14.13', 'eval_steps_per_second': '14.13', 'epoch': '0.8'}                                                                                                  
{'loss': '2.661', 'grad_norm': '1.543', 'learning_rate': '3.712e-05', 'epoch': '0.816'}                                                                                                                                                
{'loss': '2.686', 'grad_norm': '1.005', 'learning_rate': '3.392e-05', 'epoch': '0.832'}                                                                                                                                                
{'loss': '2.71', 'grad_norm': '1.006', 'learning_rate': '3.072e-05', 'epoch': '0.848'}                                                                                                                                                 
{'loss': '2.776', 'grad_norm': '0.8763', 'learning_rate': '2.752e-05', 'epoch': '0.864'}                                                                                                                                               
{'loss': '2.713', 'grad_norm': '1.32', 'learning_rate': '2.432e-05', 'epoch': '0.88'}                                                                                                                                                  
{'loss': '2.885', 'grad_norm': '0.6903', 'learning_rate': '2.112e-05', 'epoch': '0.896'}                                                                                                                                               
{'loss': '2.66', 'grad_norm': '1.541', 'learning_rate': '1.792e-05', 'epoch': '0.912'}                                                                                                                                                 
{'loss': '2.825', 'grad_norm': '1.944', 'learning_rate': '1.472e-05', 'epoch': '0.928'}                                                                                                                                                
{'loss': '2.625', 'grad_norm': '1.936', 'learning_rate': '1.152e-05', 'epoch': '0.944'}                                                                                                                                                
{'loss': '2.647', 'grad_norm': '1.361', 'learning_rate': '8.32e-06', 'epoch': '0.96'}                                                                                                                                                  
{'eval_loss': '2.679', 'eval_runtime': '21.73', 'eval_samples_per_second': '13.8', 'eval_steps_per_second': '13.8', 'epoch': '0.96'}                                                                                                   
{'loss': '2.735', 'grad_norm': '2.154', 'learning_rate': '5.12e-06', 'epoch': '0.976'}                                                                                                                                                 
{'loss': '2.719', 'grad_norm': '0.9278', 'learning_rate': '1.92e-06', 'epoch': '0.992'}                                                                                                                                                
{'eval_loss': '2.681', 'eval_runtime': '20.66', 'eval_samples_per_second': '14.52', 'eval_steps_per_second': '14.52', 'epoch': '1'}                                                                                                    
{'train_runtime': '2218', 'train_samples_per_second': '2.254', 'train_steps_per_second': '0.282', 'train_loss': '2.734', 'epoch': '1'}                                                                                                 
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [36:57<00:00,  3.55s/it]
训练峰值显存：2854.12 MiB
当前 allocated： 1008.91748046875 MiB
当前 reserved： 4058.0 MiB
|===========================================================================|
|                  PyTorch CUDA memory summary, device ID 0                 |
|---------------------------------------------------------------------------|
|            CUDA OOMs: 0            |        cudaMalloc retries: 0         |
|===========================================================================|
|        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  |
|---------------------------------------------------------------------------|
| Allocated memory      |   1008 MiB |   2854 MiB |  31002 GiB |  31001 GiB |
|       from large pool |    947 MiB |   2775 MiB |  20163 GiB |  20162 GiB |
|       from small pool |     61 MiB |    116 MiB |  10838 GiB |  10838 GiB |
|---------------------------------------------------------------------------|
| Active memory         |   1008 MiB |   2854 MiB |  31002 GiB |  31001 GiB |
|       from large pool |    947 MiB |   2775 MiB |  20163 GiB |  20162 GiB |
|       from small pool |     61 MiB |    116 MiB |  10838 GiB |  10838 GiB |
|---------------------------------------------------------------------------|
| Requested memory      |   1008 MiB |   2852 MiB |  30434 GiB |  30433 GiB |
|       from large pool |    947 MiB |   2773 MiB |  19600 GiB |  19599 GiB |
|       from small pool |     60 MiB |    116 MiB |  10833 GiB |  10833 GiB |
|---------------------------------------------------------------------------|
| GPU reserved memory   |   4058 MiB |   8736 MiB | 365406 MiB | 361348 MiB |
|       from large pool |   3934 MiB |   8614 MiB | 359724 MiB | 355790 MiB |
|       from small pool |    124 MiB |    130 MiB |   5682 MiB |   5558 MiB |
|---------------------------------------------------------------------------|
| Non-releasable memory |  60500 KiB |   1079 MiB |  37917 GiB |  37917 GiB |
|       from large pool |  41056 KiB |   1067 MiB |  26254 GiB |  26254 GiB |
|       from small pool |  19444 KiB |     42 MiB |  11662 GiB |  11662 GiB |
|---------------------------------------------------------------------------|
| Allocations           |    1303    |    2039    |   53668 K  |   53667 K  |
|       from large pool |     123    |     172    |    5914 K  |    5914 K  |
|       from small pool |    1180    |    1914    |   47754 K  |   47753 K  |
|---------------------------------------------------------------------------|
| Active allocs         |    1303    |    2039    |   53668 K  |   53667 K  |
|       from large pool |     123    |     172    |    5914 K  |    5914 K  |
|       from small pool |    1180    |    1914    |   47754 K  |   47753 K  |
|---------------------------------------------------------------------------|
| GPU reserved segments |      71    |      91    |    4444    |    4373    |
|       from large pool |       9    |      27    |    1603    |    1594    |
|       from small pool |      62    |      65    |    2841    |    2779    |
|---------------------------------------------------------------------------|
| Non-releasable allocs |      73    |     268    |   22971 K  |   22971 K  |
|       from large pool |       3    |      15    |    2861 K  |    2861 K  |
|       from small pool |      70    |     263    |   20110 K  |   20110 K  |
|---------------------------------------------------------------------------|
| Oversize allocations  |       0    |       0    |       0    |       0    |
|---------------------------------------------------------------------------|
| Oversize GPU segments |       0    |       0    |       0    |       0    |
|===========================================================================|

LoRA adapter saved to outputs\qwen-medical-lora
~~~

</details>
<!-- /raw-log:C -->
