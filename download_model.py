from huggingface_hub import snapshot_download

# 官方模型仓库
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 指定模型保存位置
local_dir = r"D:\LLMProjects\LLMModels\Qwen2.5-0.5B-Instruct"

# 下载整个模型仓库
model_path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir
)

print("模型下载完成！")
print(f"模型位置：{model_path}")