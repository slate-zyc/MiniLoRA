import torch

print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())
print("GPU数量：", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU名称：", torch.cuda.get_device_name(0))
    print("是否支持BF16：", torch.cuda.is_bf16_supported())