import torch


print("PyTorch版本:", torch.__version__)
print("CUDA版本:", torch.version.cuda)
print("CUDA是否可用:", torch.cuda.is_available())