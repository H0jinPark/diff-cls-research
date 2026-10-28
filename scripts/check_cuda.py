"""Fail before a long run if the selected CUDA device cannot execute PyTorch kernels."""
import sys
import torch

print('PyTorch:', torch.__version__)
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit('No CUDA GPU visible. Check your GPU allocation and CUDA-enabled PyTorch install.')
try:
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
    print('PyTorch architectures:', torch.cuda.get_arch_list())
    a = torch.randn(256, 256, device='cuda')
    b = a @ a
    torch.cuda.synchronize()
    print('GPU matrix multiplication:', float(b.mean()))
except Exception as exc:
    sys.exit(f'CUDA kernel test failed: {exc}')
print('CUDA execution check passed.')
