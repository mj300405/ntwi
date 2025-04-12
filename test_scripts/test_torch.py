import torch
import time

# Check if MPS (Metal Performance Shaders) is available
print(f"MPS (Metal) available: {torch.backends.mps.is_available()}")
print(f"PyTorch version: {torch.__version__}")

# Create tensors on CPU and MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Simple matrix multiplication test
size = 2000
a = torch.randn(size, size)
b = torch.randn(size, size)

# Test on CPU
start = time.time()
c = torch.mm(a, b)
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

# Test on MPS (Metal)
if torch.backends.mps.is_available():
    a = a.to(device)
    b = b.to(device)
    
    start = time.time()
    c = torch.mm(a, b)
    mps_time = time.time() - start
    print(f"MPS (Metal) time: {mps_time:.4f} seconds")
    if cpu_time > mps_time:
        print(f"Metal is {cpu_time/mps_time:.2f}x faster than CPU!") 