"""实验 6：host 地址空间 与 device 地址空间是两回事（单卡也能证明）。

要证明：
  1) host 张量与 device 张量的 data_ptr 落在完全不同的地址区间；
  2) "线程共享进程地址空间"共享的是 (1) 里的 host 那部分，跟 HBM 无关；
  3) host 代码直接解引用 device 指针会失败（这里用子进程隔离，失败即证据）；
  4) device 之间的可访问性是一个要显式查询/开启的能力。
"""
import ctypes
import os
import subprocess
import sys

import torch

print(f"设备数 = {torch.cuda.device_count()}")
print(f"CUDA 可用 peer 访问矩阵: ", end="")
n = torch.cuda.device_count()
print([[torch.cuda.can_device_access_peer(i, j) for j in range(n)] for i in range(n)])

h = torch.zeros(8)                     # host（可分页内存）
p = torch.zeros(8, pin_memory=True)    # host（锁页内存）
d = torch.zeros(8, device="cuda")      # device HBM

print(f"\nhost  可分页张量 data_ptr = {hex(h.data_ptr())}")
print(f"host  锁页张量   data_ptr = {hex(p.data_ptr())}")
print(f"device HBM 张量  data_ptr = {hex(d.data_ptr())}")
print(f"host/device 指针差 = {abs(d.data_ptr() - h.data_ptr()) / 2**40:.2f} TiB 量级差距")

print("\n--- 现在换到子进程里，尝试把 device 指针当 host 内存读（预期失败）---")
child = r'''
import ctypes, torch
d = torch.zeros(8, device="cuda")
ptr = d.data_ptr()
print(f"  子进程 device data_ptr = {hex(ptr)}，尝试 ctypes.string_at(ptr, 4) ...")
try:
    raw = ctypes.string_at(ptr, 4)
    print(f"  居然读到了: {raw!r}  <-- 在 host 上直接解引用 device 指针")
except Exception as e:
    print(f"  失败（符合预期）: {type(e).__name__}: {e}")
'''
r = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=120,
                   env={**os.environ, "PYTHONIOENCODING": "utf-8"})
print((r.stdout or "").strip())
if (r.stderr or "").strip():
    print("  子进程 stderr:", r.stderr.strip().splitlines()[-1][:200])
print(f"  子进程 exitcode = {r.returncode}")

print("\n--- 反过来：host 指针可以被 device 直接访问吗？---")
print(f"  可分页内存 h.data_ptr() = {hex(h.data_ptr())}：device 不能直接解引用，只能靠 H2D 拷贝")
print(f"  锁页内存 p.data_ptr()  = {hex(p.data_ptr())}：注册后可零拷贝映射给 device（UVA 下可解引用）")
