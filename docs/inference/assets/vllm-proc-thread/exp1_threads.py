"""实验 1：同一个进程内的多个线程，CUDA 上下文/显存是共享的，而 current stream 是线程私有的。"""
import os, threading, torch

torch.cuda.init()
torch.cuda.set_device(0)
x = torch.zeros(8, device="cuda")
main = {
    "pid": os.getpid(),
    "thread_ident": threading.get_ident(),
    "current_device": torch.cuda.current_device(),
    "mem_allocated": torch.cuda.memory_allocated(),
}
s = torch.cuda.Stream()
out = {}


def worker():
    out["pid"] = os.getpid()
    out["thread_ident"] = threading.get_ident()
    out["current_device"] = torch.cuda.current_device()
    out["mem_allocated_seen_from_thread"] = torch.cuda.memory_allocated()
    out["current_stream_is_main_custom_stream"] = torch.cuda.current_stream() == s
    out["current_stream_is_default"] = torch.cuda.current_stream() == torch.cuda.default_stream()
    x.add_(1)                       # 直接用主线程创建的张量：同一个 CUDA context
    out["parent_tensor_sum"] = x.sum().item()
    out["same_context_ptr_ok"] = True


with torch.cuda.stream(s):
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    main["current_stream_is_custom"] = torch.cuda.current_stream() == s

print("=== 主线程 ===")
for k, v in main.items():
    print(f"  {k} = {v}")
print("=== 子线程（threading.Thread）===")
for k, v in out.items():
    print(f"  {k} = {v}")
print("结论: pid 相同 => 同一个进程; mem_allocated 相同 => 显存/上下文进程内共享;",
      "子线程的 current_stream 不是主线程设的 s => 流是线程私有状态")
