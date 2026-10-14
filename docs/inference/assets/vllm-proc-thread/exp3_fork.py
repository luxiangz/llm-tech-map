"""实验 3：fork 出现在 CUDA 初始化之后会怎样（这正是 vLLM 强制 spawn 的原因）。"""
import multiprocessing as mp
import os, torch

torch.cuda.init()
x = torch.zeros(1024 * 1024, device="cuda")
print(f"[父进程] pid={os.getpid()} CUDA 已初始化, mem_allocated={torch.cuda.memory_allocated()}")


def child():
    print(f"[fork 子进程] pid={os.getpid()} 尝试使用 CUDA ...")
    try:
        y = torch.ones(8, device="cuda")
        print(f"[fork 子进程] 居然成功了: {y.sum().item()}")
    except Exception as e:
        print(f"[fork 子进程] 失败: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    ctx = mp.get_context("fork")
    p = ctx.Process(target=child)
    p.start()
    p.join()
    print(f"[父进程] fork 子进程 exitcode={p.exitcode}")
    print(f"[父进程] 自己仍可用 CUDA: x.sum={x.sum().item()}")
