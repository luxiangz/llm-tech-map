"""实验 2（修正版）：spawn 子进程 = 全新解释器 + 全新 CUDA context。

注意：spawn 的子进程会【重新 import 本模块】并执行所有模块级代码。
所以父进程专属的东西必须放进 `if __name__ == "__main__":`，否则会被子进程再跑一遍。
这正是 vLLM 要求 spawn + __main__ guard 的原因。
"""
import multiprocessing as mp
import os
import threading
import torch

MARK = os.environ.get("EXP_ROLE", "?")   # 由外部设置，用来区分是谁在打印


def child(tag: str) -> None:
    print(f"[{tag}] pid={os.getpid()} ppid={os.getppid()} 线程ident={threading.get_ident()} "
          f"__name__={__name__}", flush=True)
    print(f"[{tag}] 进入子进程时 mem_allocated={torch.cuda.memory_allocated()} "
          f"devices={torch.cuda.device_count()}", flush=True)
    torch.cuda.set_device(0)
    y = torch.ones(8, device="cuda")
    print(f"[{tag}] set_device(0) 后 current_device={torch.cuda.current_device()} "
          f"y.sum={y.sum().item()} mem_allocated={torch.cuda.memory_allocated()}", flush=True)


if __name__ == "__main__":
    # 只有父进程会走到这里；子进程 spawn 后 __name__ == "__mp_main__"
    torch.cuda.init()
    x = torch.zeros(1024 * 1024, device="cuda")          # 4 MB
    print(f"[父进程] pid={os.getpid()} __name__={__name__} "
          f"mem_allocated={torch.cuda.memory_allocated()} devices={torch.cuda.device_count()}",
          flush=True)

    ctx = mp.get_context("spawn")                        # 显式指定 spawn
    p = ctx.Process(target=child, args=("spawn子进程",), daemon=False)
    p.start()                                            # 创建子进程
    p.join()                                             # 父进程阻塞等待

    print(f"[父进程] 子进程 exitcode={p.exitcode}；父进程自己的 x.sum={x.sum().item()} "
          f"mem_allocated={torch.cuda.memory_allocated()}", flush=True)
    print("[父进程] 关键点：子进程的 mem_allocated 与父进程无关 —— 两个进程各有独立 CUDA context；",
          "父进程创建的 x 在子进程里根本不存在。", flush=True)
