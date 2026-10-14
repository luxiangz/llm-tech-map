"""实验 4：fork 的真实语义（Linux/WSL，不需要 GPU）。

要证明三件事：
  1) fork 出来的子进程只保留"调用 fork 的那个线程"，父进程里的其他线程在子进程中不存在；
  2) 子进程拿到的是父进程内存的一份写时复制快照 —— 父进程后续的修改，子进程看不到；
  3) CPython 自己在多线程进程里调用 fork 时会发 DeprecationWarning，
     这正是"CUDA 初始化之后再 fork 很危险"的同源问题：库内部线程不会被复制过去。
"""
import os
import sys
import threading
import time

counter = 0
side_thread_ok = False


def side_thread() -> None:
    """父进程里的一条后台线程（类比 CUDA runtime / libzmq 的内部线程）。"""
    global side_thread_ok
    time.sleep(0.5)
    side_thread_ok = True
    while True:
        time.sleep(1)


if __name__ == "__main__":
    t = threading.Thread(target=side_thread, daemon=True)
    t.start()
    time.sleep(1.0)

    counter = 1
    print(f"[父进程] pid={os.getpid()} fork 前存活线程数={threading.active_count()}", flush=True)
    pid = os.fork()                       # Linux only
    if pid == 0:
        # ---- 子进程 ----
        print(f"[fork子进程] pid={os.getpid()}")
        print(f"[fork子进程] 存活线程数={threading.active_count()} "
              f"（父进程 fork 时有 2 条：主线程 + 后台线程）")
        print(f"[fork子进程] 后台线程留下的变量 side_thread_ok={side_thread_ok} "
              f"（值被复制过来了，但线程本身没有跟着过来）")
        print(f"[fork子进程] counter 的快照 = {counter}")
        sys.stdout.flush()                # os._exit 不刷缓冲区，必须手动 flush
        os._exit(0)
    else:
        counter = 999                     # 父进程改了
        os.waitpid(pid, 0)
        print(f"[父进程] pid={os.getpid()} 存活线程数={threading.active_count()}", flush=True)
        print(f"[父进程] 父进程的 counter = {counter}（子进程看到的是快照 1，互不影响）", flush=True)
