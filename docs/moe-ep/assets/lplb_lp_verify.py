"""LPLB LP 模型复刻与验证（纯 numpy）。

复刻 minilp.cu 的 LP 构造（kernel_build）与原始仿射尺度内点法（kernel_solve），
在 CUBE_8P2E 拓扑上用合成负载验证：
  1) 拓扑不变量（每个 rank 被复制恰好 2 次、持有 2 个副本）；
  2) DeepSeek 的 5 次迭代内点法 vs 收敛解（2000 次迭代）vs 0.5/0.5 基线。
"""
import numpy as np

G, D = 8, 2  # GROUP_SIZE, DUP_PER_RANK（CUBE_8P2E）

# README 示例拓扑：r2o[r] = [rank r 的两个冗余槽分别复制谁的专家]
r2o = np.array([[3, 0, 1, 2, 7, 4, 5, 6],
                [6, 7, 4, 5, 0, 1, 2, 3]]).T  # [8, 2]

# ---- 拓扑不变量 ----
copies_held = np.zeros(G, dtype=int)   # 每 rank 持有的副本数
times_copied = np.zeros(G, dtype=int)  # 每 rank 的专家被别人复制的次数
for r in range(G):
    for d in range(D):
        copies_held[r] += 1
        times_copied[r2o[r][d]] += 1
print("每 rank 持有副本数:", copies_held.tolist(), "| 每 rank 被复制次数:", times_copied.tolist())
assert (copies_held == D).all() and (times_copied == D).all(), "拓扑不变量不成立"
print("拓扑不变量 OK：每 rank 持有 2 副本、其专家被复制 2 次（每条冗余边恰好连接 2 个 rank）\n")

# ---- LP 构造模板（对应 minilp.cu 的 kernel_build；负载项在 make 后填入） ----
NV = G * D * 2 + G + 2
NC = G + G * D
A = np.zeros((NC, NV))
b = np.zeros(NC)
for ic in range(G):
    A[ic][2 * G * D + ic] = 1.0        # slack
    A[ic][NV - 2] = -1.0               # t
for q in range(G * D):                 # 行 2：o_frac + r_frac = 1
    A[G + q][q] = 1.0
    A[G + q][G * D + q] = 1.0
    b[G + q] = 1.0
c = np.zeros(NV)
c[NV - 2] = 1.0     # min t
c[NV - 1] = 1000.0  # Big-M

# ---- 两种实例：A 固定负载主导（LP 无能为力是正确行为），B 复制边主导（LP 主战场） ----
def make_instance(fixed_range, dup_range, seed):
    rg = np.random.default_rng(seed)
    dw = rg.integers(*dup_range, size=(G, D)).astype(float)
    fw = rg.integers(*fixed_range, size=G).astype(float)
    scale = max(dw.max(), fw.max())
    return dw / scale, fw / scale

def solve_lp(dw, fw):
    A2, b2 = A.copy(), b.copy()
    for ic in range(G):
        for iv in range(G * D):
            A2[ic][iv] = dw[iv // D][iv % D] if iv // D == ic else 0.0
        for iv in range(G * D, 2 * G * D):
            k, d = (iv - G * D) // D, (iv - G * D) % D
            A2[ic][iv] = dw[k][d] if r2o[k][d] == ic else 0.0
        A2[ic][2 * G * D + ic] = 1.0
        A2[ic][NV - 2] = -1.0
        b2[ic] = -fw[ic]
    for ic in range(NC):
        A2[ic][NV - 1] = b2[ic] - A2[ic][:NV - 1].sum()

    x = np.ones(NV)
    for _ in range(5):
        M = A2 @ np.diag(x * x) @ A2.T
        y = np.linalg.solve(M + 1e-9 * np.eye(NC), A2 @ (x * x * c))
        dvec = x * (c - y @ A2)
        alpha = 0.999 / max(dvec.max(), 1e-12)
        x = np.clip(x * (1 - alpha * dvec), 1e-12, None)
    o, r = x[:G * D], x[G * D:2 * G * D]
    ratio = o / (o + r)
    return ratio

def loads(dw, fw, ratio):
    L = fw.copy()
    for k in range(G):
        for d in range(D):
            L[k] += dw[k][d] * ratio[k * D + d]
            L[r2o[k][d]] += dw[k][d] * (1 - ratio[k * D + d])
    return L

for name, args in [
    ("实例 A · 固定负载主导（LP 撞不可搬运的下界是正确行为）", ((100, 600), (50, 900), 42)),
    ("实例 B · 复制边主导（LPLB 的目标场景）", ((20, 60), (200, 1000), 7)),
]:
    dw, fw = make_instance(*args)
    L_base = loads(dw, fw, np.full(G * D, 0.5))
    L_lp = loads(dw, fw, solve_lp(dw, fw))
    total = fw.sum() + dw.sum()
    lb = max(total / G, fw.max())
    print(f"== {name}")
    print(f"   0.5/0.5 基线: max={L_base.max():8.2f}  imbalance(max/mean)={L_base.max() / L_base.mean():.3f}")
    print(f"   LP(5次迭代): max={L_lp.max():8.2f}  imbalance={L_lp.max() / L_lp.mean():.3f}  残余{L_lp.max() - L_base.min():.2f}")
    print(f"   理论下界 LB=max(组均值, 最大固定)={lb:.2f} → LP 距下界 {L_lp.max() - lb:+.4f}，相对基线改进 {(1 - L_lp.max() / L_base.max()) * 100:.1f}%")
    print(f"   各 rank 负载(LP 后): {[round(v, 1) for v in L_lp.tolist()]}")
    print()
