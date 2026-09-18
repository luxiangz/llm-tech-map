"""纯 Python 复刻 DeepSeek EPLB eplb.py，逐步打印推演过程。

对齐 raw.githubusercontent.com/deepseek-ai/EPLB/main/eplb.py（2025-09 快照）。
"""

def balanced_packing(weight, num_packs, trace=None):
    """按权重降序，逐个放入'当前最轻且未满'的包；每包恰好 n/m 个。"""
    num_layers, num_groups = len(weight), len(weight[0])
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs
    pack_index = [[-1] * num_groups for _ in range(num_layers)]
    rank_in_pack = [[-1] * num_groups for _ in range(num_layers)]
    for i in range(num_layers):
        order = sorted(range(num_groups), key=lambda g: -weight[i][g])
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for group in order:
            cand = [p for p in range(num_packs) if pack_items[p] < groups_per_pack]
            pack = min(cand, key=lambda p: pack_weights[p])
            pack_index[i][group] = pack
            rank_in_pack[i][group] = pack_items[pack]
            pack_weights[pack] += weight[i][group]
            pack_items[pack] += 1
        if trace is not None:
            trace.append((i, 'packing', order, pack_weights[:]))
    return pack_index, rank_in_pack


def replicate_experts(weight, num_phy, trace=None):
    """每轮选 weight/logcnt 最大的逻辑专家加一个副本。"""
    n, num_log = len(weight), len(weight[0])
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    phy2log = [list(range(num_phy)) for _ in range(n)]
    rank = [[0] * num_phy for _ in range(n)]
    logcnt = [[1] * num_log for _ in range(n)]
    for i in range(num_log, num_phy):
        red = [max(range(num_log), key=lambda g: weight[l][g] / logcnt[l][g])
               for l in range(n)]
        for l in range(n):
            phy2log[l][i] = red[l]
            rank[l][i] = logcnt[l][red[l]]
            logcnt[l][red[l]] += 1
        if trace is not None:
            eff = [[round(weight[l][g] / logcnt[l][g], 2) for g in range(num_log)]
                   for l in range(n)]
            trace.append((i, 'replicate', red, eff))
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(weight, num_phy, num_groups, num_nodes, num_gpus, tr):
    L, num_log = len(weight), len(weight[0])
    group_size = num_log // num_groups
    groups_per_node = num_groups // num_nodes
    phy_per_gpu = num_phy // num_gpus

    # Step 1: 组 -> 节点
    tpg = [[sum(weight[l][g * group_size:(g + 1) * group_size]) for g in range(num_groups)]
           for l in range(L)]
    tr.append(('step1', 'tokens_per_group', tpg))
    gpi, gri = balanced_packing(tpg, num_nodes, tr)
    # log2mlog：节点主序的局部逻辑编号
    log2mlog = [[0] * num_log for _ in range(L)]
    for l in range(L):
        for g in range(num_groups):
            node = gpi[l][g]
            r = gri[l][g]
            base = (node * groups_per_node + r) * group_size
            for k in range(group_size):
                log2mlog[l][g * group_size + k] = base + k
    mlog2log = [[0] * num_log for _ in range(L)]
    for l in range(L):
        for idx, m in enumerate(log2mlog[l]):
            mlog2log[l][m] = idx
    tr.append(('step1', 'log2mlog', log2mlog))

    # Step 2: 节点内复制（每节点 6 逻辑 -> 8 物理）
    mlog_per_node = num_log // num_nodes
    # tpm：[L*num_nodes, mlog_per_node]，每层每节点一个视图，按节点主序重排后的权重
    tpm = []
    for l in range(L):
        for n_ in range(num_nodes):
            seg = [weight[l][mlog2log[l][n_ * mlog_per_node + j]] for j in range(mlog_per_node)]
            tpm.append(seg)
    tr.append(('step2', 'tokens_per_mlog(per-node view)', tpm))
    phy2mlog, phyrank, mlogcnt = [], [], []
    for row in tpm:
        p, r, c = replicate_experts([row], num_phy // num_nodes)
        phy2mlog.append(p[0]); phyrank.append(r[0]); mlogcnt.append(c[0])
    tr.append(('step2', 'phy2mlog', phy2mlog))
    tr.append(('step2', 'mlogcnt', mlogcnt))

    # Step 3: 物理 -> GPU（每节点 8 物理 -> 4 卡，每卡 2 个）
    phy_per_node = num_phy // num_nodes
    tpp = []
    for row_i, row in enumerate(tpm):
        tpp.append([row[phy2mlog[row_i][p]] / mlogcnt[row_i][phy2mlog[row_i][p]]
                    for p in range(phy_per_node)])
    tr.append(('step3', 'tokens_per_phy(load/cnt)', [[round(x, 2) for x in r] for r in tpp]))
    pi, ri = [], []
    for row_i in range(len(tpp)):
        p, r = balanced_packing([tpp[row_i]], num_gpus // num_nodes)
        pi.append(p[0]); ri.append(r[0])
    phy2pphy = [[pi[row_i][p] * phy_per_gpu + ri[row_i][p] for p in range(phy_per_node)]
                for row_i in range(len(tpp))]
    tr.append(('step3', 'phy2pphy(节点内物理->GPU槽位)', phy2pphy))

    # 组装回全局 [L, num_phy]
    pphy2phy = [[0] * phy_per_node for _ in range(len(phy2pphy))]
    for row_i in range(len(phy2pphy)):
        for p, pp in enumerate(phy2pphy[row_i]):
            pphy2phy[row_i][pp] = p
    out = [[0] * num_phy for _ in range(L)]
    for l in range(L):
        for n_ in range(num_nodes):
            row_i = l * num_nodes + n_
            for slot in range(phy_per_node):
                mlog = phy2mlog[row_i][pphy2phy[row_i][slot]]
                glob_mlog = n_ * mlog_per_node + mlog
                out[l][n_ * phy_per_node + slot] = mlog2log[l][glob_mlog]
    return out


weight = [[90, 132, 40, 61, 104, 165, 39, 4, 73, 56, 183, 86],
          [20, 107, 104, 64, 19, 197, 187, 157, 172, 86, 16, 27]]
tr = []
result = rebalance_experts_hierarchical(weight, 16, 4, 2, 8, tr)
expect = [[5, 6, 5, 7, 8, 4, 3, 4, 10, 9, 10, 2, 0, 1, 11, 1],
          [7, 10, 6, 8, 6, 11, 8, 9, 2, 4, 5, 1, 5, 0, 3, 1]]

for t in tr:
    print(t)
print()
print('phy2log =', result)
print('expect =', expect)
print('MATCH' if result == expect else 'MISMATCH')
