# -*- coding: utf-8 -*-
"""把 3 个研究笔记 HTML 整理进 gitpage 站点的 notes/ 目录"""
import io, os, shutil

SRC = r"E:\Desktop\研究\2_03_并行研究"
DST = r"E:\Desktop\gitpage"
NOTES = os.path.join(DST, "notes")
ASSETS = os.path.join(NOTES, "assets")
os.makedirs(ASSETS, exist_ok=True)

BACKLINK = ('<a href="../index.html" style="position:fixed;top:12px;right:14px;z-index:99;'
            'background:#f6f7f9;border:1px solid #e2e5ea;border-radius:999px;padding:4px 12px;'
            'font-size:14px;color:#2c5fa8;text-decoration:none;'
            'box-shadow:0 1px 4px rgba(0,0,0,.08)">← 技术地图首页</a>')

# (源文件, 目标文件名)
files = [
    ("attention_tp_shape_forward.html", "attention-tp-shape-forward.html"),
    ("pcp_code_walkthrough.html", "pcp-code-walkthrough.html"),
    ("sp_vs_cp_sequence_context_parallel.html", "sp-vs-cp.html"),
]

# 图片复制：(源路径, 目标路径)
images = [
    (os.path.join(SRC, "DeepSeek-V3.2 的注意力架构MHA.png"), os.path.join(ASSETS, "dsv32-attention-mha.png")),
    (os.path.join(SRC, "DeepSeek-V3.2 的注意力架构.png"), os.path.join(ASSETS, "dsv32-attention-mla.png")),
    (os.path.join(SRC, "DeepSeek-V3.2 的注意力架构MQA.png"), os.path.join(ASSETS, "dsv32-attention-mqa.png")),
    (r"E:\Desktop\deepseek_v3_architecture.jpg", os.path.join(ASSETS, "deepseek-v3-architecture.jpg")),
]

for s, d in images:
    shutil.copy2(s, d)
    print("img:", os.path.basename(d), os.path.getsize(d), "bytes")

def apply(name, pairs):
    p = os.path.join(NOTES, name)
    s = io.open(p, encoding="utf-8").read()
    for old, new in pairs:
        if old not in s:
            print("MISSING in", name, ":", old[:60])
            raise SystemExit(1)
        s = s.replace(old, new)
    # 返回链接插到 <body> 后
    assert s.count("<body>") == 1, name
    s = s.replace("<body>", "<body>\n" + BACKLINK, 1)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s)
    print("ok:", name, "(", len(pairs), "replacements )")

# --- attention-tp-shape-forward.html ---
apply("attention-tp-shape-forward.html", [
    # 图片路径：file:// 与 URL 编码中文名 → notes/assets 英文名
    ("file:///E:/Desktop/deepseek_v3_architecture.jpg", "assets/deepseek-v3-architecture.jpg"),
    ("DeepSeek-V3.2%20%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9E%B6%E6%9E%84MHA.png", "assets/dsv32-attention-mha.png"),
    ("DeepSeek-V3.2%20%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9E%B6%E6%9E%84MQA.png", "assets/dsv32-attention-mqa.png"),
    ("DeepSeek-V3.2%20%E7%9A%84%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9E%B6%E6%9E%84.png", "assets/dsv32-attention-mla.png"),
    # 头部衔接（未上线的本地笔记 → 纯文本）
    ('<a href="parallelism_degrees_research.html">02_并行研究·各种并行度研究</a>', '<span class="nolink">02_并行研究·各种并行度研究</span>'),
    ('<a href="../01_vllm/execution_pipeline_dsv32_dsv4_glm5_kimiv3.html">01_vllm 执行流水报告</a>', '<span class="nolink">01_vllm 执行流水报告</span>'),
    # 页脚配套
    ('<a href="parallelism_degrees_research.html">02_并行研究·各种并行度研究</a>（TP/PP/DP/EP/SP/CP 全景）',
     '<span class="nolink">02_并行研究·各种并行度研究</span>（TP/PP/DP/EP/SP/CP 全景）'),
    ('<a href="../01_vllm/execution_pipeline_dsv32_dsv4_glm5_kimiv3.html">01_vllm 执行流水拆解报告</a>（MTP/DSA 算子级细节）。',
     '<span class="nolink">01_vllm 执行流水拆解报告</span>（MTP/DSA 算子级细节）；'
     '<a href="sp-vs-cp.html">SP vs CP 详解</a>（SP/CP 两线演化的完整对比）。'),
])

# --- sp-vs-cp.html ---
apply("sp-vs-cp.html", [
    # 站内互链：attention 页面新文件名
    ('<a href="attention_tp_shape_forward.html">注意力家族 TP 拆解</a>', '<a href="attention-tp-shape-forward.html">注意力家族 TP 拆解</a>'),
    ('<a href="attention_tp_shape_forward.html">注意力 TP 报告</a>', '<a href="attention-tp-shape-forward.html">注意力 TP 报告</a>'),
    # 头部衔接
    ('<a href="parallelism_degrees_research.html">各种并行度研究 §6</a>', '<span class="nolink">各种并行度研究 §6</span>'),
    # 正文中的并行度报告引用 → 纯文本
    ('<a href="parallelism_degrees_research.html">并行度报告 §6.3</a>', '<span class="nolink">并行度报告 §6.3</span>'),
    ('<a href="parallelism_degrees_research.html">并行度报告 §6.2</a>', '<span class="nolink">并行度报告 §6.2</span>'),
    # 页脚配套：加 PCP 页互链
    ('<a href="attention_tp_shape_forward.html">注意力家族 TP 拆解</a>（MLA 的 KV 为何切不动、TP 的 KV 复制与 DCP 解药）。',
     '<a href="attention-tp-shape-forward.html">注意力家族 TP 拆解</a>（MLA 的 KV 为何切不动、TP 的 KV 复制与 DCP 解药）；'
     '<a href="pcp-code-walkthrough.html">PCP 完整过程梳理</a>（vLLM 里 PCP 的落地走查）。'),
])

# --- pcp-code-walkthrough.html（无外链，只加返回链接）---
apply("pcp-code-walkthrough.html", [])

print("ALL DONE")
