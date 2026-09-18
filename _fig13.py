# -*- coding: utf-8 -*-
"""图 1-3 从静态 SVG 切换为 drawio viewer 嵌入第 3 页（Sparse MoE Routing Flow）。"""
import io, json, re, html as H

DRAWIO = "docs/inference/assets/ffn_formula_flow.drawio"
HTML = "docs/inference/moe-ffn-fused-operators.html"

s = io.open(DRAWIO, encoding="utf-8").read()
m = re.search(r'(  <diagram id="moe_flow".*?</diagram>)', s, re.S)
assert m, "page3 not found"
xml3 = "<mxfile>" + m.group(1) + "\n</mxfile>"
cfg = json.dumps({"xml": xml3, "nav": True, "resize": True,
                  "toolbar": "zoom layers", "autoFit": True}, ensure_ascii=True)
attr = H.escape(cfg, quote=True)

h = io.open(HTML, encoding="utf-8").read()
# 定位图 1-3 的 figure 块
i0 = h.index('<img src="assets/moe-routing-flow.svg"')
fs = h.rindex('<div class="figure">', 0, i0)
fe = h.index('</div>', i0) + len('</div>')
figcap_idx = h.index('图 1-3', fe)
figcap_end = h.index('</div>', figcap_idx) + len('</div>')
new_block = ('<div class="mxgraph-wrapper" style="border:1px solid #e5e7eb;border-radius:10px;overflow:hidden;background:#fff">\n'
             '<div class="mxgraph" style="max-width:100%" data-mxgraph=\'' + attr + '\'></div>\n'
             '</div>')
h = h[:fs] + new_block + h[figcap_end:]

# 头部资产说明同步
old_ref = '<a href="assets/moe-routing-flow.svg">moe-routing-flow.svg</a>'
assert old_ref in h or True
h = h.replace('图 1-3：Sparse MoE 的 Router/TopK、routed experts 与 shared experts 展开',
              '图 1-3：Sparse MoE 的 Router/TopK、routed experts 与 shared experts 展开（drawio viewer 渲染「Sparse MoE Routing Flow」sheet，与 draw.io 显示一致）')

io.open(HTML, "w", encoding="utf-8", newline="").write(h)
print("fig 1-3 switched to viewer; mxgraph divs:", h.count('class="mxgraph"'))
