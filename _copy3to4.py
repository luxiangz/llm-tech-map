# -*- coding: utf-8 -*-
"""第 4 页 = 第 3 页（Sparse MoE Routing Flow）的深拷贝 + 最小插入。"""
import io, re
import xml.etree.ElementTree as ET

P = "docs/inference/assets/ffn_formula_flow.drawio"
s = io.open(P, encoding="utf-8").read()

# ---- 定位第 3 页 ----
m3 = re.search(r'  <diagram id="([^"]+)" name="Sparse MoE Routing Flow">', s)
assert m3
start = m3.start()
end = s.index("</diagram>", start) + len("</diagram>")
page3 = s[start:end]

page4 = page3
# 1) 改 id/name
page4 = page4.replace(m3.group(1), "ep_dispatch_combine_flow", 1)
page4 = page4.replace('name="Sparse MoE Routing Flow"', 'name="EP Dispatch/Combine Flow"', 1)
# 2) 标题
old_t = "Sparse MoE：Router → TopK → 激活少数专家 → 加权求和"
new_t = "Sparse MoE + EP：同一张图上标注 dispatch / combine（虚线框 = rank 泳道，红条 = 通信阶段）"
assert old_t in page4
page4 = page4.replace(old_t, new_t, 1)
# 3) shared 边标签补注
old_e = 'value="所有 token"'
assert old_e in page4
page4 = page4.replace(old_e, 'value="所有 token（本地计算，不走 dispatch）"', 1)

# 4) 插入 cells（放在 root 最前 = z 序最底层，原边从横幅上穿过）
insert = '''        <mxCell id="ep_g0" value="GPU0 · token 所在 rank" style="rounded=1;dashed=1;html=1;fillColor=none;strokeColor=#7ea6c9;fontSize=14;fontStyle=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=4;" vertex="1" parent="1">
          <mxGeometry x="60" y="70" width="1180" height="510" as="geometry" />
        </mxCell>
        <mxCell id="ep_g1" value="GPU1 · 专家所在 rank（专家权重不动，动的是 token）" style="rounded=1;dashed=1;html=1;fillColor=none;strokeColor=#7ea6c9;fontSize=14;fontStyle=1;verticalAlign=top;align=left;spacingLeft=10;spacingTop=4;" vertex="1" parent="1">
          <mxGeometry x="60" y="655" width="1180" height="100" as="geometry" />
        </mxCell>
        <mxCell id="ep_dispatch" value="DISPATCH all-to-all：token 发往各专家所在 rank（计数向量先行交换，设备侧完成）" style="rounded=1;html=1;fillColor=#fdeceb;strokeColor=#b85450;fontSize=13;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="250" y="595" width="800" height="42" as="geometry" />
        </mxCell>
        <mxCell id="ep_combine" value="COMBINE all-to-all：专家输出送回源 rank，按 gᵢ 加权求和" style="rounded=1;html=1;fillColor=#fdeceb;strokeColor=#b85450;fontSize=13;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="250" y="765" width="800" height="42" as="geometry" />
        </mxCell>
        <mxCell id="ep_legend2" value="蓝色虚线框 = rank 泳道；红条 = dispatch/combine 通信阶段（TopK→专家 与 专家→Sum 的六条边分别构成这两次 all-to-all）；专家权重全程不移动" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;fontSize=13;" vertex="1" parent="1">
          <mxGeometry x="240" y="975" width="800" height="30" as="geometry" />
        </mxCell>
'''
anchor = '        <mxCell id="0" />\n        <mxCell id="1" parent="0" />\n'
assert anchor in page4
page4 = page4.replace(anchor, anchor + insert, 1)

# ---- 替换现有第 4 页 ----
m4 = re.search(r'  <diagram id="ep_dispatch_combine_flow"[^>]*>', s)
if m4:
    s4end = s.index("</diagram>", m4.start()) + len("</diagram>")
    s = s[:m4.start()] + page4 + s[s4end:]
else:
    s = s.replace("</mxfile>", page4 + "\n</mxfile>")

io.open(P, "w", encoding="utf-8", newline="").write(s)

t = ET.parse(P)
diags = t.getroot().findall("diagram")
assert len(diags) == 4, len(diags)
for dg in diags:
    for c in dg.findall(".//mxCell"):
        assert c.get("parent") != c.get("id")
d4 = diags[3]
ids4 = [c.get("id") for c in d4.findall(".//mxCell")]
n_orig = len([i for i in ids4 if not i.startswith("ep_") or i == "ep_dispatch_combine_flow"])
print("page4 cells:", len(ids4), "| inserted ep_*:", len([i for i in ids4 if i.startswith("ep_") and i != "ep_dispatch_combine_flow"]))
# 第3、4页节点集合对比（除 ep_ 插入外应一致）
ids3 = set(c.get("id") for c in diags[2].findall(".//mxCell"))
diff = ids4_set = set(ids4) - ids3 - {"ep_g0","ep_g1","ep_dispatch","ep_combine","ep_legend2"}
print("unexpected diff vs page3:", diff if diff else "none (exact copy)")
