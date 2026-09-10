# 大模型技术地图（GitHub Pages 版）

面向大模型方向的全栈知识体系清单，发布为静态站点：

**https://luxiangz.github.io/llm-tech-map/**

## 已发布文档

站点内可直接访问的文档页（HTML）：

### 指南

- [MiniMind 全链路完整指南：从零训练到强化学习与部署](https://luxiangz.github.io/llm-tech-map/guides/minimind-guide.html)

### 推理引擎与 Serving（vLLM）

- [Continuous Batching 完整梳理——机制、意义与前世今生](https://luxiangz.github.io/llm-tech-map/notes/continuous-batching-history-flow.html)
- [Chunked Prefill 完整梳理——流程、意义与前世今生](https://luxiangz.github.io/llm-tech-map/notes/chunked-prefill-history-flow.html)
- [PCP 完整过程梳理 — 从拉起服务到长序列请求流转](https://luxiangz.github.io/llm-tech-map/notes/pcp-code-walkthrough.html)
- [MLA 计算流全图解 & 吸收矩阵对比分析](https://luxiangz.github.io/llm-tech-map/notes/mla-absorb-compute-flow.html)
- [前缀命中在 Transformer / MoE 中到底节省了什么计算](https://luxiangz.github.io/llm-tech-map/notes/prefix-cache-kv-moe.html)

### 并行与通信

- [SP 与 CP 详解 — 序列并行 vs 上下文并行的来龙去脉与完整对比](https://luxiangz.github.io/llm-tech-map/notes/sp-vs-cp.html)
- [注意力家族的张量并行（TP）拆解 — MHA · GQA · MLA · DSA 切分策略 × 完整前向 shape 走查](https://luxiangz.github.io/llm-tech-map/notes/attention-tp-shape-forward.html)
- [大模型端到端数据流 × 内存与通信技术栈](https://luxiangz.github.io/llm-tech-map/notes/llm-dataflow-memory-communication-stack.html)

### MoE 专家并行（EP）负载均衡研究

- [落地索引页](https://luxiangz.github.io/llm-tech-map/docs/)
- [EPLB · UltraEP · MoonEP 方案深度解析（v2.0 联网一手复核）](https://luxiangz.github.io/llm-tech-map/docs/moe-ep-overview.html)
- [源码级对比分析（附本地代码与研读路线图）](https://luxiangz.github.io/llm-tech-map/docs/moe-ep-code-analysis.html)

## 更新进度

1. 编辑 `index.md`，把对应条目 `- [ ]` 改为 `- [x]` 即打勾
2. 提交并推送，约 1 分钟后网站自动更新

## 仓库结构

| 文件 | 作用 |
|---|---|
| `index.md` | 统一入口：顶部为已发布文档索引，下方为完整学习地图 |
| `guides/` | 独立长文指南（MiniMind 等） |
| `notes/` | 知识点的详细笔记页（与地图条目互相链接） |
| `docs/` | 专题研究笔记（MoE 专家并行负载均衡） |
| `assets/css/` | Jekyll 主题样式与字体排版覆盖 |
| `assets/lib/` | 前端库（MathJax / highlight.js） |
| `assets/images/` | 全站图片，按指南与笔记分目录存放 |
| `minimind-guide.html` | 兼容旧链接的跳转页，指向 `guides/minimind-guide.html` |
| `_config.yml` | Jekyll 配置：GFM 引擎 + cayman 主题 |

## 发布方式

GitHub Pages 从 `main` 分支根目录直接构建（Settings → Pages → Deploy from a branch → main / root）。无需本地构建。
