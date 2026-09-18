# 大模型技术地图（GitHub Pages 版）

面向大模型方向的全栈知识体系清单，发布为静态站点：

**https://luxiangz.github.io/llm-tech-map/**

## 目录约定

所有已发布长文和专题统一放在 `docs/` 下，按主题分子目录：

- `docs/minimind/`：MiniMind 全链路指南
- `docs/inference/`：推理引擎与 Serving 笔记
- `docs/parallel/`：并行与通信笔记
- `docs/moe-ep/`：MoE 专家并行负载均衡专题
- `docs/index.html`：内容层统一入口

## 已发布文档

### 学习指南

- [MiniMind 全链路完整指南](https://luxiangz.github.io/llm-tech-map/docs/minimind/)

### 推理引擎与 Serving

- [Continuous Batching 完整梳理](https://luxiangz.github.io/llm-tech-map/docs/inference/continuous-batching-history-flow.html)
- [Chunked Prefill 完整梳理](https://luxiangz.github.io/llm-tech-map/docs/inference/chunked-prefill-history-flow.html)
- [PCP 完整过程梳理](https://luxiangz.github.io/llm-tech-map/docs/inference/pcp-code-walkthrough.html)
- [MLA 计算流全图解 & 吸收矩阵对比分析](https://luxiangz.github.io/llm-tech-map/docs/inference/mla-absorb-compute-flow.html)
- [前缀命中在 Transformer / MoE 中到底节省了什么计算](https://luxiangz.github.io/llm-tech-map/docs/inference/prefix-cache-kv-moe.html)
- [MoE 与 FFN 原理及 vLLM / vLLM-Ascend 前馈融合算子](https://luxiangz.github.io/llm-tech-map/docs/inference/moe-ffn-fused-operators.html)
- [投机解码完全指南：从拒绝采样到并行草稿生成](https://luxiangz.github.io/llm-tech-map/docs/inference/speculative-decoding-complete-guide.html)

### 并行与通信

- [SP 与 CP 详解](https://luxiangz.github.io/llm-tech-map/docs/parallel/sp-vs-cp.html)
- [注意力家族的张量并行（TP）拆解](https://luxiangz.github.io/llm-tech-map/docs/parallel/attention-tp-shape-forward.html)
- [大模型端到端数据流 × 内存与通信技术栈](https://luxiangz.github.io/llm-tech-map/docs/parallel/llm-dataflow-memory-communication-stack.html)

### MoE 专家并行（EP）负载均衡

- [方案深度解析](https://luxiangz.github.io/llm-tech-map/docs/moe-ep/moe-ep-overview.html)
- [源码级对比分析](https://luxiangz.github.io/llm-tech-map/docs/moe-ep/moe-ep-code-analysis.html)
- [UltraEP 深度阅读笔记](https://luxiangz.github.io/llm-tech-map/docs/moe-ep/ultraep-deep-reading.html)
- [MoE 基础、演进、问题与负载均衡前沿](https://luxiangz.github.io/llm-tech-map/docs/moe-ep/moe-load-balancing-report.html)
- [TEMPO 深度阅读笔记](https://luxiangz.github.io/llm-tech-map/docs/moe-ep/tempo-deep-reading.html)

## 仓库结构

| 路径 | 作用 |
|---|---|
| `index.md` | 站点首页：已发布文档入口 + 学习地图 |
| `docs/` | 所有已发布专题内容，按主题分子目录 |
| `assets/css/` | Jekyll 主题样式 |
| `assets/lib/` | 前端库（MathJax / highlight.js） |
| `assets/images/` | 图片资源，按专题分子目录 |
| `_config.yml` | Jekyll 配置与 GitHub Pages 插件 |

## 旧链接兼容

旧路径（`/minimind-guide.html`、`/guides/minimind-guide.html`、`/notes/*.html`、`/docs/moe-ep-*.html`）通过 `jekyll-redirect-from` 自动跳转到新路径，不需要在源码中保留旧目录。

## 更新进度

1. 编辑 `index.md`，把对应条目 `- [ ]` 改为 `- [x]` 即打勾
2. 提交并推送，约 1 分钟后网站自动更新

## 发布方式

GitHub Pages 从 `main` 分支根目录直接构建，无需本地构建。
