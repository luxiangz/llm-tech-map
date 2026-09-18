# MoE EP 负载均衡研究笔记（Git Pages 静态发布）

本目录由本地工作区 E:\Desktop\业务\EP 发布（2026-08 快照，2026-09 增补），四份主文件：

| 文件 | 对应本地中文名 | 说明 |
|---|---|---|
| moe-ep-overview.html | MoE专家并行方案深度解析_EPLB_UltraEP_MoonEP.html | 方案级 v2.0（联网一手复核） |
| moe-ep-code-analysis.html | 源码级对比分析_EPLB_UltraEP_MoonEP.html | 源码级对比（附本地代码与研读路线图） |
| ultraep-deep-reading.html | — | UltraEP 深度阅读笔记（2026-09-16 全文重写：Part I 源码精读 + Part II 十六问详解） |
| tempo-deep-reading.html | — | TEMPO 深度阅读笔记（相图、Makespan 调度与「代理失配」） |
| eplb-deep-reading.html | — | EPLB 深度阅读笔记（2026-09-18：原理 / 三步算法逐位推演 / vLLM·SGLang 集成 / 实测数据；附 drawio 三图与纯 Python 复刻脚本 assets/eplb_trace.py） |
| moe-load-balancing-report.html | moe_load_balancing_report.html | MoE 基础、演进、问题与负载均衡前沿（2026-09-18：重写 2.3 经典负载均衡损失六步拆解、2.5 评估指标口径与算例） |

研究范围：DeepSeek EPLB、UltraEP（arXiv:2606.04101，小红书×北大）、MoonEP（Moonshot AI）。
证据标注：V=一手原文核对 / R=合理推断 / U=待核实（见 moe-ep-overview.html §1）。
