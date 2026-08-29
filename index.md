---
title: 大模型技术地图（全栈知识体系）
description: 面向大模型方向的完整技术知识目录，覆盖从数学基础到工程实践的整条链路
---
# 大模型技术地图（全栈知识体系）

> 面向大模型方向的完整技术知识目录，覆盖从数学基础到工程实践的整条链路。
> **进度追踪**：在源码中把 `- [ ]` 改为 `- [x]` 即为已完成（GitHub Pages 渲染为复选框）。
> 建议学习顺序：领域一 → 领域五/三（核心双主线）→ 领域四 → 领域六 → 领域七 → 领域二 → 领域八 → 领域九 → 领域十（贯穿全程）。

## 目录
1. [深度学习与机器学习基础](#一深度学习与机器学习基础)
2. [Transformer 与模型架构](#二transformer-与模型架构)
3. [预训练](#三预训练)
4. [后训练：对齐与微调](#四后训练对齐与微调)
5. [推理引擎与 Serving](#五推理引擎与-serving)
6. [算子与高性能计算](#六算子与高性能计算)
7. [AI 硬件与集群基础设施](#七ai-硬件与集群基础设施)
8. [Agent 与应用层](#八agent-与应用层)
9. [模型生态与前沿](#九模型生态与前沿)
10. [工程实践与工具链](#十工程实践与工具链)
11. [专业术语表](#十一专业术语表)
12. [个人项目](#十二个人项目)

---

## 一、深度学习与机器学习基础
### 1.1 数学与理论基础
- [ ] 线性代数：矩阵乘法、特征分解、SVD、范数
- [ ] 概率与信息论：分布、熵/交叉熵/KL 散度、最大似然估计
- [ ] 数值优化：SGD/Momentum/Adam/AdamW/LAMB/Sophia、学习率调度（warmup/cosine/RSQRt）
- [ ] 机器学习基础：过拟合与正则化（L1/L2/Dropout）、偏差方差、交叉验证、评估指标（P/R/F1/ROC）

### 1.2 深度学习基础
- [ ] 神经网络与反向传播、自动微分原理
- [ ] CNN 基础与经典网络
- [ ] 序列建模：RNN/LSTM/GRU、seq2seq、双向模型
- [ ] 归一化：BN/LN/RMSNorm/DeepNorm、PreNorm vs PostNorm
- [ ] 激活函数演进：ReLU/GELU/SiLU/SwiGLU
- [ ] 初始化：Xavier/Kaiming/大模型缩放初始化
- [ ] 训练稳定性：loss spike、梯度裁剪、NaN 排查

### 1.3 表示学习与 Tokenizer
- [ ] Tokenizer：BPE/WordPiece/SentencePiece/BBPE、训练与退化问题
- [ ] 词向量：Word2Vec/GloVe、上下文无关 vs 上下文相关
- [ ] 嵌入与权重绑定（tied embedding）

---

## 二、Transformer 与模型架构
### 2.1 Attention 全谱系
- [ ] Attention 动机：序列建模困境、RNN 串行计算与信息瓶颈
- [ ] 核心机制：QKV、缩放点积注意力、为什么除以 √dk、Softmax 与掩码
- [ ] 变体：MHA / MQA / GQA / MLA（多潜在注意力） — 📄 [MLA 计算流图解笔记](notes/mla-absorb-compute-flow.html)
- [ ] 稀疏注意力：窗口/全局/BigBird/NSA/DSA（DeepSeek 稀疏注意力）
- [ ] 高效实现：FlashAttention 1/2/3、PagedAttention、线性注意力（Performer）
- [ ] KV 压缩：H2O/SnapKV/StreamingLLM
- [ ] 工程实现对比：FlashAttention vs PagedAttention vs MLA kernel

### 2.2 位置编码
- [ ] 绝对/相对位置编码、ALiBi、NoPE
- [ ] RoPE 旋转位置编码原理与实现
- [ ] 外推方法：NTK-aware、YaRN、ABF

### 2.3 模型结构组件
- [ ] Transformer 全景：Encoder/Decoder/Encoder-Decoder 三种范式
- [ ] FFN 变体：SwiGLU/GELU/门控机制
- [ ] 主流开源架构差异：LLaMA/Qwen/GLM/Kimi/Mistral 组件选择对比
- [ ] DeepSeek 家族：V3/V3.2/V4 架构演进、V4 PRO 模型代码精读

### 2.4 MoE 专家混合
- [ ] MoE 历史：GShard/Switch/GLaM/ST-MoE/DeepSeekMoE/MoE-LLM
- [ ] 关键设计：细粒度专家、共享专家、负载均衡损失、路由策略
- [ ] 专家并行训练与推理

### 2.5 新架构探索
- [ ] 状态空间模型：RWKV/SSM/Mamba/Mamba-2
- [ ] 线性注意力与 RetNet
- [ ] 混合架构：Jamba、DeepSeek-V3.2 DSA 混合、MoR
- [ ] Test-Time Training（TTT）层、扩散语言模型（dLLM）

### 2.6 生成模型架构
- [ ] VAE：编码器解码器、KL 散度、隐空间、ConvVAE
- [ ] 扩散模型理论：DDPM/DDIM、流匹配、CFG 引导
- [ ] DiT（Diffusion Transformer）架构
- [ ] 自回归图像生成、VQ-VAE/VQGAN 离散 tokenizer
- [ ] 轻量网络：MobileNet/SqueezeNet/ShuffleNet/EfficientNet/GhostNet/MobileViT 等

---

## 三、预训练
### 3.1 数据工程
- [ ] 数据收集：Common Crawl/多语言/代码/数学数据源
- [ ] 清洗与去重（MinHash）、质量过滤、毒性过滤
- [ ] 数据配比策略、课程学习
- [ ] 合成数据：生成-筛选循环
- [ ] 数据清洗框架与管线：DataJuicer、RefinedWeb 处理链
- [ ] 数据规模与模型规模匹配

### 3.2 Scaling Laws
- [ ] 基础三定律：参数/数据/计算量扩展
- [ ] Chinchilla 最优计算、数据效率曲线
- [ ] MuP（最大更新参数化）

### 3.3 并行训练
- [ ] 并行度全景：DP/TP/PP/EP/CP/SP 对比与选择
- [ ] 数据并行：DDP、ZeRO 三阶段、offload
- [ ] 张量并行：Megatron 算子切分 — 📄 [注意力家族 TP 拆解笔记](notes/attention-tp-shape-forward.html)
- [ ] 流水线并行：PP 气泡、调度算法、DualPipe/PgPipe
- [ ] 专家并行 EP、上下文并行 CP、序列并行 SP — 📄 [SP vs CP 详解笔记](notes/sp-vs-cp.html)
- [ ] PCP：Prefill Context Parallel 设计与实现 — 📄 [PCP 完整过程梳理笔记](notes/pcp-code-walkthrough.html)
- [ ] 集合通信：原语、AllReduce 算法、通信计算重叠

### 3.4 训练加速
- [ ] FlashAttention、MLA、Ulysses、RingAttention
- [ ] MTP（多 Token 预测）、激活重计算（Checkpoint）
- [ ] 混合精度：FP16/BF16/FP8、梯度累积、梯度桶化
- [ ] torch.compile/inductor、AOTAutograd
- [ ] Domino、FLUX 等前沿加速方法

### 3.5 训练稳定性与监控
- [ ] 超参选择：学习率/批大小/warmup
- [ ] loss spike 与梯度问题排查
- [ ] 实验追踪：W&B/MLflow、配置管理（hydra）
- [ ] 断点续训、checkpoint 格式、训练故障恢复

### 3.6 训练框架与平台
- [ ] Megatron-LM、DeepSpeed、Torchtitan、Nemotron
- [ ] 训练框架源码：deepseek-harness、minimind 从零训练
- [ ] Slurm/HPC 调度器、Kubernetes 训练调度
- [ ] 训练集群观测：DCGM/GPU 监控、任务队列管理

---

## 四、后训练：对齐与微调
### 4.1 SFT 监督微调
- [ ] 指令数据构建方法论：多样性/质量/格式、ChatML
- [ ] 多轮对话数据、CPT 继续预训练
- [ ] SFT 过拟合与灾难性遗忘、混合比例

### 4.2 参数高效微调 PEFT
- [ ] LoRA 原理与秩选择
- [ ] QLoRA（4bit NF4）、DoRA、PiSSA
- [ ] P-Tuning/Prefix-Tuning、Adapter
- [ ] LoRA 与全参微调对比实验设计

### 4.3 强化学习与偏好对齐
- [ ] RLHF 流程：奖励模型训练、KL 正则、PPO 细节（GAE/重要性采样/clip）
- [ ] DPO 及变体：IPO/KTO/ORPO/cDPO、在线 vs 离线 RL
- [ ] GRPO、RLVR 可验证奖励、R1 式强化学习
- [ ] 测试时扩展：MCTS、RFT/STaR、搜索与采样结合
- [ ] 多阶段训练流水线：SFT→RM→RL 整体设计
- [ ] RLAIF / Constitutional AI（AI 反馈对齐）
- [ ] RL 训练框架：veRL（字节火山引擎）、OpenRLHF、TRL、NeMo-Aligner、OpenRL；veRL 架构（vLLM 采样 + Megatron 训练双引擎、rollout 并行、PPO/GRPO 流水线）

### 4.4 蒸馏
- [ ] 知识蒸馏基础：logit/特征/分布蒸馏
- [ ] MiniLLM、在线蒸馏、自蒸馏
- [ ] 蒸馏与量化/剪枝的工程组合

### 4.5 模型评估
- [ ] 基准全景：MMLU/GSM8K/BBH/HumanEval/MMLU-Pro
- [ ] 评测方法论：few-shot/CoT/投票
- [ ] 评测维度分类：知识/推理/代码/数学/多语言/长上下文/指令遵循
- [ ] 评测数据建设与污染防护、LLM-as-judge、人工评测与 Arena 模式
- [ ] 长上下文评测：Needle-in-Haystack
- [ ] 评测框架（Harness）：lm-evaluation-harness、HELM、OpenCompass
- [ ] 公开排行榜与竞技场：LMSYS Chatbot Arena

---

## 五、推理引擎与 Serving
### 5.1 推理机制
- [ ] 自回归解码与 KV Cache
- [ ] PagedAttention、连续批处理（Continuous Batching） — 📄 [Continuous Batching 完整梳理笔记](notes/continuous-batching-history-flow.html)
- [ ] 分块预填充（Chunked Prefill）、PD 分离 — 📄 [Chunked Prefill 完整梳理笔记](notes/chunked-prefill-history-flow.html)
- [ ] 前缀缓存（prefix cache）、RadixAttention
- [ ] 采样家族：top-k/top-p/temperature/min-p/typical、平行采样
- [ ] 投机解码：草稿模型/自投机/Medusa/EAGLE、接受率与加速比建模
- [ ] 结构化输出与约束解码（grammar/JSON Schema/guided decoding）

### 5.2 推理引擎
- [ ] vLLM 源码解析（调度/执行/注意力/显存管理）
- [ ] SGLang、RadixAttention 与结构化生成
- [ ] TensorRT-LLM（NVIDIA 优化栈）
- [ ] TGI、llama.cpp、MLC-LLM、Ollama、Triton Inference Server
- [ ] 昇腾引擎：MindIE、vLLM-Ascend、nano-vllm
- [ ] 引擎横向对比：功能/性能/生态选型、主流引擎执行流水线对比
- [ ] 端侧与边缘推理：手机 NPU（Qualcomm/Apple/麒麟）、端侧量化部署、WebGPU/WASM、llama.cpp 端侧生态、端云协同

### 5.2b API 与协议层
- [ ] OpenAI 兼容 API 规范、流式输出（SSE）、function calling 传输层
- [ ] gRPC/HTTP 性能对比、长连接与连接池
- [ ] 推理网关与协议转换、多模态请求协议

### 5.3 长上下文
- [ ] 位置外推：YaRN/NTK/ABF 与长上下文训练
- [ ] 序列并行与 RingAttention、Ulysses、LongLoRA
- [ ] 长上下文工程：FastGen/FlexPrefill/IceFormer/StarAttention/HybridAttention
- [ ] Infini-attention 无限上下文、KV 分层压缩
- [ ] 长上下文成本模型与评测

### 5.4 量化与压缩
- [ ] PTQ：RTN/GPTQ/AWQ/SmoothQuant/FP8
- [ ] QAT 量化感知训练、INT4/INT8
- [ ] KV Cache 量化、2:4 结构化稀疏
- [ ] 量化管线：compressed-tensors/llm-compressor、离线量化与误差评估

### 5.5 Serving 与部署
- [ ] 容量规划：吞吐/延迟 SLO、QPS 建模、并发与批大小
- [ ] 推理集群调度：GPU 放置、多租户、任务优先级
- [ ] 分离式推理（disaggregation）部署拓扑
- [ ] Autoscaling、多模型路由（L7/语义路由）、模型网关
- [ ] 成本优化：spot/混部、GPU 利用率
- [ ] 可观测性：prometheus 指标、逐请求延迟追踪、guidellm 容量治理
- [ ] 多 LoRA/多适配器服务（S-LoRA/Punica）、嵌入与重排模型服务
- [ ] 离线批量推理（batch inference、异步任务队列）

### 5.6 性能分析
- [ ] Roofline 模型与 ECM 多级模型、组件化 Roofline（ASPLOS25）
- [ ] Prefill vs Decode 延迟分解、MFU/MBU
- [ ] profiling 实操：nsys/ncu/torch profiler
- [ ] 吞吐-延迟权衡、batching 效应与最优批大小

---

## 六、算子与高性能计算
### 6.0 前置基础：体系结构与操作系统
- [ ] 计算机体系结构：内存层次（cache/局部性）、CPU 流水线、SIMD/向量化、冯诺依曼与哈佛
- [ ] 操作系统：进程线程、虚拟内存、I/O 模型、NUMA
- [ ] 性能分析的底层依据：延迟数字（内存/磁盘/网络）、Amdahl 定律

### 6.1 GPU 编程
- [ ] CUDA 完整知识：线程组织、共享内存、bank conflict、合并访问、occupancy、warp divergence
- [ ] Triton 算子编程
- [ ] CUDA Graph、异步与流水线
- [ ] 常用库：cuBLAS/cuDNN/NCCL

### 6.2 核心内核
- [ ] GEMM 优化方法论：tiling、register blocking、vectorization、CUTLASS 模板库
- [ ] FlashAttention 内核演进、FlashMLA、FlashKDA、DeepGEMM、MSA
- [ ] GEMV 与 decode 阶段瓶颈、融合 kernel
- [ ] megakernel 范式
- [ ] 专家并行内核：UltraEP、moonep
- [ ] Lightning Indexer 算子
- [ ] HPC 算子库工程：源码/测试/基准

### 6.3 通信内核
- [ ] NCCL 源码分析：AllReduce Ring/Tree/双通道
- [ ] AllGather/ReduceScatter、梯度桶
- [ ] 通信与计算重叠、拓扑感知通信调度

### 6.4 AI 编译器与图优化
- [ ] 传统编译器：GCC/LLVM/LLVM IR、前端与后端
- [ ] AI 编译器架构：TVM/MLIR、编译流水线（图优化→算子生成→代码生成）
- [ ] 前端图优化：算子融合、布局转换、常量折叠、CSE/DCE、代数化简
- [ ] 后端优化：循环优化、自动调优（AutoTVM/Ansor）、kernel 生成
- [ ] torch.compile/inductor、Triton 编译器原理（与 6.1 Triton 编程衔接）

### 6.5 性能建模与工具
- [ ] 算术强度分析、memory-bound vs compute-bound 判定
- [ ] Roofline/ECM 建模工具：Empirical-Roofline-Toolkit、LLM-Viewer、inferflux
- [ ] NVIDIA Nsight Systems/Compute、perf
- [ ] FLOPS 分析器：模型/算子 FLOPS 计算

### 6.6 昇腾软件栈
- [ ] 昇腾 NPU 达芬奇架构
- [ ] CANN、AscendC 编程范式
- [ ] MindIE 推理引擎、AscendCL
- [ ] 昇腾算子开发与性能调优

### 6.7 其他硬件编程
- [ ] TPU/XLA（JAX 生态）、Groq LPU
- [ ] 国产芯片开发栈：寒武纪/摩尔线程

---

## 七、AI 硬件与集群基础设施
### 7.0 分布式系统基础
- [ ] 分布式系统原理：一致性（强/最终）、CAP、共识（Raft/Paxos）、故障容错与重试
- [ ] 消息传递与 RPC、分布式存储模型（KV/对象/块）
- [ ] 分布式训练与推理中的系统问题：时钟/幂等/checkpoint、故障恢复
- [ ] 调度理论：任务调度、资源分配、队列模型

### 7.1 芯片架构
- [ ] GPU 原理与演进：Fermi→Blackwell 八代白皮书
- [ ] TensorCore、NVLink/NVSwitch
- [ ] HBM/显存带宽与容量、内存墙、计算墙
- [ ] CPU/NPU 基础、Chiplet/先进封装、存算一体、CXL 内存池化
- [ ] 能效比：PUE、每瓦性能

### 7.2 芯片格局
- [ ] NVIDIA 生态、Apple DOJO、Google TPU 1-4、Groq
- [ ] 国产芯片：百度 BR100、寒武纪、昇腾、算源

### 7.3 AI 集群
- [ ] 机房基础：服务器形态、风冷/液冷、机柜、供电
- [ ] 超节点设计：SuperPod 挑战/方案/演进、CloudMatrix
- [ ] 集群路线图：硬件/软件/应用趋势、架构设计

### 7.4 网络
- [ ] RDMA、InfiniBand vs RoCE、无损网络
- [ ] 网络拓扑：fat-tree/Dragonfly、拓扑算法
- [ ] 拥塞控制、负载均衡、DPU/SmartNIC
- [ ] 光互联：CPO 共封装光学、线性直驱（LPO）
- [ ] 大模型训练流量模式与带宽需求计算

### 7.5 存储
- [ ] 并行文件系统：Lustre/GPFS/WEKA
- [ ] 对象存储（S3/Ceph）、checkpoint 存储设计
- [ ] AI 存储与数据湖

### 7.6 调度与运维
- [ ] Slurm 调度器、Kubernetes（编排/CSI/网络/调度）
- [ ] GPU 共享与虚拟化：MIG/vGPU、任务优先级与抢占
- [ ] GPU 故障诊断（Xid/掉卡/温度）、功耗与热管理
- [ ] 集群监控告警、实际集群案例
- [ ] AI 硬件评测基准：MLPerf Training/Inference

---

## 八、Agent 与应用层
### 8.1 Agent
- [ ] 范式：ReAct、Plan-and-Execute、Reflexion
- [ ] Function Calling 协议、结构化输出
- [ ] Agent 框架：LangGraph/AutoGen/CrewAI/OpenAI Agents SDK
- [ ] 多智能体协作、Agent 记忆（短期/长期/向量）
- [ ] Agent 评测：SWE-bench/GAIA
- [ ] Computer Use 与浏览器智能体、语音 Agent（ASR/TTS、全双工对话）

### 8.2 MCP（Model Context Protocol）
- [ ] MCP 协议规范、Server/Client 架构
- [ ] 工具注册与安全边界、资源/提示词能力
- [ ] MCP 服务开发实践

### 8.3 RAG
- [ ] 分块策略、嵌入模型、向量检索
- [ ] 混合检索（BM25+向量）、重排（Reranker）
- [ ] Advanced/Modular/Agentic RAG
- [ ] RAG 评测、幻觉问题
- [ ] GraphRAG 与知识图谱增强检索

### 8.4 向量数据库
- [ ] FAISS/Milvus/pgvector/Qdrant/Chroma 对比
- [ ] HNSW/IVF 索引、量化检索

### 8.5 Prompt 工程与 LLMOps
- [ ] Prompt 工程：CoT/ToT/少样本/角色设定
- [ ] LLMOps：成本监控、A/B 测试、反馈闭环、可观测性（langfuse）
- [ ] 应用编排平台：Dify/Coze/n8n、工作流引擎

### 8.6 安全与行业应用
- [ ] 越狱/红队/提示注入、对齐安全、攻击与防御
- [ ] 模型安全：数据隐私与合规（GDPR）、模型水印、供应链安全
- [ ] 自动驾驶、具身智能、代码生成、对话产品

### 8.7 Agent 运行时基础设施
- [ ] Agent 执行环境：代码沙箱、浏览器自动化、工具运行时
- [ ] AgentOps：轨迹追踪、成本监控、可观测性（langfuse/LangSmith）

---

## 九、模型生态与前沿
### 9.1 开源模型谱系
- [ ] GPT 系列、LLaMA 1-4、Qwen 1-3
- [ ] DeepSeek V2/V3/R1、GLM 系列、Kimi、Mistral/Mixtral、Gemma、Phi
- [ ] 各代际关键创新点：注意力/数据/训练方法演进
- [ ] 开源协议与合规：Apache/MIT/Llama License/CC

### 9.2 世界模型
- [ ] Matrix-Game 系列：世界模型验证环境、3.0/3.5
- [ ] 世界模型理论：minWM/MoWorld/LingBot/BlockVid/Causal Forcing/Flash-VAED/Lyra
- [ ] 世界模型工程：ForgeWM、NVIDIA Cosmos、MAGI-2、mammothmoda
- [ ] 3D 引导生成、3DGS 高斯泼溅、LLM 记忆综述
- [ ] 世界模型评测：一致性/长时稳定性

### 9.3 多模态与视频生成
- [ ] 多模态架构：ViT、CLIP 对比学习、视觉编码器
- [ ] 多模态对齐训练：图文对齐、指令跟随（LLaVA 范式）、统一多模态模型（GPT-4V/Qwen-VL 架构）
- [ ] 文生图/视频/音频生成：Wan、DiT、Flux
- [ ] 扩散模型 + MoE 优化范式
- [ ] 视频生成作为世界模型
- [ ] 模态融合与 tokenizer：图像 tokenizer（VQ-VAE）、视频 tokenizer、音频 tokenizer

### 9.4 推理时扩展与前沿
- [ ] o1/o3 机制、DeepSeek-R1 复现（RL+搜索+CoT）
- [ ] 思维链蒸馏、快慢双速推理
- [ ] 前沿方向：长思维链、Agentic 推理、合成数据循环、小模型蒸馏
- [ ] 机制可解释性：SAE 稀疏自编码器、电路分析、注意力模式研究、模型行为分析
- [ ] AI for Science：蛋白质/材料/科学大模型（AlphaFold/ESM）

### 9.5 必读论文清单
- [ ] 奠基：Attention Is All You Need、GPT-2/3、BERT、InstructGPT
- [ ] 推理：FlashAttention、vLLM/PagedAttention、Speculative Decoding、Medusa
- [ ] 训练：LoRA、ZeRO、Megatron-LM、Chinchilla、RLHF、DPO、GRPO
- [ ] 前沿：DeepSeek-V3/R1、Mamba、DiT、Sora、世界模型系列
- [ ] 论文阅读方法：三遍法、复现实验、笔记体系

---

## 十、工程实践与工具链
### 10.1 软件工程
- [ ] Git 高级：rebase/merge 策略、子模块、回滚
- [ ] CI/CD、代码评审、测试（单元/集成/回归）、monorepo
- [ ] LLM 应用评测自动化

### 10.2 开发工具链
- [ ] Python 数据生态：numpy/pandas/matplotlib、Jupyter
- [ ] 实验管理：W&B/MLflow、配置管理（hydra）
- [ ] 远程 GPU 工作流：SSH 端口转发、tmux/screen、rsync
- [ ] Vibe Coding：提示工程、迭代调试、测试验收、边界与风险

### 10.3 部署与运维
- [ ] Docker/K8s、模型服务部署模式（在线/批量/边缘）
- [ ] 模型仓库与格式：HuggingFace Hub、Safetensors/GGUF/ONNX 格式、模型版本管理与注册
- [ ] GPU 资源配置、镜像管理
- [ ] 监控告警：Prometheus/Grafana、日志收集、成本报表
- [ ] 性能压测：locust/k6、MLPerf loadgen

### 10.4 知识管理与产出
- [ ] 笔记体系：Obsidian/学习笔记站构建、Markdown 文档站
- [ ] 架构图工程：draw.io/Mermaid、文档自动化（PDF 构建）
- [ ] 技术写作、讲义与 PPT、对外输出

### 10.5 职业发展
- [ ] 面试准备：八股/算法刷题/系统设计/项目深挖
- [ ] 论文阅读与投稿、开源贡献流程

---

## 十一、专业术语表
> 按领域分组的核心术语（中英对照 + 一句话释义），背完打勾。

### 11.1 深度学习基础
- [ ] **反向传播 Backpropagation**：用链式法则逐层计算梯度的算法
- [ ] **梯度消失/爆炸 Vanishing/Exploding Gradient**：深层网络梯度过小/过大的问题
- [ ] **正则化 Regularization**：L1/L2/Dropout 等抑制过拟合的手段
- [ ] **学习率调度 LR Scheduler**：warmup、cosine decay 等随训练调整步长的策略
- [ ] **归一化 Normalization**：BN/LN/RMSNorm，稳定训练分布
- [ ] **交叉熵 Cross Entropy**：分类任务的标准损失函数
- [ ] **KL 散度 KL Divergence**：度量两个分布差异的指标
- [ ] **困惑度 Perplexity**：语言模型对文本不确定度的度量
- [ ] **Tokenizer（BPE/WordPiece/SentencePiece/BBPE）**：文本切分为子词 token 的方法
- [ ] **词嵌入 Word Embedding**：词的稠密向量表示
- [ ] **权重绑定 Tied Embedding**：输入输出嵌入共享参数

### 11.2 Transformer 与架构
- [ ] **自注意力 Self-Attention**：序列内按相关性加权聚合的机制
- [ ] **QKV（Query/Key/Value）**：注意力三要素，查询/键/值
- [ ] **缩放点积注意力 Scaled Dot-Product Attention**：QK^T/√d 后接 softmax
- [ ] **MHA/MQA/GQA**：多头/多查询/分组查询注意力
- [ ] **MLA（Multi-head Latent Attention）**：DeepSeek 的低秩压缩注意力
- [ ] **KV Cache**：缓存历史键值避免重复计算的显存结构
- [ ] **FlashAttention**：分块+在线 softmax 的高效注意力实现
- [ ] **PagedAttention**：vLLM 的分页式 KV 缓存管理
- [ ] **稀疏注意力 Sparse Attention**：按窗口/全局/选择性子集计算（DSA/NSA）
- [ ] **RoPE（Rotary Position Embedding）**：旋转位置编码
- [ ] **ALiBi**：线性偏置注意力位置编码
- [ ] **因果掩码 Causal Mask**：禁止看到未来 token 的注意力掩码
- [ ] **残差连接 Residual Connection**：跨层加法，缓解退化
- [ ] **MoE（Mixture of Experts）**：路由到多个专家子网络
- [ ] **负载均衡损失 Load Balancing Loss**：鼓励专家均匀使用的辅助损失
- [ ] **SSM（State Space Model）**：状态空间模型（Mamba 基础）
- [ ] **扩散模型 Diffusion Model**：噪声逐步去噪生成（DDPM）
- [ ] **流匹配 Flow Matching**：扩散模型的连续化形式
- [ ] **DiT（Diffusion Transformer）**：以 Transformer 为骨干的扩散模型
- [ ] **VAE 重参数化 Reparameterization**：采样过程重写为可反传的形式
- [ ] **世界模型 World Model**：在内部建模环境动态的模型

### 11.3 预训练
- [ ] **语言建模 LM Objective**：预测下一个 token 的训练目标
- [ ] **掩码语言建模 MLM**：BERT 式遮词预测
- [ ] **Scaling Law**：损失随参数/数据/算力的幂律关系
- [ ] **Chinchilla 最优**：参数与数据按 20:1 匹配的计算最优原则
- [ ] **MinHash 去重**：近似集合去重的哈希技术
- [ ] **合成数据 Synthetic Data**：由模型生成的数据
- [ ] **数据配比 Data Mixing**：多来源数据比例设计
- [ ] **DP/TP/PP/EP/CP/SP**：数据/张量/流水线/专家/上下文/序列并行
- [ ] **ZeRO**：零冗余优化器，切分优化器状态/梯度/参数
- [ ] **梯度累积 Gradient Accumulation**：多步累加模拟大 batch
- [ ] **激活重计算 Activation Checkpointing**：前向丢弃激活、反向重算
- [ ] **混合精度 Mixed Precision**：FP16/BF16/FP8 混合计算
- [ ] **梯度裁剪 Gradient Clipping**：限制梯度范数防爆炸
- [ ] **loss spike**：训练中损失突然飙升的现象
- [ ] **集合通信 Collective Communication**：AllReduce/AllGather/ReduceScatter 等
- [ ] **MFU（Model FLOPs Utilization）**：模型算力利用率
- [ ] **MTP（Multi-Token Prediction）**：一次预测多个未来 token
- [ ] **断点续训 Checkpoint/Resume**：保存与恢复训练状态

### 11.4 后训练
- [ ] **SFT（Supervised Fine-Tuning）**：监督微调
- [ ] **指令微调 Instruction Tuning**：按指令-回答对微调
- [ ] **PEFT**：参数高效微调总称
- [ ] **LoRA（Low-Rank Adaptation）**：低秩增量矩阵微调
- [ ] **QLoRA**：4bit 量化 + LoRA
- [ ] **RLHF**：人类反馈强化学习
- [ ] **奖励模型 Reward Model**：给输出打分的模型
- [ ] **PPO**：近端策略优化，RLHF 主算法
- [ ] **GAE（Generalized Advantage Estimation）**：广义优势估计
- [ ] **DPO**：直接偏好优化，免奖励模型的对齐
- [ ] **GRPO**：分组相对策略优化（DeepSeek-R1 用）
- [ ] **RLVR**：可验证奖励强化学习（数学/代码）
- [ ] **蒸馏 Distillation/KD**：大模型知识迁移给小模型
- [ ] **对齐 Alignment**：使模型行为符合人类意图
- [ ] **灾难性遗忘 Catastrophic Forgetting**：新任务学习覆盖旧知识
- [ ] **幻觉 Hallucination**：模型生成不实内容
- [ ] **CoT（Chain-of-Thought）**：思维链提示
- [ ] **LLM-as-judge**：用模型当评测员
- [ ] **基准 Benchmark（MMLU/GSM8K/HumanEval）**：知识/数学/代码评测集
- [ ] **Harness（评测/训练脚手架）**：标准化驱动模型评测或训练的框架（lm-evaluation-harness、HELM、deepseek-harness）
- [ ] **RLAIF**：AI 反馈强化学习（Constitutional AI）

### 11.5 推理与 Serving
- [ ] **自回归解码 Autoregressive Decoding**：逐 token 生成
- [ ] **Prefill / Decode**：预填充（并行处理输入）与解码（逐 token 生成）两阶段
- [ ] **连续批处理 Continuous Batching**：请求级动态调度批处理
- [ ] **Chunked Prefill**：预填充分块以降低首 token 延迟
- [ ] **PD 分离（Disaggregated Prefill/Decode）**：两阶段分机部署
- [ ] **投机解码 Speculative Decoding**：小模型草稿+大模型验证
- [ ] **接受率 Acceptance Rate**：草稿 token 被接受的比例
- [ ] **前缀缓存 Prefix Caching / RadixAttention**：共享前缀复用 KV
- [ ] **采样 Sampler（top-k/top-p/temperature/min-p）**：解码随机策略
- [ ] **TTFT / TPOT**：首 token 时间 / 每 token 输出时间
- [ ] **量化 Quantization**：低精度表示（PTQ/QAT、GPTQ/AWQ/SmoothQuant）
- [ ] **结构化稀疏 2:4**：每 4 个元素保留 2 个的硬件加速稀疏
- [ ] **SLO（Service Level Objective）**：服务等级目标（延迟/吞吐）
- [ ] **容量规划 Capacity Planning**：按 QPS/并发规划 GPU 数量
- [ ] **分离式推理 Disaggregation**：见 PD 分离
- [ ] **位置外推 Extrapolation（YaRN/NTK）**：把短训练长度外推到更长
- [ ] **RingAttention**：环形分块长序列注意力
- [ ] **可观测性 Observability**：指标/日志/追踪三支柱
- [ ] **结构化输出 Structured Output**：约束解码为 JSON/grammar
- [ ] **多 LoRA 服务 S-LoRA**：单 GPU 并发服务多个适配器

### 11.6 算子与 HPC
- [ ] **CUDA 线程模型（block/grid/warp）**：GPU 并行执行层级
- [ ] **共享内存 Shared Memory**：块内高速片上存储
- [ ] **bank conflict**：多线程访问同一内存 bank 导致的串行化
- [ ] **occupancy**：SM 上活跃线程占比
- [ ] **合并访问 Coalesced Access**：相邻线程访问连续地址
- [ ] **GEMM / GEMV**：矩阵乘 / 矩阵向量乘
- [ ] **CUTLASS**：NVIDIA CUDA C++ 模板矩阵乘库（FlashAttention 的 GEMM 基础）
- [ ] **Tiling**：分块计算以利用片上存储
- [ ] **Register Blocking**：寄存器级数据复用
- [ ] **CUDA Graph**：预录 kernel 序列减少启动开销
- [ ] **Triton**：OpenAI 的类 Python 算子语言
- [ ] **Roofline 模型**：以算术强度定性能上限的分析模型
- [ ] **算术强度 Arithmetic Intensity**：FLOP/Byte 比值
- [ ] **Memory-bound / Compute-bound**：受限于访存/计算
- [ ] **ECM 模型**：执行缓存多级性能模型
- [ ] **NCCL**：NVIDIA 集合通信库
- [ ] **IR（中间表示）**：编译器内部表示（LLVM IR/MLIR）
- [ ] **算子融合 Operator Fusion**：合并多个 kernel 减少访存
- [ ] **CSE / DCE**：公共子表达式消除 / 死代码消除
- [ ] **自动调优 Auto-tuning**：搜索最优 kernel 配置（AutoTVM/Ansor）
- [ ] **CANN / AscendC**：华为昇腾计算架构与编程语言
- [ ] **达芬奇架构 DaVinci**：昇腾 NPU 的 AI Core 架构
- [ ] **Nsight（nsys/ncu）**：NVIDIA 系统/内核性能分析器

### 11.7 硬件与集群
- [ ] **TensorCore**：NVIDIA 矩阵加速单元
- [ ] **NVLink / NVSwitch**：GPU 高速互联与交换
- [ ] **HBM**：高带宽显存
- [ ] **CXL（Compute Express Link）**：跨设备内存一致性互联
- [ ] **内存墙 Memory Wall**：访存速度远低于计算速度的瓶颈
- [ ] **PUE**：数据中心能耗效率比
- [ ] **液冷/风冷 Liquid/Air Cooling**：散热方案
- [ ] **RDMA / RoCE / InfiniBand**：远程直接内存访问网络
- [ ] **无损网络 Lossless Network**：不丢包的网络（PFC 等）
- [ ] **拥塞控制 Congestion Control**：网络流量控制
- [ ] **fat-tree 拓扑**：叶脊多级交换网络
- [ ] **超节点 SuperPod**：大规模 GPU 高速互连单元
- [ ] **MIG（Multi-Instance GPU）**：单卡切分为多实例
- [ ] **Slurm**：HPC 作业调度器
- [ ] **CSI（Container Storage Interface）**：K8s 存储插件接口
- [ ] **并行文件系统（Lustre/GPFS）**：高性能分布式文件系统
- [ ] **一致性 Consistency / CAP / 共识 Consensus（Raft）**：分布式系统基础理论
- [ ] **DPU/SmartNIC**：可编程网络处理芯片

### 11.8 Agent 与应用
- [ ] **Agent**：能感知、规划、调用工具完成任务的智能体
- [ ] **ReAct**：推理+行动交替的 Agent 范式
- [ ] **Function Calling**：模型调用外部函数的协议能力
- [ ] **多智能体 Multi-Agent**：多个 Agent 协作
- [ ] **MCP（Model Context Protocol）**：模型与工具/数据源的开放协议
- [ ] **RAG（Retrieval-Augmented Generation）**：检索增强生成
- [ ] **向量检索 Vector Search**：按语义相似度检索
- [ ] **重排 Reranking**：对召回结果二次排序
- [ ] **HNSW / IVF**：近似最近邻索引结构
- [ ] **提示注入 Prompt Injection**：恶意指令注入攻击
- [ ] **越狱 Jailbreak**：绕过模型安全限制
- [ ] **红队 Red Teaming**：主动攻击测试模型安全
- [ ] **ToT（Tree of Thoughts）**：思维树搜索
- [ ] **GraphRAG**：知识图谱增强的 RAG
- [ ] **Computer Use**：操作 GUI/浏览器的智能体
- [ ] **AgentOps**：Agent 生产运维（追踪/监控）
- [ ] **沙箱 Sandbox**：隔离的代码执行环境

### 11.9 模型生态
- [ ] **基础模型 Foundation Model**：可微调复用的通用大模型
- [ ] **推理时扩展 Test-Time Scaling/Compute**：推理阶段增加算力换质量
- [ ] **思维链蒸馏 CoT Distillation**：把长思维过程蒸馏进模型
- [ ] **模型合并 Model Merging**：多模型权重融合
- [ ] **CLIP**：图文对比学习模型
- [ ] **ViT（Vision Transformer）**：Transformer 视觉骨干
- [ ] **LLaVA 范式**：视觉编码器+投影层+LLM 的多模态结构
- [ ] **3DGS（3D Gaussian Splatting）**：三维高斯泼溅渲染
- [ ] **SAE（Sparse Autoencoder）**：稀疏自编码器，可解释性工具
- [ ] **TTT（Test-Time Training）**：在推理中更新参数的层
- [ ] **dLLM（Diffusion LLM）**：扩散式语言模型
- [ ] **Sora 类视频模型**：DiT+VAE 的可扩展视频生成

### 11.10 工程与工具
- [ ] **CI/CD**：持续集成/持续部署
- [ ] **monorepo**：单仓库多项目工程
- [ ] **实验追踪 Experiment Tracking（W&B/MLflow）**：训练实验记录与对比
- [ ] **模型格式 Safetensors/GGUF/ONNX**：模型序列化格式
- [ ] **模型注册 Model Registry**：模型版本管理
- [ ] **MLPerf**：AI 训练/推理硬件评测基准
- [ ] **Jekyll**：GitHub Pages 默认静态站点生成器
- [ ] **任务列表 Task List**：Markdown `- [ ]` 勾选语法

---

## 十二、个人项目
> 学以致用区：推进地图的过程中，把动手做的项目记录在这里。
> 按 12.1 模板逐个添加；每个项目标注关联的知识点条目，形成「学—做」闭环。

### 12.1 项目模板（复制即可新建）

#### 项目：〈项目名〉
- [ ] 状态：规划中 / 进行中 / 已完成
- [ ] 一句话简介：做什么、解决什么问题
- [ ] 关联知识点：领域 X → X.X 条目（链接到上方对应小节）
- [ ] 技术栈：语言 / 框架 / 硬件
- [ ] 仓库/演示：GitHub 仓库或在线 demo 链接
- [ ] 深入介绍：架构设计、关键实现、踩坑记录（单开小节或外链）
- [ ] 成果与收获：可展示的产出、学到的知识点、可复用的经验

### 12.3 我的项目

> 逐个添加中……

---

## 学习路径建议

<pre class="mermaid">
flowchart LR
  A["领域一 基础"] --> B["领域五 推理引擎 Serving"]
  A --> C["领域三 预训练（并行与加速）"]
  B --> D["领域四 后训练：对齐与微调"]
  C --> D
  D --> E["领域六 算子/HPC"]
  E --> F["领域二 Transformer 架构深挖"]
  F --> G["领域七 硬件集群"]
  G --> H["领域八 Agent 应用"]
  H --> I["领域九 前沿生态"]
  J["领域十 工程工具＋术语表＋论文＋个人项目（做中学）"] -.贯穿全程.-> A
  J -.贯穿全程.-> D
  J -.贯穿全程.-> G
</pre>

