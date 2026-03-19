GRPO-Distributed-Trainer一个基于 Ray + vLLM + PyTorch FSDP 的轻量级 GRPO（Group Relative Policy Optimization）分布式训练框架。本项目吸收了阿里巴巴 SWIFT、PAI-Megatron-Patch 及 verl 的核心设计思想，旨在提供一个只需安装 vLLM 即可快速运行的 Reinforcement Learning 训练方案。🌟 项目亮点极简部署：无需复杂的 C++ 编译或重型中间件，核心依赖仅为 vLLM, Ray 和 PyTorch。混合架构 (Hybrid Architecture)：Inference: 使用 vLLM Actor 独立占用 GPU 进行高速采样。Training: 使用 PyTorch FSDP (Fully Sharded Data Parallel) 进行多卡分布式模型更新。分布式死锁规避：针对 Ray Worker 与 vLLM 采样进程之间的异步通信，实现了严格的 Triple-Barrier Synchronization（三点一线同步）机制，确保海量步数训练不挂起。解耦设计：Reward Function 与 Training Engine 彻底解耦，支持自定义数学逻辑、格式检查等多种奖励策略。🚀 快速开始1. 环境准备Bashconda create -n grpo python=3.10
conda activate grpo
pip install torch vllm ray transformers pandas
2. 项目结构Plaintext.
├── src/
│   ├── trainer/          # 核心引擎 (GRPOEngine, RayTrainer)
│   ├── rewards/          # 奖励逻辑 (GRPORewarder)
│   └── utils/            # 通用工具
├── data/                 # 训练数据集 (parquet/jsonl)
└── train_grpo.py         # 启动入口
3. 运行指南只需配置好模型路径和数据路径，即可一键启动 4 卡分布式训练（1 vLLM + 3 Training Workers）：Bash# 清理残留进程 (推荐)
pkill -9 python && ray stop

# 启动训练
python src/trainer/grpo_ray_vllm_trainer.py
🛠️ 核心架构实现组内优势计算 (Group-Relative Advantage)参考 DeepSeek 原理，我们在 GRPOEngine 中实现了跨卡奖励同步：所有 Worker 通过 dist.all_gather 汇总当前 Group 的所有奖励。在全局范围内计算 $\text{mean}$ 和 $\text{std}$。通过 $(R - \mu) / \sigma$ 计算 Advantage，有效降低强化学习中的方差。关键同步逻辑 (Barrier Sync)为了防止分布式训练中的“木桶效应”导致显存溢出或 NCCL 超时，我们在每个 Step 嵌入了物理屏障：Step-Start: 同步所有卡进入采样。Step-End: 强制 Rank 0 等待所有计算 Worker 完成梯度更新后再进行 ray.train.report。📊 运行 Demo 示例在 4 张 A100 (40GB) 环境下运行 Qwen2.5-3B-Instruct：Plaintext🚩 [Rank 0] --- 进入 train_step ---
📈 [Rank 0] Advantage计算完毕, Mean: 0.4333, Std: 0.4924
🔮 [Rank 0] 开始模型前向 (Policy Model)...
📉 [Rank 0] Step完成 | TotalLoss: -1.3682 | KL: 0.0125 | GradNorm: 111.00
🚀 Step 4 完成同步 | Reward: 1.10
