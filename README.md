# GRPO-Distributed-Trainer

基于 **Ray**, **vLLM** 和 **PyTorch FSDP** 实现的高效轻量化 **GRPO** (Group Relative Policy Optimization) 分布式训练框架。

本项目通过 Ray + vLLM + FSDP 的选型，将训练逻辑与复杂的底层基础设施解耦，彻底解决了环境配置难、层级依赖深的问题。
它既吸取了 verl 等成熟框架的设计精髓，又保持了代码的高度简洁与可维护性，实现了性能与易用性的平衡。


## 1. 核心架构设计

为了最大化异构资源利用率，系统采用 **推理-训练分离 (Hybrid Orchestration)** 策略：
* **推理引擎 (vLLM Actor)**: 利用 vLLM 的 Continuous Batching 提高采样吞吐量。
* **训练引擎 (FSDP Workers)**: 在剩余 GPU 上通过 PyTorch FSDP 实现模型参数、梯度和优化器状态的分片（Sharding），大幅降低单卡显存占用。
* **同步控制层**: 针对 Ray 与 NCCL 混合环境下的死锁风险，实现了 **Triple-Barrier** 状态机协议，确保异步采样与同步训练的步调一致。

## 2. 技术规格

| 维度 | 技术栈 |
| :--- | :--- |
| **RL 算法** | GRPO (Group Relative Policy Optimization) |
| **分布式后端** | PyTorch FSDP (Fully Sharded Data Parallel) |
| **推理加速** | vLLM (Eager Mode / CUDA Graph) |
| **资源调度** | Ray Core / Ray Train |
| **计算精度** | BFloat16 混合精度 |

## 3. GRPO 核心实现

### 组内优势函数 (Group-Relative Advantage)
不依赖 Critic 网络，通过组内样本奖励的归一化计算 Advantage：

$$A_{i,j} = \frac{r_{i,j} - \text{mean}(r_{i,k})}{\text{std}(r_{i,k}) + \epsilon}$$

框架通过 `dist.all_gather` 跨卡同步全量 Reward，确保在全局范围内进行组内均值与标准差的准确计算。

### 分布式死锁消除逻辑 (Triple-Barrier)
针对分布式采样中常见的 Rank 0 抢跑及 NCCL 通信挂起问题，强制执行以下同步点：

1.  **采样前置同步 (Pre-sampling Barrier)**: 确保所有训练卡完成上一轮梯度更新并对齐参数后，再进入 vLLM 采样阶段。
2.  **数据广播同步 (Data Broadcast)**: 由 Rank 0 获取采样数据后通过 `dist.broadcast_object_list` 分发，保证所有 Worker 训练输入完全一致。
3.  **汇报后置同步 (Post-reporting Barrier)**: 确保 `ray.train.report` 在分布式通信彻底结束后执行，防止 Rank 0 提前进入下一轮迭代导致 NCCL 环崩溃。

## 4. 目录结构

```text
.
├── src/
│   ├── trainer/
│   │   ├── grpo_engine.py          # GRPO 损失函数与 KL 散度计算核心
│   │   ├── vllm_sampler_actor.py   # vLLM 推理 Actor 封装 (Ray Actor)
│   │   └── grpo_ray_vllm_trainer.py# Ray 编排与主训练循环逻辑
│   ├── rewards/
│   │   └── grpo_rewards.py         # 解耦的奖励函数逻辑 (支持自定义规则)
│   └── utils/
│       └── grpo_utils.py           # 分布式辅助工具与同步原语
├── data/                           # 训练数据集 (支持 Parquet/JSONL)
└── requirements.txt                # 核心依赖清单
```
## 5. 快速开始
环境安装
```
Bash
pip install torch vllm ray transformers pandas
```
启动训练
在启动前请务必清理残留进程以释放显存与 NCCL 句柄：
```
Bash
# 清理环境
pkill -9 python && ray stop
```
# 运行分布式训练 (默认配置：1 vLLM Actor + 3 FSDP Workers)
```
python src/trainer/grpo_ray_vllm_trainer.py
```
## 6. 性能表现 (Benchmark)
测试环境: 4x NVIDIA A100 (40GB)

测试模型: Qwen2.5-3B-Instruct

显存占用: 单卡约 32GB (FSDP 开启状态下)

稳定性: 成功解决分布式死锁问题，支持 100+ Step 连续高压运行。
