import torch
import ray
import ray.train.torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
# 导入 FSDP2 所需的 composable API
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
# 导入旧版 FSDP 仅用于权重导出的配置（如果需要兼容之前的保存逻辑）
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
import os
import shutil

# 假设你的文件结构
from .grpo_engine import GRPOEngine
from .vllm_sampler_actor import VLLMSamplerActor
from ..utils.math_lighteval import extract_solution

# --- 🛰️ 权重管理类 (适配 vLLM 同步和最终保存) ---
class CheckpointManager:
    @staticmethod
    def save_to_disk(model, tokenizer, path, rank):
        """保存 FSDP 模型为 HF 格式"""
        # FSDP2 的简单导出逻辑：在 context 开启下提取 state_dict
        # 注意：这里假设你使用的是简单的 save_pretrained 逻辑
        if rank == 0:
            if os.path.exists(path): shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            # 对于 composable fsdp，通常直接对原 module 调用 save_pretrained 并传入 state_dict
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)

# --- 训练循环函数 ---
def train_loop_per_worker(config):
    import torch.distributed as dist  # 内部导入也行，更保险
    import torch
    # 1. 基础环境获取
    rank = ray.train.get_context().get_world_rank()
    local_rank = ray.train.get_context().get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    vllm_sampler = config["vllm_sampler_handle"]
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---------------------------------------------------------
    # 🛰️ 关键点 1: 显式加载并强制全员集结
    # ---------------------------------------------------------
    print(f"🛰️ [Rank {rank}] 正在加载模型 (这可能需要 1-2 分钟)...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.gradient_checkpointing_enable()

    ref_model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    ref_model.eval()

    # FSDP 包装
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    model = fully_shard(model, mp_policy=mp_policy)
    ref_model = fully_shard(ref_model, mp_policy=mp_policy)

    # 引擎与优化器
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=0.01)
    scheduler = get_scheduler("cosine", optimizer, 0, config["total_steps"])
    engine = GRPOEngine(model, ref_model, optimizer, scheduler, config)

    # 🚩 核心同步：只有所有 Rank 都执行到这一行，才会一起往下走
    print(f"⌛ [Rank {rank}] 加载完毕，进入起跑线等待...")
    dist.barrier() 
    print(f"🚀 [Rank {rank}] 所有人到齐！开始 Step 0")

    # 3. 数据处理
    dataset = load_dataset("parquet", data_files=config["data_path"])["train"]
    # 确保数据集长度对齐，取最小值防止某个进程早退导致死锁
    num_samples = len(dataset) // ray.train.get_context().get_world_size()
    worker_dataset = dataset.shard(num_shards=ray.train.get_context().get_world_size(), index=rank, contiguous=True)

    sync_path = "/dev/shm/grpo_sync_model"

    for step in range(config["total_steps"]):
        # 🟢 A. 步头强制同步：每一轮开始都对齐一次
        dist.barrier()
        
        # 获取当前样本
        item = worker_dataset[step]
        prompt_text = item["prompt"][0]["content"]
        ground_truth = item["reward_model"]["ground_truth"]

        # ---------------------------------------------------------
        # 🟢 B. 广播采样逻辑 (彻底解决抢跑)
        # ---------------------------------------------------------
        # 这个容器在所有 Rank 上都必须定义
        broadcast_container = [None] 

        if rank == 0:
            print(f"📡 [Rank 0] Step {step} 正在请求 vLLM...")
            try:
                # 只有 Rank 0 去拿数据
                raw_completions = ray.get(vllm_sampler.get_samples.remote(prompt_text), timeout=300)
                broadcast_container = [raw_completions]
            except Exception as e:
                print(f"🚨 [Rank 0] 采样异常: {e}")
                broadcast_container = [[""] * config["group_size"]]
        
        # 📣 全员广播：Rank 1 会卡在这里等 Rank 0 拿完数据发过来
        dist.broadcast_object_list(broadcast_container, src=0)
        raw_completions = broadcast_container[0]

        # 异常跳过（全员同步跳过）
        if not raw_completions or raw_completions[0] == "":
            print(f"⚠️ [Rank {rank}] 采样为空，跳过 Step {step}")
            continue

        # ---------------------------------------------------------
        # 🟢 C. 奖励与训练 (所有 Rank 拿到相同的数据进行本地计算)
        # ---------------------------------------------------------
        rewards = []
        for text in raw_completions:
            ans = extract_solution(text)
            acc = 1.0 if str(ans) == str(ground_truth) else 0.0
            fmt = 0.1 if "\\boxed{" in text else 0.0
            rewards.append(acc + fmt)
        
        total_rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        # 预处理 IDs
        inputs = tokenizer(raw_completions, padding=True, truncation=True,
                          max_length=config["max_new_tokens"], return_tensors="pt").to(device)
        prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        batch = {"input_ids": inputs.input_ids, "rewards": total_rewards, "prompt_mask": prompt_len}
        
        # 执行训练步
        model.train() 
        loss_val, p_loss, k_loss, g_norm = engine.train_step(batch, device)
        
        # 🟢 D. 步尾同步：确保 train_step 里的分布式操作全部闭环
        dist.barrier()

        # ---------------------------------------------------------
        # 🟢 E. 权重同步逻辑
        # ---------------------------------------------------------
        if (step + 1) % config["sync_steps"] == 0:
            if rank == 0:
                print(f"📦 [Rank 0] 同步权重到 vLLM...")
                model.save_pretrained(sync_path)
                ray.get(vllm_sampler.update_model.remote(sync_path))
            # 所有人等 Rank 0 存完读完再走
            dist.barrier()

        # 🟢 F. 日志汇报
        if rank == 0:
            avg_reward = total_rewards.mean().item()
            print(f"🚀 Step {step} | Loss: {loss_val:.4f} | Reward: {avg_reward:.2f}")
            ray.train.report({"loss": loss_val, "reward": avg_reward, "kl": k_loss})
        # 5. 步尾同步：确保报告完成后再进入下一轮循环
        dist.barrier()  
    print(f"🏁 [Rank {rank}] 训练任务全部完成")
# --- 启动逻辑 ---
if __name__ == "__main__":
    ray.init()
    
    # 强制开启显存碎片整理，防止 OOM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config = {
        "model_path": "/mnt/GRPO_project/models/Qwen2.5-3B-Instruct",
        "data_path": "/mnt/GRPO_project/data/datasets/MATH-lighteval2/train.parquet",
        "use_vllm": True,
        "lr": 1e-6,
        "group_size": 4, 
        "max_new_tokens": 512,
        "total_steps": 200,
        "sync_steps": 20,
        "beta": 0.04,
    }

    # 1. 🛰️ 让 vLLM 独占 GPU 0
    # 我们通过 .options 显式指定它占用 1 个 GPU 资源
    vllm_sampler = VLLMSamplerActor.options(num_gpus=1).remote(
        config["model_path"],
        {
            **config,
            "gpu_memory_utilization": 0.8, # 👈 因为独占，可以给到 0.8，保证采样速度
            "max_model_len": 1024,
        }
    )
    config["vllm_sampler_handle"] = vllm_sampler

    # 2. 🏋️ 让训练 Worker 占用剩下的 GPU
    # 你有 4 张卡，1 张给了 vLLM，剩下 3 张给训练
    # 这样每个 Worker 都会被分配到独立的 GPU (GPU 1, 2, 3)
    scaling_config = ScalingConfig(
        num_workers=3,        # 👈 修改为 3 个 Worker
        use_gpu=True, 
        resources_per_worker={"GPU": 1} 
    )

    trainer = TorchTrainer(
        train_loop_per_worker, 
        train_loop_config=config, 
        scaling_config=scaling_config
    )
    
    print("🚀 正在启动 1(vLLM) + 3(Training) 混合架构...")
    trainer.fit()