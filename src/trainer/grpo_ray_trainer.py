import torch
import ray.train.torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
import os
import sys

# 自动处理路径，确保能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# 导入你刚刚更新好的完整版 GRPOEngine
from src.trainer.grpo_engine import GRPOEngine
from src.utils.math_lighteval import extract_solution
# --- 🛰️ 核心解耦：权重管理类 ---
class CheckpointManager:
    @staticmethod
    def save_fsdp_model(model, tokenizer, config, step, rank):
        """将分片的 FSDP 权重合并并保存为 HF 格式"""
        save_path = os.path.join(config["output_dir"], f"checkpoint-{step}")
        
        # 1. 定义 FSDP 状态导出策略：聚合到 CPU，仅在 Rank 0 执行导出
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        
        # 2. 提取全量 State Dict
        with torch.no_grad():
            state_dict = get_model_state_dict(model, options=options)
        
        # 3. 只有 Rank 0 负责写入磁盘
        if rank == 0:
            print(f"💾 正在导出 Step {step} 的全量权重到 {save_path}...")
            if os.path.exists(save_path):
                shutil.rmtree(save_path)
            os.makedirs(save_path, exist_ok=True)
            
            # 使用 model.module (unwrapped) 来保存，确保兼容 HF
            # 注意：fully_shard 包装后的模型可以通过原始模型实例保存
            model.save_pretrained(save_path, state_dict=state_dict)
            tokenizer.save_pretrained(save_path)
            print(f"✅ 权重保存成功！")
#分布式死锁debug
import os
os.environ["NCCL_P2P_DISABLE"] = "1" # 如果是 PCIE 环境，有时候 P2P 会导致死锁，关掉试试
os.environ["NCCL_DEBUG"] = "INFO"    # 开启 NCCL 日志，死锁在哪一目了然
def train_loop_per_worker(config):
    # --- 1. 分布式环境初始化 ---
    rank = ray.train.get_context().get_world_rank()
    world_size = ray.train.get_context().get_world_size()
    local_rank = ray.train.get_context().get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- 2. 加载 Tokenizer 和模型 ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token
    # 注意：必须设置 padding_side 为 left，否则 generate 出来的结果会有偏置
    tokenizer.padding_side = "left" 
    
    # 加载训练模型
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(device)
    model.gradient_checkpointing_enable()
    

    # --- 3. FSDP2 策略包装 ---
    # 使用混合精度策略节省显存
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    model = fully_shard(model, mp_policy=mp_policy)
        # 加载 Reference 模型 (不计梯度)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(device)
    ref_model.eval()
    ref_model = fully_shard(ref_model, mp_policy=mp_policy)

    # --- 4. 初始化优化器与引擎 ---
    optimizer = AdamW(model.parameters(), lr=config.get("lr", 1e-6), weight_decay=0.01)
    # 计算总步数用于 Scheduler
    scheduler = get_scheduler("cosine", optimizer, 0, config["total_steps"])
    
    # 传入 5 个参数：model, ref_model, optimizer, scheduler, config
    engine = GRPOEngine(model, ref_model, optimizer, scheduler, config)

    # --- 5. 数据准备 ---
    full_dataset = load_dataset("parquet", data_files={"train": config["data_path"]})["train"]
    # 关键：按 Rank 分片数据
    worker_dataset = full_dataset.shard(num_shards=world_size, index=rank)

    # --- 6. 训练主循环 ---
    print(f"📡 Worker {rank} 准备就绪，开始训练...")
    
    for step, item in enumerate(worker_dataset):
        if step >= config["total_steps"]:
            break

        # 解析数据 (根据 MATH-lighteval 格式)
        prompt_text = item["prompt"][0]["content"] 
        ground_truth = item["reward_model"]["ground_truth"]

        # A. 采样阶段 (Rollout)
        model.eval() # 采样时切到 eval 模式
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            group_size = config.get("group_size", 8)
            # 复制输入以进行组采样
            sampling_ids = inputs.input_ids.repeat(group_size, 1)
            
            # 使用 model.generate 进行采样
            output_ids = model.generate(
                sampling_ids,
                max_new_tokens=config.get("max_new_tokens", 512),
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # B. 奖励计算 (Reward)
        # 提取模型生成的 Answer 部分（剔除前面的 Prompt）
        prompt_len = inputs.input_ids.shape[1]
        raw_completions = tokenizer.batch_decode(
            output_ids[:, prompt_len:], 
            skip_special_tokens=True
        )
        
        rewards = []
        for text in raw_completions:
            model_answer = extract_solution(text)
            # 基础奖励逻辑
            acc_reward = 1.0 if model_answer == ground_truth else 0.0
            format_reward = 0.1 if "\\boxed{" in text else 0.0
            rewards.append(acc_reward + format_reward)
            
        total_rewards = torch.tensor(rewards).to(device)

        # C. 策略更新 (Update)
        model.train() # 切回训练模式
        batch = {
            "input_ids": output_ids,
            "rewards": total_rewards,
            "prompt_mask": prompt_len
        }
        
        # 获取训练指标
        loss, p_loss, k_loss, g_norm = engine.train_step(batch, device)
        
        # D. 日志汇报
        avg_reward = total_rewards.mean().item()
        # --- 🛰️ 关键：定期保存逻辑 ---
        if (step + 1) % config["save_steps"] == 0 or step == config["total_steps"] - 1:
            CheckpointManager.save_fsdp_model(model, tokenizer, config, step, rank)
        if rank == 0:
            print(f"🔥 Step {step} | Loss: {loss:.4f} | KL: {k_loss:.4f} | Avg_Reward: {avg_reward:.2f}")
            if step % 5 == 0:
                print(f"📝 [Sample Answer]: {raw_completions[0][:150]}...")

        # 向 Ray 汇报进度
        ray.train.report({"loss": loss, "reward": avg_reward, "kl": k_loss})

# --- 7. 启动脚本 ---
if __name__ == "__main__":
    # 路径配置
    BASE_DIR = "/mnt/GRPO_project"
    config = {
        "model_path": os.path.join(BASE_DIR, "models/Qwen2.5-3B-Instruct"),
        "data_path": os.path.join(BASE_DIR, "data/datasets/MATH-lighteval2/train.parquet"),
        "output_dir": os.path.join(BASE_DIR, "output/GRPO_model"),
        "lr": 1e-6,
        "total_steps": 200,
        "save_steps": 50,      # <--- 加上这一行，每 50 步存一次
        "group_size": 4,
        "max_new_tokens": 512,
        "beta": 0.04,
    }
    # 设置分布式配置：2 张卡，每个 worker 占 1 张
    scaling_config = ScalingConfig(
        num_workers=2, 
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 4}
    )

    trainer = TorchTrainer(
        train_loop_per_worker, 
        train_loop_config=config, 
        scaling_config=scaling_config
    )
    
    print("🚀 GRPO 训练启动中...")
    result = trainer.fit()
    print("✅ 训练完成！")