import torch
import ray.train.torch
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.optim import AdamW

# 导入我们解耦的组件
from .grpo_engine import GRPOEngine
from utils.grpo_utils import GRPORewards

def train_loop_per_worker(config):
    # --- 1. 环境初始化 ---
    rank = ray.train.get_context().get_world_rank()
    local_rank = ray.train.get_context().get_local_rank()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # --- 2. 加载模型与分片 ---
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    
    # FSDP2 关键配置：确保两张卡平分 3B 参数
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    model = fully_shard(model, mp_policy=mp_policy)

    # --- 3. 准备引擎 ---
    optimizer = AdamW(model.parameters(), lr=config.get("lr", 1e-6))
    scheduler = get_scheduler("linear", optimizer, 0, config["total_steps"])
    engine = GRPOEngine(model, optimizer, scheduler, config)

    # --- 4. 真实数据训练循环 ---
    # 定义一个经典的数学推理题作为测试
    test_prompt = "问题：若 x + y = 10 且 x - y = 2，求 x 的值。请给出推理步骤，并将最终结果写在【】内。"
    
    for step in range(config["total_steps"]):
        # A. 采样 (Rollout) - 需要切换到 eval 模式
        model.eval()
        inputs = tokenizer([test_prompt], return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            # 为组采样复制 G 次 (Group Size)
            # 在 A100 上，group_size=8 比较稳妥
            sampling_ids = inputs.input_ids.repeat(config["group_size"], 1)
            output_ids = model.generate(
                sampling_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # B. 奖励计算 (Reward)
        # 提取模型生成的回答部分（扣除 Prompt）
        completions = tokenizer.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        # 计算正确性分(1.0)和格式分(0.2)
        rewards_acc = GRPORewards.check_accuracy(completions, solution="6")
        rewards_fmt = GRPORewards.check_format(completions)
        total_rewards = (rewards_acc + rewards_fmt).to(device)

        # C. 策略更新 (Update)
        batch = {
            "input_ids": output_ids,
            "rewards": total_rewards
        }
        loss_val = engine.train_step(batch, device)
        
        # D. 日志汇报
        avg_r = total_rewards.mean().item()
        if rank == 0:
            print(f"🚀 Step {step} | Loss: {loss_val:.4f} | Reward: {avg_r:.2f}")
            if step % 2 == 0:
                print(f"📖 样本回答: {completions[0][:100]}...") # 打印前100字看效果

        ray.train.report({"loss": loss_val, "reward": avg_r})