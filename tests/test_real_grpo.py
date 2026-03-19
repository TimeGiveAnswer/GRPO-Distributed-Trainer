import ray
import os
from src.trainer.grpo_ray_trainer import train_loop_per_worker
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

def main():
    # 强制清理显存残留
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    # 路径核对：确保这是你 3B 模型的真实位置
    model_path = "/mnt/GRPO_project/models/Qwen2.5-3B-Instruct"

    config = {
        "model_path": model_path,
        "lr": 1e-6,           # RL 阶段学习率要极小
        "total_steps": 20,    # 测试跑 20 步即可
        "group_size": 8,      # GRPO 组大小
        "beta": 0.04,         # KL 惩罚系数
    }

    # 配置 2 张 A100
    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=True,
        resources_per_worker={"GPU": 1}
    )

    trainer = TorchTrainer(
        train_loop_per_worker,
        train_loop_config=config,
        scaling_config=scaling_config
    )

    print("🛰️ 正在拉起分布式进程，请观察 nvidia-smi...")
    trainer.fit()

if __name__ == "__main__":
    main()