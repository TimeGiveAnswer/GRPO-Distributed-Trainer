import ray
import torch
import os
import time
import gc
from transformers import AutoModelForCausalLM, AutoConfig

# 环境锁死，防止联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH = os.path.abspath("./models/Qwen2.5-3B-Instruct")

if not ray.is_initialized():
    ray.init(num_gpus=2, object_store_memory=20 * 1024**3)

@ray.remote(num_gpus=1)
class TrainerActor:
    def __init__(self, model_path):
        self.device = "cuda:0"
        print(f"Trainer: 正在加载模型...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, local_files_only=True
        ).to(self.device)
        print("Trainer: 状态已就绪 ✅")

    def step_and_export(self):
        # 1. 模拟微小更新
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn_like(p) * 0.001)
                    break
        
        # 2. 【关键】直接返回 CPU 字典
        # Ray 的 remote 函数返回一个大对象时，会自动将其放入共享内存并返回一个 ObjectRef
        return {k: v.detach().cpu().clone().contiguous() for k, v in self.model.state_dict().items()}

@ray.remote(num_gpus=1)
class RolloutActor:
    def __init__(self, model_path):
        self.device = "cuda:0"
        print("Rollout: 正在构建架构...")
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        with torch.device("cpu"):
            self.model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
        print("Rollout: 状态已就绪 ✅")

    def sync_weights(self, weight_ref):
        print("Rollout: 收到权重数据...")
        start = time.time()
        
        # --- 健壮性修复 ---
        # 如果 weight_ref 已经是 dict (解包后的数据)，直接用
        # 如果 weight_ref 是 ObjectRef (单号)，则 ray.get
        if isinstance(weight_ref, dict):
            new_sd = weight_ref
            print("Rollout: 接收到直接字典数据")
        else:
            new_sd = ray.get(weight_ref)
            print("Rollout: 从共享内存拉取数据成功")
        # ------------------
        
        self.model.load_state_dict(new_sd)
        self.model.to(self.device)
        
        del new_sd
        gc.collect()
        return time.time() - start

def main():
    trainer = TrainerActor.remote(MODEL_PATH)
    rollout = RolloutActor.remote(MODEL_PATH)
    
    # 彻底确保初始化结束
    ray.get(trainer.step_and_export.remote())
    print("\n--- 正式开始测试 ---")

    for i in range(1, 3):
        # 拿到 Trainer 吐出来的“单号”
        # 注意：这里我们拿到了引用
        ref = trainer.step_and_export.remote()
        
        # 直接把引用丢给下一级，不要在 Driver 层做任何 get 操作
        # 让两个 Actor 之间自己去“接头”
        cost = ray.get(rollout.sync_weights.remote(ref))
        
        print(f"迭代 {i}: 同步耗时 {cost:.2f}s")

if __name__ == "__main__":
    main()