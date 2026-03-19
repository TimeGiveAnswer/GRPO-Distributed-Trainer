import ray
import time
import torch
import asyncio
from vllm import LLM, SamplingParams

# --- 1. 模拟你的 Actor 逻辑 ---
@ray.remote(num_gpus=1)
class VLLMSamplerActor:
    def __init__(self, model_path, group_size=4):
        print(f"🚀 正在初始化 vLLM...")
        self.group_size = group_size
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.8, # 既然是测试，可以给大一点
            enforce_eager=True,          # 👈 核心：禁用 CUDA Graph
            max_model_len=1024           # 👈 核心：限制长度
        )
        print(f"✅ vLLM 初始化成功！")

    async def get_samples(self, prompt_text):
        try:
            sampling_params = SamplingParams(
                n=self.group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=256,
            )
            # 模拟异步非阻塞调用
            loop = asyncio.get_event_loop()
            print(f"🔮 正在生成采样（Group Size: {self.group_size}）...")
            
            # 使用 lambda 包装同步的 generate
            outputs = await loop.run_in_executor(None, 
                lambda: self.llm.generate([prompt_text], sampling_params, use_tqdm=False)
            )
            
            if outputs:
                return [o.text for o in outputs[0].outputs]
            return ["EMPTY_RESULT"] * self.group_size
        except Exception as e:
            return [f"ERROR: {str(e)}"] * self.group_size

# --- 2. 采样测试主程序 ---
def main():
    # 初始化 Ray
    if not ray.is_initialized():
        ray.init()

    model_path = "/mnt/GRPO_project/models/Qwen2.5-3B-Instruct" # 确认路径正确
    test_prompt = "Question: What is 2+2? Answer: "
    group_size = 4

    print("🛰️ 正在拉起 VLLMSamplerActor...")
    sampler = VLLMSamplerActor.remote(model_path, group_size=group_size)

    # 进行 3 次连续采样测试
    for i in range(3):
        print(f"\n--- ⚡ 测试轮次 {i+1} ---")
        start_time = time.time()
        
        try:
            # 发起请求
            future = sampler.get_samples.remote(test_prompt)
            results = ray.get(future, timeout=60)
            
            duration = time.time() - start_time
            print(f"🕒 耗时: {duration:.2f}s")
            
            for idx, res in enumerate(results):
                print(f"📄 Sample {idx}: {res[:100]}...") # 只打印前100字
                
            if any("ERROR" in r for r in results):
                print("❌ 采样结果中包含错误消息！")
            else:
                print("✅ 采样成功！")

        except Exception as e:
            print(f"🚨 采样超时或崩溃: {e}")

    ray.shutdown()

if __name__ == "__main__":
    main()