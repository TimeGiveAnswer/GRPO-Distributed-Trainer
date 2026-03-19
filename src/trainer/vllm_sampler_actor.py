import torch
import ray
import gc
import os
import asyncio
import traceback
from vllm import LLM, SamplingParams

@ray.remote(num_gpus=1) # 👈 确保独占 GPU 0
class VLLMSamplerActor:
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.group_size = config.get("group_size", 4)
        
        # 🔒 核心：增加异步锁，防止 Rank 0 和 Rank 1 同时进入 generate 导致底层算子冲突
        self.lock = asyncio.Lock()
        
        # 初始化引擎
        self._init_engine(self.model_path)
        print(f"✅ [vLLM Actor] 引擎初始化成功，模型路径: {self.model_path}")

    def _init_engine(self, path):
        """内部初始化函数，支持热更新"""
        # 强制设置一些环境变量，防止 FlashAttention 在某些显卡上报 AssertionError
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN" 
        
        self.llm = LLM(
            model=path,
            trust_remote_code=True,
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.85),
            enforce_eager=True, # 👈 必须开启，防止与训练端的 CUDA 上下文冲突
            max_model_len=self.config.get("max_model_len", 1024), # 建议给到 1024 以上
            tensor_parallel_size=1,
            disable_custom_all_reduce=True # 减少分布式通信干扰
        )

    async def get_samples(self, prompt_text):
        """
        异步采样函数：
        使用 Lock 确保多个训练 Worker 的请求串行执行，彻底解决 AssertionError
        """
        async with self.lock: # 👈 关键点：排队进入，防止底层 Sequence Group 溢出
            try:
                # 打印简单的 Debug 信息
                # print(f"🔍 [vLLM] 正在处理采样，Prompt 长度: {len(prompt_text)}")

                sampling_params = SamplingParams(
                    n=self.group_size,
                    temperature=self.config.get("temperature", 0.8),
                    top_p=0.95,
                    max_tokens=self.config.get("max_new_tokens", 512),
                    stop=["<|endoftext|>", "<|im_end|>", "### Instruction:"]
                )
                
                # 将阻塞的 vLLM 生成操作丢进执行器，避免阻塞 Actor 的事件循环
                loop = asyncio.get_event_loop()
                
                def _run_generate():
                    # 运行前确保显存状态干净
                    return self.llm.generate([prompt_text], sampling_params, use_tqdm=False)

                outputs = await loop.run_in_executor(None, _run_generate)
                
                if not outputs or len(outputs[0].outputs) < self.group_size:
                    print(f"⚠️ [vLLM] 警告：采样结果不足 {self.group_size} 个")
                    return [""] * self.group_size
                    
                return [o.text for o in outputs[0].outputs]
            
            except Exception as e:
                print("\n" + "="*60)
                print("❌ [vLLM Actor] 发生采样异常！")
                print(f"错误信息: {str(e)}")
                # 打印完整堆栈，方便定位底层算子错误
                traceback.print_exc()
                print("="*60 + "\n")
                
                # 返回空字符串列表，防止 Trainer 挂掉
                return [""] * self.group_size

    def update_model(self, new_model_path):
        """
        热更新模型：用于训练若干步后同步权重
        """
        print(f"🔄 [vLLM Actor] 收到模型更新请求，新路径: {new_model_path}")
        
        # 确保新路径权重已经写完
        import time
        for _ in range(5):
            if os.path.exists(os.path.join(new_model_path, "config.json")):
                break
            time.sleep(1)

        try:
            # 1. 显存深度清理
            if hasattr(self, 'llm'):
                # 这种清理在 Ray 环境下对显存回收至关重要
                del self.llm
            
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
            
            # 2. 重新加载引擎
            self._init_engine(new_model_path)
            print(f"🚀 [vLLM Actor] 权重同步成功！")
            return True
        except Exception as e:
            print(f"❌ [vLLM Actor] 更新模型失败: {e}")
            traceback.print_exc()
            return False

    def health_check(self):
        """健康检查"""
        return True