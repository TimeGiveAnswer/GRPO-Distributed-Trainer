import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class GRPOEngine:
    def __init__(self, model, ref_model, optimizer, scheduler, config):
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        self.kl_beta = config.get("beta", 0.04)
        self.clip_eps = config.get("clip_eps", 0.2)
        self.group_size = config.get("group_size", 8)
        self.pad_token_id = config.get("pad_token_id", 0)
        
        # 修正 rank 获取方式
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def _get_per_token_logprobs(self, logits, input_ids):
        # logits 形状: [B, L, Vocab]
        # 我们预测的是下一个 token，所以 logits 取到倒数第二个，target 取从第二个开始
        per_token_logits = logits[:, :-1, :]
        target_ids = input_ids[:, 1:].long() # 🛡️ 强制转为 long 用于 gather 索引
        
        log_probs = F.log_softmax(per_token_logits, dim=-1)
        # 从 Vocab 维度根据 target_ids 提取对应的 logprob
        per_token_logprobs = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        return per_token_logprobs

    def train_step(self, batch, device):
        print(f"🚩 [Rank {self.rank}] --- 进入 train_step ---")
        self.model.train()
        
        # 🛡️ 核心修复 1：强制 input_ids 必须是 long，解决 Embedding 报错
        input_ids = batch["input_ids"].to(device).long() 
        # 🛡️ 核心修复 2：rewards 需要是 float32 用于后续计算 advantage
        rewards = batch["rewards"].to(device).float()
        prompt_len = batch.get("prompt_mask", 0)
        # 🚨 [防御性检查] 检查序列长度是否合法
        batch_size, seq_len = input_ids.shape
        if seq_len <= 1:
            print(f"⚠️ [Rank {self.rank}] 警告: 收到异常序列长度 {seq_len}，跳过此 Step")
            return 0.0, 0.0, 0.0, 0.0
        print(f"📊 [Rank {self.rank}] 输入形状: {input_ids.shape}, dtype: {input_ids.dtype}")

        # --- 1. 跨卡同步 Rewards (GRPO 的核心) ---
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rewards_flat = rewards.view(-1)
            gathered_rewards = [torch.zeros_like(rewards_flat) for _ in range(world_size)]
            dist.all_gather(gathered_rewards, rewards_flat)
            all_rewards = torch.cat(gathered_rewards)
        else:
            all_rewards = rewards

        # --- 2. 计算优势 (Advantages) ---
        # 在整个 Group (所有卡汇总) 维度计算 mean 和 std
        mean = all_rewards.mean()
        std = all_rewards.std() + 1e-8
        advantages = (rewards - mean) / std
        # 展平以便后续逐 token 相乘
        adv_tiled = advantages.view(-1, 1) 
        print(f"📈 [Rank {self.rank}] Advantage计算完毕, Mean: {mean.item():.4f}, Std: {std.item():.4f}")

        # --- 3. 获取 Logprobs ---
        print(f"🔮 [Rank {self.rank}] 开始模型前向 (Policy Model)...")
        # 使用 autocast 提速并节省显存，注意 input_ids 必须在 autocast 外转 long
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = self.model(input_ids)
            all_logprobs = self._get_per_token_logprobs(outputs.logits, input_ids)
            # 及时清理大对象
            del outputs
        
        # --- 4. 获取 Reference Logprobs ---
        if self.ref_model is not None:
            print(f"❄️ [Rank {self.rank}] 开始 Reference 模型前向...")
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    ref_outputs = self.ref_model(input_ids)
                    ref_logprobs = self._get_per_token_logprobs(ref_outputs.logits, input_ids)
                    del ref_outputs
        else:
            ref_logprobs = all_logprobs.detach().clone()

        # --- 5. 构造 Mask (只计算 Answer 部分的 Loss) ---
        # input_ids: [B, L] -> logprobs: [B, L-1]
        seq_len_minus_1 = input_ids.shape[1] - 1
        mask = torch.zeros((input_ids.shape[0], seq_len_minus_1), device=device)
        
        # 只对 Prompt 之后的部分计算 loss
        if prompt_len > 0:
            mask[:, (prompt_len - 1):] = 1.0
        else:
            mask[:, :] = 1.0
            
        # 排除掉 Padding 部分
        padding_mask = (input_ids[:, 1:] != self.pad_token_id).float()
        mask = mask * padding_mask

        # --- 6. 计算 GRPO Loss ---
        # 重要：adv_tiled 乘以 exp(logp - logp_ref)
        ratio = torch.exp(all_logprobs - ref_logprobs)
        surr1 = ratio * adv_tiled
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_tiled
        
        # 策略损失
        policy_loss = -(torch.min(surr1, surr2) * mask).sum() / (mask.sum() + 1e-8)

        # KL 散度约束 (防止模型跑偏)
        # KL(ref || pol) = exp(log_ref - log_pol) - (log_ref - log_pol) - 1
        kl = torch.exp(ref_logprobs - all_logprobs) - (ref_logprobs - all_logprobs) - 1
        kl_loss = self.kl_beta * (kl * mask).sum() / (mask.sum() + 1e-8)

        total_loss = policy_loss + kl_loss
        
        # --- 7. 反向传播 ---
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪，防止大幅波动
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
            
        print(f"📉 [Rank {self.rank}] Step完成 | Loss: {total_loss.item():.4f} | GradNorm: {grad_norm.item():.2f}")

        return total_loss.item(), policy_loss.item(), kl_loss.item(), grad_norm.item()