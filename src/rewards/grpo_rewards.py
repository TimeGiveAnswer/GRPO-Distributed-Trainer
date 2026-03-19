import torch
import re

def extract_solution(text):
    """从模型输出中提取 \boxed{...} 里的内容"""
    pattern = r"\\boxed\{(.*?)\}"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""

class GRPORewarder:
    def __init__(self, config):
        self.config = config

    def compute_reward(self, completion, ground_truth):
        """
        计算单条样本的原始奖励值
        """
        reward = 0.0
        
        # 1. 准确率奖励 (Accuracy)
        predicted_ans = extract_solution(completion)
        if str(predicted_ans) == str(ground_truth):
            reward += 1.0
            
        # 2. 格式奖励 (Format)
        # 检查是否包含 \boxed{} 和思考过程 <think>
        if "\\boxed{" in completion:
            reward += 0.1
        if "<think>" in completion and "</think>" in completion:
            reward += 0.1
            
        # 3. 长度惩罚 (防止模型刷屏，可选)
        if len(completion) > 1000:
            reward -= 0.1
            
        return reward

    def get_group_rewards(self, completions, ground_truth):
        """
        处理一组生成结果，返回 Tensor
        """
        rewards = [self.compute_reward(c, ground_truth) for c in completions]
        return torch.tensor(rewards, dtype=torch.float32)