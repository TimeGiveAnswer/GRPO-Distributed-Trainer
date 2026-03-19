import torch
import re

class GRPOMath:
    @staticmethod
    def get_per_token_logprobs(logits, input_ids):
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_probs = log_probs[:, :-1, :]
        target_ids = input_ids[:, 1:, None]
        return torch.gather(target_log_probs, 2, target_ids).squeeze(-1)

    @staticmethod
    def compute_group_advantages(rewards, group_size):
        rewards = rewards.view(-1, group_size)
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True)
        return ((rewards - mean) / (std + 1e-8)).view(-1)

class GRPORewards:
    @staticmethod
    def check_format(completions):
        """格式奖励：必须包含【】"""
        return torch.tensor([0.2 if "【" in c and "】" in c else 0.0 for c in completions])

    @staticmethod
    def check_accuracy(completions, solution="6"):
        """准确性奖励：内容是否匹配"""
        rewards = []
        for text in completions:
            match = re.search(r"【(.*?)】", text)
            if match and match.group(1).strip() == solution:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return torch.tensor(rewards)