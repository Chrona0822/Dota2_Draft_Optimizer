import torch
import torch.nn as nn


class BPDrafterNet(nn.Module):
    """
    BP Drafter 的 Actor 网络
    用于 PPO 策略的动作选择
    """

    def __init__(self, obs_dim, action_shape, H, emb_dim=128):
        super().__init__()
        self.H = H

        # 处理 action_shape（可能是 Space 对象或整数）
        if hasattr(action_shape, 'n'):
            self.action_dim = action_shape.n
        else:
            self.action_dim = int(action_shape)

        # 特征提取网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Actor 头：输出动作 logits
        self.actor_head = nn.Linear(256, self.action_dim)

    def forward(self, obs, state=None, info={}):
        """
        Args:
            obs: 观察值，格式为 dict:
                - 'observation': [batch, obs_dim] 观察向量
                - 'action_mask': [batch, action_dim] 可用动作掩码
            state: RNN 隐状态（不使用，返回 None）
            info: 额外信息

        Returns:
            logits: [batch, action_dim] 动作 logits（已应用 mask）
            state: None
        """
        # 处理输入格式
        if isinstance(obs, dict):
            x = obs['observation']
            mask = obs.get('action_mask', None)
        else:
            # 如果是纯 tensor，假设没有 mask
            x = obs
            mask = None

        # 确保是 2D [batch, obs_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 特征提取
        features = self.net(x)

        # 输出 logits
        logits = self.actor_head(features)

        # 【关键】应用 action mask
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            # 将不可用动作的 logit 设为负无穷
            logits = logits.masked_fill(~mask.bool(), float('-inf'))

        return logits, state


class BPCriticNet(nn.Module):
    """
    BP Drafter 的 Critic 网络
    用于 PPO 策略的价值估计
    """

    def __init__(self, obs_dim, emb_dim=128):
        super().__init__()

        # 特征提取网络
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Critic 头：输出状态价值（标量）
        self.critic_head = nn.Linear(256, 1)

    def forward(self, obs, state=None, info={}):
        """
        Args:
            obs: 观察值，格式为 dict 或 tensor
            state: RNN 隐状态（不使用）
            info: 额外信息

        Returns:
            value: [batch, 1] 状态价值
            state: None
        """
        # 处理输入格式
        if isinstance(obs, dict):
            x = obs['observation']
        else:
            x = obs

        # 确保是 2D
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # 特征提取
        features = self.net(x)

        # 输出价值
        value = self.critic_head(features)

        return value, state


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("测试 BPDrafterNet 和 BPCriticNet...")

    # 参数
    obs_dim = 145  # 观察维度
    action_dim = 125  # 动作数量
    H = 124  # 英雄总数
    batch_size = 32

    # 创建网络
    actor = BPDrafterNet(
        obs_dim=obs_dim, action_shape=action_dim, H=H, emb_dim=128)
    critic = BPCriticNet(obs_dim=obs_dim, emb_dim=128)

    print(f"✅ Actor 参数: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"✅ Critic 参数: {sum(p.numel() for p in critic.parameters()):,}")

    # 测试 1: 字典输入（带 mask）
    obs_dict = {
        'observation': torch.randn(batch_size, obs_dim),
        'action_mask': torch.randint(0, 2, (batch_size, action_dim)).bool()
    }

    logits, _ = actor(obs_dict)
    value, _ = critic(obs_dict)

    print(f"\n✅ 测试 1: 批量输入（带 mask）")
    print(f"  Actor 输出: {logits.shape}")
    print(f"  Critic 输出: {value.shape}")

    # 验证 mask 是否生效
    masked_pos = ~obs_dict['action_mask'][0]
    if torch.isinf(logits[0][masked_pos]).all():
        print(f"  ✅ Action mask 正确应用")
    else:
        print(f"  ❌ Action mask 未生效！")

    # 测试 2: 单样本输入
    single_obs = {
        'observation': torch.randn(obs_dim),
        'action_mask': torch.randint(0, 2, (action_dim,)).bool()
    }

    logits, _ = actor(single_obs)
    value, _ = critic(single_obs)

    print(f"\n✅ 测试 2: 单样本输入")
    print(f"  Actor 输出: {logits.shape}")
    print(f"  Critic 输出: {value.shape}")

    # 测试 3: PPO 兼容性
    from torch.distributions import Categorical

    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)

    print(f"\n✅ 测试 3: PPO 采样")
    print(f"  采样动作: {action}")
    print(f"  Log prob: {log_prob}")

    print("\n" + "="*50)
    print("🎉 所有测试通过！")
    print("="*50)
