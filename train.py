import os
import json
import torch
import torch.nn as nn
import numpy as np
import tianshou as ts
from torch.utils.tensorboard import SummaryWriter

from bp_env import DotaBPEnv
from agent import BPDrafterNet, BPCriticNet
from mock_eva import MockEvaluator

# ==============================================================================
#  超参数配置
# ==============================================================================
LOG_PATH = "logs/"
MODEL_PATH = "models/"
MAP_PATH = "models/hero_mappings.json"

BATCH_SIZE = 64
BUFFER_SIZE = 20000
LEARNING_RATE = 1e-4
EPOCHS = 100
STEP_PER_EPOCH = 10000
STEP_PER_COLLECT = 2048
REPEAT_PER_COLLECT = 10
NUM_TRAIN_ENVS = 4
NUM_TEST_ENVS = 2

GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
EPS_CLIP = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
#  辅助函数
# ==============================================================================


def _obs_to_state_dict(obs_vector, idx2hero, first_pick, agent_selection):
    """将向量观察转换为字典状态"""
    r_picks_idx = obs_vector[:5]
    d_picks_idx = obs_vector[5:10]
    b_picks_idx = obs_vector[10:20]

    r_picks_orig_id = [idx2hero[int(idx)] for idx in r_picks_idx if idx != 0]
    d_picks_orig_id = [idx2hero[int(idx)] for idx in d_picks_idx if idx != 0]
    b_picks_orig_id = [idx2hero[int(idx)] for idx in b_picks_idx if idx != 0]

    state_dict = {
        "R": r_picks_orig_id,
        "D": d_picks_orig_id,
        "B": b_picks_orig_id,
        "first_pick": 1 if first_pick == 0 else 0,
        "turn": "R" if agent_selection == "player_0" else "D"
    }
    return state_dict

# ==============================================================================
#  主训练函数
# ==============================================================================


def train_agent():
    # 1. 加载映射
    with open(MAP_PATH, 'r') as f:
        mappings = json.load(f)
    hero2idx = {int(k): v for k, v in mappings['hero2idx'].items()}
    idx2hero = {int(k): v for k, v in mappings['idx2hero'].items()}
    H = len(hero2idx)

    evaluator = MockEvaluator(hero2idx=hero2idx, idx2hero=idx2hero)

    # 2. 创建环境
    def env_fn():
        return DotaBPEnv(evaluator, hero2idx, idx2hero)

    train_envs = ts.env.DummyVectorEnv([env_fn for _ in range(NUM_TRAIN_ENVS)])
    test_envs = ts.env.DummyVectorEnv([env_fn for _ in range(NUM_TEST_ENVS)])

    # 3. 获取空间信息
    env = env_fn()
    observation_space = env.observation_space('player_0')
    action_space = env.action_space('player_0')
    obs_dim = observation_space['observation'].shape[0]  # 现在应该是 26

    print(f"观察空间维度: {obs_dim}")
    print(f"动作空间: {action_space}")
    print(f"BP 总步数: {len(env.bp_sequence)}")

    # 4. 创建网络
    # Actor 网络（输出动作分布）
    actor = BPDrafterNet(
        obs_dim=obs_dim,
        action_shape=action_space,  # 传入 action_space 对象
        H=H,
        emb_dim=128
    ).to(DEVICE)

    # Critic 网络（输出状态价值）
    critic = BPCriticNet(
        obs_dim=obs_dim,
        emb_dim=128
    ).to(DEVICE)

    # 5. 创建优化器
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=LEARNING_RATE
    )

    # 6. 创建 PPO 策略
    try:
        # 尝试新版 API（不传 action_space）
        student_policy = ts.policy.PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            action_space=action_space,
            action_scaling=False,
            # PPO 相关超参数
            discount_factor=GAMMA,
            gae_lambda=GAE_LAMBDA,
            vf_coef=VALUE_COEF,
            ent_coef=ENTROPY_COEF,
            eps_clip=EPS_CLIP,
            # 其他 Tianshou 参数
            reward_normalization=False,
            dual_clip=None,
            value_clip=False,
            max_grad_norm=0.5,
        ).to(DEVICE)
    except TypeError:
        # 如果失败，尝试旧版 API
        student_policy = ts.policy.PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            action_space=action_space,
            discount_factor=GAMMA,
            max_grad_norm=0.5,
            vf_coef=VALUE_COEF,
            ent_coef=ENTROPY_COEF,
            gae_lambda=GAE_LAMBDA,
            reward_normalization=False,
            dual_clip=None,
            value_clip=False,
            eps_clip=EPS_CLIP,
        ).to(DEVICE)

    # 7. 创建对手策略（规则策略）
    class OpponentPolicy(ts.policy.BasePolicy):
        """基于规则的对手策略"""

        def __init__(self, evaluator, hero2idx, idx2hero):
            super().__init__(action_space=action_space)
            self.evaluator = evaluator
            self.hero2idx = hero2idx
            self.idx2hero = idx2hero

        def forward(self, batch, state=None, **kwargs):
            """根据评估器生成动作"""
            acts = []
            for i in range(len(batch)):
                obs = batch.obs[i]
                info = batch.info[i]

                # 获取环境信息
                env_info = info.get('env', {})
                first_pick = getattr(env_info, 'first_pick', 0)
                agent_sel = getattr(env_info, 'agent_selection', 'player_1')

                # 转换为字典状态
                state_dict = _obs_to_state_dict(
                    obs['observation'],
                    self.idx2hero,
                    first_pick,
                    agent_sel
                )

                # 获取动作
                action_id = self.evaluator.get_teacher_action(state_dict)
                action_idx = self.hero2idx[action_id]
                acts.append(action_idx)

            return ts.data.Batch(act=np.array(acts), state=state)

        def learn(self, batch, **kwargs):
            """对手不学习"""
            return {}

    opponent_policy = OpponentPolicy(evaluator, hero2idx, idx2hero)

    # 8. 创建多智能体策略管理器
    env_instance = env_fn()
    env_instance.agents = ["player_0", "player_1"]
    env_instance.action_space = env_instance.action_space('player_0')
    env_instance.observation_space = env_instance.observation_space('player_0')
    env_instance.agent_idx = 0

    policies = ts.policy.MultiAgentPolicyManager(
        policies=[student_policy, opponent_policy],
        env=env_instance

    )

    # 9. 创建数据收集器
    train_collector = ts.data.Collector(
        policies,
        train_envs,
        ts.data.VectorReplayBuffer(BUFFER_SIZE, len(train_envs)),
        exploration_noise=True
    )

    test_collector = ts.data.Collector(
        policies,
        test_envs,
        exploration_noise=True
    )

    # 10. 日志记录
    writer = SummaryWriter(LOG_PATH)
    logger = ts.utils.TensorboardLogger(writer)

    # 11. 定义保存函数
    def save_best_fn(policy):
        # 保存学生策略
        student = policy.policies[0]  # 第一个是学生策略
        torch.save(
            student.state_dict(),
            os.path.join(MODEL_PATH, "student_agent.pth")
        )
        print("保存最佳模型")

    # 12. 创建训练器
    result = ts.trainer.OnpolicyTrainer(
        policy=policies,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=EPOCHS,
        step_per_epoch=STEP_PER_EPOCH,
        repeat_per_collect=REPEAT_PER_COLLECT,
        episode_per_test=10,
        batch_size=BATCH_SIZE,
        step_per_collect=STEP_PER_COLLECT,
        stop_fn=lambda mean_rewards: mean_rewards >= 0.8,  # BP 胜率 > 80%
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False,
    )

    print("训练完成！")
    print(f"最佳奖励: {result.best_reward}")


if __name__ == "__main__":
    os.makedirs(LOG_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    train_agent()
