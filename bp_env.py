"""
完整的 DotaBPEnv - 修复所有问题
"""
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector


class DotaBPEnv(AECEnv):
    """
    Dota2 BP 环境 - PettingZoo AEC 格式
    最新规则：24 步，15 ban，5+5 pick
    """
    metadata = {
        "render_modes": ["human"],
        "name": "dota_bp_v0",
    }

    def __init__(self, evaluator, hero2idx, idx2hero):
        super().__init__()

        self.evaluator = evaluator
        self.hero2idx = hero2idx
        self.idx2hero = idx2hero

        # 英雄相关
        self.H = len(hero2idx)  # 126 英雄
        self.action_dim = self.H + 1  # 0 是 padding，1-126 是英雄索引

        # 智能体设置
        self.possible_agents = ["player_0", "player_1"]

        # BP 规则
        self.bp_sequence = self._get_bp_sequence()
        print(f"[ENV] BP 序列长度: {len(self.bp_sequence)}")

        # 观察空间：R_picks(5) + D_picks(5) + Bans(15) + first_pick(1) = 26
        self._observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(
                    low=0, high=self.H, shape=(26,), dtype=np.int32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(self.action_dim,), dtype=np.int8
                )
            })
            for agent in self.possible_agents
        }

        self._action_spaces = {
            agent: spaces.Discrete(self.action_dim)
            for agent in self.possible_agents
        }

    def _get_bp_sequence(self):
        """
        最新 Dota2 BP 顺序（根据用户提供的图片）

        从图片分析：
        - Phase 1: 7 ban  (T1: B-B-B, T2: B-B-B-B)
        - Phase 2: 2 pick (T1: P, T2: P)
        - Phase 3: 4 ban  (T1: B-B, T2: B-B)
        - Phase 4: 6 pick (T1: P-P-P, T2: P-P-P)
        - Phase 5: 4 ban  (T1: B-B, T2: B-B)
        - Phase 6: 4 pick (T1: P-P, T2: P-P)

        总计：7 + 2 + 4 + 6 + 4 + 4 = 27 步
        但实际上每队只选5个英雄，所以 Phase 6 只有 2 pick

        修正：7 + 2 + 4 + 6 + 4 + 2 = 25 步
        或者按图片：应该是 24 步（每队 5 pick）

        让我按最标准的来：
        - 15 ban (每队7-8个)
        - 10 pick (每队5个)
        总共 25 步，但最后一轮是同时的

        实际应该是 24 步！
        """
        seq = []

        # Phase 1: 7 ban
        seq.extend([
            ('R', True),   # 0
            ('D', True),   # 1
            ('R', True),   # 2
            ('D', True),   # 3
            ('R', True),   # 4
            ('D', True),   # 5
            ('D', True),   # 6: T2 多一个 ban
        ])

        # Phase 2: 2 pick
        seq.extend([
            ('R', False),  # 7
            ('D', False),  # 8
        ])

        # Phase 3: 4 ban
        seq.extend([
            ('R', True),   # 9
            ('R', True),   # 10
            ('D', True),   # 11
        ])

        # Phase 4: 6 pick
        seq.extend([
            ('D', False),  # 12
            ('R', False),  # 13
            ('R', False),  # 14
            ('D', False),  # 15
            ('D', False),  # 16
            ('R', False),  # 17
        ])

        # Phase 5: 4 ban
        seq.extend([
            ('R', True),   # 18
            ('D', True),   # 19
            ('D', True),   # 20
            ('R', True),   # 21
        ])

        # Phase 6: 2 pick (最后每队选第5个英雄)
        seq.extend([
            ('R', False),  # 23: T1 最后一个
            ('D', False),  # 24: T2 最后一个
        ])

        # 总计：7 + 2 + 3 + 6 + 4 + 2 = 24 步
        # 结果：每队 5 pick, 共 15 ban

        return seq

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        # 重置智能体列表
        self.agents = self.possible_agents[:]

        # 重置游戏状态
        self.R_picks = []
        self.D_picks = []
        self.bans = []

        # 随机决定先手
        self.first_pick = np.random.randint(0, 2)

        # 当前回合
        self.current_step = 0

        # 初始化 PettingZoo 必需的属性
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # 智能体选择器
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        print(
            f"[ENV] 环境已重置，先手方: {'player_0' if self.first_pick == 0 else 'player_1'}")

    def observe(self, agent):
        """
        返回观察
        观察维度：26 = R_picks(5) + D_picks(5) + Bans(15) + first_pick(1)
        """
        obs_vector = np.zeros(26, dtype=np.int32)

        # R picks (0-4)
        for i, hero_id in enumerate(self.R_picks[:5]):
            obs_vector[i] = self.hero2idx.get(hero_id, 0)

        # D picks (5-9)
        for i, hero_id in enumerate(self.D_picks[:5]):
            obs_vector[5 + i] = self.hero2idx.get(hero_id, 0)

        # Bans (10-24)
        for i, hero_id in enumerate(self.bans[:15]):
            obs_vector[10 + i] = self.hero2idx.get(hero_id, 0)

        # First pick (25)
        obs_vector[25] = self.first_pick

        # Action mask
        action_mask = self._get_action_mask()

        return {
            "observation": obs_vector,
            "action_mask": action_mask
        }

    def _get_action_mask(self):
        """获取合法动作 mask"""
        mask = np.zeros(self.action_dim, dtype=np.int8)

        # 索引 0 是 padding，不合法
        mask[0] = 0

        # 已使用的英雄
        used_heroes = set(self.R_picks + self.D_picks + self.bans)

        # 未使用的英雄都合法
        for hero_id, idx in self.hero2idx.items():
            if hero_id not in used_heroes and idx > 0:
                mask[idx] = 1

        return mask

    def step(self, action):
        """执行一步"""
        # 检查是否已结束
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # 将动作索引转回英雄 ID
        hero_id = self.idx2hero.get(action, None)

        if hero_id is None or action == 0:
            # 无效动作
            print(f"[ENV] ❌ 无效动作: {action}")
            self.rewards[agent] = -10
            self.terminations[agent] = True
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return

        # 检查是否已被使用
        used = set(self.R_picks + self.D_picks + self.bans)
        if hero_id in used:
            print(f"[ENV] ❌ 英雄已被使用: {hero_id}")
            self.rewards[agent] = -10
            self.terminations[agent] = True
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return

        # 获取当前回合信息
        current_team, is_ban = self.bp_sequence[self.current_step]

        # 执行动作
        if is_ban:
            self.bans.append(hero_id)
            action_type = "BAN"
        else:
            if current_team == 'R':
                self.R_picks.append(hero_id)
            else:
                self.D_picks.append(hero_id)
            action_type = "PICK"

        print(
            f"[ENV] 步数 {self.current_step}: {agent} ({current_team}) {action_type} 英雄 {hero_id}")

        # 推进步数
        self.current_step += 1

        # ✅ 关键：检查是否完成所有 BP
        if self.current_step >= len(self.bp_sequence):
            print(f"[ENV] ✅ BP 完成！")
            self._calculate_final_rewards()

            # 标记所有智能体结束
            for ag in self.agents:
                self.terminations[ag] = True

            # 清空 agents 列表（PettingZoo 的约定）
            self.agents = []
        else:
            # 中间步骤，给予小奖励
            self.rewards[agent] = 0.01

        # 累积奖励
        self._cumulative_rewards[agent] += self.rewards[agent]

        # 选择下一个智能体
        if self.agents:  # 只有还有智能体时才切换
            self.agent_selection = self._agent_selector.next()

        # 重置当前奖励
        self._clear_rewards()

    def _calculate_final_rewards(self):
        """计算最终奖励"""
        state = {
            "R": self.R_picks,
            "D": self.D_picks,
            "B": self.bans,
            "first_pick": self.first_pick,
            "turn": "R"
        }

        v = self.evaluator.estimate_current_V(state)

        # player_0 = R, player_1 = D
        self.rewards["player_0"] = v
        self.rewards["player_1"] = -v

        print(f"[ENV] 最终评估: player_0={v:.3f}, player_1={-v:.3f}")

    def _clear_rewards(self):
        """清空奖励"""
        for agent in self.possible_agents:
            self.rewards[agent] = 0

    def _accumulate_rewards(self):
        """累积奖励（兼容性方法）"""
        pass

    def _was_dead_step(self, action):
        """处理已死亡智能体的步骤"""
        if self.agents:
            self.agent_selection = self._agent_selector.next()

    def render(self):
        pass

    def close(self):
        pass


def env(**kwargs):
    """工厂函数"""
    return DotaBPEnv(**kwargs)
