import numpy as np
import random


class MockEvaluator:
    """
    一个假的评估器，只用于测试。
    改进版：会返回合法的动作
    """

    def __init__(self, hero2idx, idx2hero):
        print("--- HIHI  你正在使用 MockEvaluator (假评估器) ---")
        self.hero2idx = hero2idx
        self.idx2hero = idx2hero
        self.ALL_HEROES = list(self.hero2idx.keys())
        self.H = len(self.hero2idx)

    def estimate_current_V(self, state, N=16, beta=0.0):
        """
        假装估计当前状态的价值
        返回随机值 [0, 1]
        """
        return random.uniform(0.3, 0.7)

    def legal_pool(self, state):
        """
        获取当前合法的英雄池

        Args:
            state: 状态字典，包含 R, D, B

        Returns:
            list: 合法英雄的原始 ID 列表
        """
        # 收集所有已使用的英雄
        used = set(state.get("R", [])) | set(
            state.get("D", [])) | set(state.get("B", []))

        # 返回未使用的英雄
        legal = [h for h in self.ALL_HEROES if h not in used]

        return legal

    def get_teacher_action(self, state):
        """
        模拟"老师"策略，返回一个合法的动作

        Args:
            state: 状态字典

        Returns:
            int: 英雄的原始 ID
        """
        # 获取合法动作
        legal = self.legal_pool(state)

        if not legal:
            # 如果没有合法动作（不应该发生），返回随机英雄
            print("警告：没有合法动作！")
            return random.choice(self.ALL_HEROES)

        # 随机选择一个合法英雄
        action_id = random.choice(legal)

        return action_id


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 创建假的映射
    hero2idx = {i: i for i in range(1, 125)}
    idx2hero = {i: i for i in range(1, 125)}

    evaluator = MockEvaluator(hero2idx, idx2hero)

    # 测试 1: 空状态
    state1 = {"R": [], "D": [], "B": []}
    legal1 = evaluator.legal_pool(state1)
    print(f"测试 1 - 空状态")
    print(f"  合法英雄数: {len(legal1)}")
    print(f"  应该等于: {len(hero2idx)}")
    assert len(legal1) == len(hero2idx), "空状态应该所有英雄都合法"

    # 测试 2: 部分英雄已选
    state2 = {
        "R": [1, 2, 3],
        "D": [4, 5],
        "B": [6, 7, 8, 9, 10]
    }
    legal2 = evaluator.legal_pool(state2)
    print(f"\n测试 2 - 部分选择")
    print(f"  已使用英雄: {len(state2['R']) + len(state2['D']) + len(state2['B'])}")
    print(f"  合法英雄数: {len(legal2)}")
    print(f"  应该等于: {len(hero2idx) - 10}")
    assert len(legal2) == len(hero2idx) - 10, "合法英雄数量错误"

    # 测试 3: 获取老师动作
    for i in range(5):
        action = evaluator.get_teacher_action(state2)
        print(f"\n测试 3.{i+1} - 老师动作: {action}")
        assert action in legal2, f"动作 {action} 不合法！"
        assert action not in state2["R"] + \
            state2["D"] + state2["B"], "选择了已用英雄！"

    # 测试 4: 价值估计
    v = evaluator.estimate_current_V(state2)
    print(f"\n测试 4 - 价值估计: {v:.3f}")
    assert 0 <= v <= 1, "价值应该在 [0, 1] 范围内"

    print("\n" + "="*50)
    print("🎉 MockEvaluator 所有测试通过！")
    print("="*50)
