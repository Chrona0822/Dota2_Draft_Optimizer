"""
调试脚本：检查环境的观察和动作空间
"""
import json
import numpy as np
from bp_env import DotaBPEnv
from mock_eva import MockEvaluator


def debug_env():
    # 1. 加载映射
    MAP_PATH = "models/hero_mappings.json"
    with open(MAP_PATH, 'r') as f:
        mappings = json.load(f)
    hero2idx = {int(k): v for k, v in mappings['hero2idx'].items()}
    idx2hero = {int(k): v for k, v in mappings['idx2hero'].items()}

    print(f"英雄总数: {len(hero2idx)}")
    print(f"索引范围: 0-{max(hero2idx.values())}")

    # 2. 创建环境
    evaluator = MockEvaluator(hero2idx=hero2idx, idx2hero=idx2hero)
    env = DotaBPEnv(evaluator, hero2idx, idx2hero)

    # 3. 检查空间
    print("\n" + "="*60)
    print("环境空间信息")
    print("="*60)

    for agent_name in env.possible_agents:
        obs_space = env.observation_space(agent_name)
        act_space = env.action_space(agent_name)

        print(f"\nAgent: {agent_name}")
        print(f"  观察空间类型: {type(obs_space)}")
        print(f"  观察空间: {obs_space}")

        if hasattr(obs_space, 'spaces'):
            # Dict 空间
            for key, space in obs_space.spaces.items():
                print(f"    - {key}: {space}")

        print(f"  动作空间: {act_space}")
        print(f"  动作数量: {act_space.n}")

    # 4. 重置环境并检查观察
    print("\n" + "="*60)
    print("重置环境并检查第一个观察")
    print("="*60)

    env.reset()

    # 获取第一个智能体的观察
    agent = env.agent_selection
    obs, reward, termination, truncation, info = env.last()

    print(f"\n当前智能体: {agent}")
    print(f"观察类型: {type(obs)}")

    if isinstance(obs, dict):
        print(f"观察是字典，包含以下键: {list(obs.keys())}")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  - {key}: shape={value.shape}, dtype={value.dtype}")
                print(f"    前10个值: {value[:10]}")
            else:
                print(f"  - {key}: {value}")
    else:
        print(f"观察形状: {obs.shape}")
        print(f"观察前10个值: {obs[:10]}")

    print(f"\nReward: {reward}")
    print(f"Termination: {termination}")
    print(f"Truncation: {truncation}")
    print(f"Info: {info}")

    # 5. 测试一步交互
    print("\n" + "="*60)
    print("测试动作执行")
    print("="*60)

    # 获取 action_mask
    if isinstance(obs, dict) and 'action_mask' in obs:
        mask = obs['action_mask']
        legal_actions = np.where(mask)[0]
        print(f"合法动作数量: {len(legal_actions)}")
        print(f"前10个合法动作: {legal_actions[:10]}")

        # 选择第一个合法动作
        action = legal_actions[0]
    else:
        print("警告：没有 action_mask！")
        action = 1

    print(f"执行动作: {action}")

    env.step(action)

    # 获取下一个观察
    agent = env.agent_selection
    obs, reward, termination, truncation, info = env.last()

    print(f"\n执行后:")
    print(f"  当前智能体: {agent}")
    print(f"  Reward: {reward}")
    print(f"  Termination: {termination}")

    if isinstance(obs, dict) and 'action_mask' in obs:
        mask = obs['action_mask']
        legal_actions = np.where(mask)[0]
        print(f"  剩余合法动作: {len(legal_actions)}")

    # 6. 完整走完一局
    print("\n" + "="*60)
    print("完整运行一局")
    print("="*60)

    env.reset()
    step = 0
    max_steps = 50  # 防止死循环

    # ✅ 修复：当 env.agents 不为空时继续
    while env.agents and step < max_steps:
        agent = env.agent_selection
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
        else:
            # 随机选择合法动作
            if isinstance(obs, dict) and 'action_mask' in obs:
                mask = obs['action_mask']
                legal_actions = np.where(mask)[0]

                # ✅ 过滤掉 action 0 (padding)
                legal_actions = legal_actions[legal_actions > 0]

                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
                else:
                    print(f"警告：没有合法动作！")
                    break
            else:
                action = env.action_space(agent).sample()

            print(
                f"  步数 {step}: {agent} 选择动作 {action} (英雄ID: {idx2hero.get(action, '未知')})")
            env.step(action)

        step += 1

    print(f"\n游戏结束！总步数: {step}")

    # 显示最终的 BP 结果
    print("\n" + "="*60)
    print("最终 BP 结果")
    print("="*60)
    print(f"\nTeam 1 (先手={env.first_pick == 0}):")
    print(f"  Picks: {[idx2hero[hero2idx[h]] for h in env.R_picks]}")
    print(f"  总共 {len(env.R_picks)} 个英雄")

    print(f"\nTeam 2 (后手={env.first_pick == 1}):")
    print(f"  Picks: {[idx2hero[hero2idx[h]] for h in env.D_picks]}")
    print(f"  总共 {len(env.D_picks)} 个英雄")

    print(f"\n总 Bans: {len(env.bans)} 个")
    print(f"  前10个: {[idx2hero[hero2idx[h]] for h in env.bans[:10]]}")

    # 检查最终奖励
    print("\n" + "="*60)
    print("最终奖励")
    print("="*60)
    for agent in env.possible_agents:
        final_reward = env.rewards.get(agent, 0)
        cumulative = env._cumulative_rewards.get(agent, 0)
        print(f"  {agent}:")
        print(f"    - 最终奖励: {final_reward:.3f}")
        print(f"    - 累积奖励: {cumulative:.3f}")

    # 验证
    print("\n" + "="*60)
    print("验证")
    print("="*60)
    print(f"✅ 应该有 24 步: 实际 {step} 步")
    print(f"✅ 应该有 5+5 个 pick: 实际 {len(env.R_picks)}+{len(env.D_picks)} 个")
    print(f"✅ 应该有 14 个 ban: 实际 {len(env.bans)} 个")


if __name__ == "__main__":
    debug_env()
