import torch
import numpy as np
import json
import torch.nn as nn
import random
import math

# 把 data_preprocessing.py 里的 MCTS 相关函数 (estimate_V, rollout_complete...) 复制到这里
# 把 DeepSetsEvaluatorFP 模型的定义也复制到这里


class SetEncoder(nn.Module):
    def __init__(self, n_heroes, d=128, p_drop=0.2):
        super().__init__()
        self.emb = nn.Embedding(n_heroes + 1, d, padding_idx=0)  # 0=PAD
        self.proj = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(p_drop), nn.LayerNorm(d)
        )

    def forward(self, ids):  # ids:(B,5)
        x = self.emb(ids)  # (B,5,d)
        s = torch.cat([x.mean(1), x.max(1).values], dim=-1)
        return self.proj(s)  # (B,d)


class DeepSetsEvaluatorFP(nn.Module):
    """
    first_pick 嵌入（2 类）→ 融合到头部；输出反对称：
      logit = f(R,D,fp) - f(D,R,1-fp)
    """

    def __init__(self, n_heroes, d=128, p_drop=0.2):
        super().__init__()
        self.enc = SetEncoder(n_heroes, d, p_drop)
        self.fp_emb = nn.Embedding(2, d // 8)
        self.fp_proj = nn.Sequential(
            nn.Linear(d // 8, d // 4), nn.ReLU(), nn.Dropout(p_drop)
        )
        self.head = nn.Sequential(
            nn.Linear(4 * d + d // 4, 2 * d),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d, 1),
        )

    def fuse(self, SR, SD, Efp):
        feat = torch.cat([SR, SD, SR * SD, torch.abs(SR - SD), Efp], dim=-1)
        return self.head(feat).squeeze(-1)  # (B,)

    def forward(self, R, D, fp):
        SR = self.enc(R)
        SD = self.enc(D)  # (B,d)
        Efp = self.fp_proj(self.fp_emb(fp))  # (B,d/4)
        f_RD = self.fuse(SR, SD, Efp)
        f_DR = self.fuse(SD, SR, Efp)  # 注意：不翻转 fp，这里反对称在减法
        logit = f_RD - f_DR
        prob = torch.sigmoid(logit)
        return prob, logit


def legal_pool(state):  # MCTS get legal action pool
    used = set(state["R"]) | set(state["D"]) | set(state["B"])
    return [h for h in ALL_HEROES if h not in used]


def score_hero_weighted(h, R, D, alpha=0.6):  # MCTS sample weighted hero
    if E64_torch is None:
        return 0.0
    if not R and not D:
        return 0.0
    Eh = E64_torch[hero2idx[h]]  # (k,)

    def mean_vec(lst):
        if not lst:
            return torch.zeros_like(Eh)
        return E64_torch[torch.tensor([hero2idx[x] for x in lst])].mean(dim=0)

    sR, sD = mean_vec(R), mean_vec(D)
    return float(torch.dot(Eh, sR) - alpha * torch.dot(Eh, sD))


def sample_one(pool, R, D, policy="weighted"):
    if not pool:
        return None
    if policy == "uniform" or E64_torch is None:
        return random.choice(pool)
    # weighted by exp(score/τ)
    scores = [score_hero_weighted(h, R, D) for h in pool]
    mx = max(scores)
    # 数值稳定
    w = [math.exp(s - mx) for s in scores]
    tot = sum(w)
    probs = [x / tot for x in w] if tot > 0 else [1 / len(pool)] * len(pool)
    r = random.random()
    acc = 0.0
    for h, p in zip(pool, probs):
        acc += p
        if r <= acc:
            return h
    return pool[-1]


def rollout_complete(state, policy="weighted"):
    R = state["R"][:]
    D = state["D"][:]
    B = set(state["B"])
    side = state["turn"]
    while len(R) < 5 or len(D) < 5:
        pool = [h for h in ALL_HEROES if h not in B and h not in R and h not in D]
        if len(R) < 5 and (side == "R" or len(D) == 5):
            h = sample_one(pool, R, D, policy=policy)
            R.append(h)
            side = "D"
        elif len(D) < 5:
            h = sample_one(pool, D, R, policy=policy)
            D.append(h)
            side = "R"
    return R, D


def apply_pick(state, hero):
    s = {k: (v[:] if isinstance(v, list) else v) for k, v in state.items()}
    if s["turn"] == "R":
        s["R"].append(hero)
        s["turn"] = "D"
    else:
        s["D"].append(hero)
        s["turn"] = "R"
    return s


def apply_ban(state, hero):
    s = {k: (v[:] if isinstance(v, list) else v) for k, v in state.items()}
    s["B"] = list(set(s["B"]) | {hero})
    # ban 后轮转规则按你定义；默认不切换
    return s


def candidate_pool(state, kind="pick", topk=None, policy="weighted"):
    pool = legal_pool(state)
    if topk is None or len(pool) <= topk or E64_torch is None or policy == "uniform":
        return pool
    # 用权重挑 topk 候选
    scores = [(h, score_hero_weighted(h, state["R"], state["D"]))
              for h in pool]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [h for h, _ in scores[:topk]]


def best_action(
    state, evaluator, kind="pick", N=64, beta=0.2, topk=24, policy="weighted"
):
    pool = candidate_pool(state, kind=kind, topk=topk, policy=policy)
    best = None
    best_score = -1.0
    memo = []
    for h in pool:
        v_adj, mean, std = (score_pick if kind == "pick" else score_ban)(
            state, h, evaluator, N=N, beta=beta, policy=policy
        )
        memo.append((h, v_adj, mean, std))
        if v_adj > best_score:
            best = (h, v_adj, mean, std)
            best_score = v_adj
    memo.sort(key=lambda x: x[1], reverse=True)
    return best, memo  # 返回最优 + 全部候选得分，便于调试/可视化


def minimax_best_action(
    state,
    evaluator,
    kind="pick",  # "pick" 或 "ban"
    N_outer=64,  # 你动作后（或ban后无对手动作）评估用的采样数
    N_inner=32,  # 对手回应那一层的采样数
    beta=0.2,  # 风险折扣系数（mean - beta*std）
    topk_self=24,  # 我方候选裁剪
    topk_opp=12,  # 对手候选裁剪
    policy="weighted",  # 补完时采样策略
    opp_after_ban=None,  # None / "pick" / "ban"
):
    """
    返回：(best_tuple, memo)
    best_tuple = (self_h, score, mean, std, opp_reply)  # opp_reply 可能为 None
    memo: [(h, score, mean, std, opp_reply), ...] 按得分降序
    """
    pool_self = candidate_pool(state, topk=topk_self, policy=policy)
    results = []

    for h in pool_self:
        # 我方动作后的状态
        s1 = apply_pick(state, h) if kind == "pick" else apply_ban(state, h)

        if kind == "pick":
            # 对手下一手：枚举其候选，取“最坏回应”
            pool_opp = candidate_pool(s1, topk=topk_opp, policy=policy)
            if len(s1["R"]) == 5 or len(s1["D"]) == 5 or not pool_opp:
                # 已满或无可选——直接评估
                v_adj, mean, std = estimate_V(
                    s1, N=N_outer, evaluator=evaluator, beta=beta, policy=policy
                )
                results.append((h, v_adj, mean, std, None))
            else:
                worst = +1e9
                worst_tuple = None
                for ho in pool_opp:
                    s2 = apply_pick(s1, ho)  # 对手回应“选”
                    v_adj, mean, std = estimate_V(
                        s2, N=N_inner, evaluator=evaluator, beta=beta, policy=policy
                    )
                    if v_adj < worst:
                        worst = v_adj
                        worst_tuple = (h, v_adj, mean, std, ho)
                results.append(worst_tuple)

        else:  # kind == "ban"
            if opp_after_ban is None:
                # 与之前逻辑一致：ban 后不切换不响应，直接评估
                v_adj, mean, std = estimate_V(
                    s1, N=N_outer, evaluator=evaluator, beta=beta, policy=policy
                )
                results.append((h, v_adj, mean, std, None))
            elif opp_after_ban == "pick":
                # 让对手立刻“选”一手作为回应（需切换回合）
                s1_pick = flip_turn(s1)  # ban 后强制轮到对手
                pool_opp = candidate_pool(
                    s1_pick, topk=topk_opp, policy=policy)
                if not pool_opp:
                    v_adj, mean, std = estimate_V(
                        s1_pick,
                        N=N_outer,
                        evaluator=evaluator,
                        beta=beta,
                        policy=policy,
                    )
                    results.append((h, v_adj, mean, std, None))
                else:
                    worst = +1e9
                    worst_tuple = None
                    for ho in pool_opp:
                        s2 = apply_pick(s1_pick, ho)
                        v_adj, mean, std = estimate_V(
                            s2, N=N_inner, evaluator=evaluator, beta=beta, policy=policy
                        )
                        if v_adj < worst:
                            worst = v_adj
                            worst_tuple = (h, v_adj, mean, std, ho)
                    results.append(worst_tuple)
            elif opp_after_ban == "ban":
                # 让对手立刻“ban”一手（竞赛有这种phase）
                s1_ban = flip_turn(s1)
                pool_opp = candidate_pool(s1_ban, topk=topk_opp, policy=policy)
                if not pool_opp:
                    v_adj, mean, std = estimate_V(
                        s1_ban, N=N_outer, evaluator=evaluator, beta=beta, policy=policy
                    )
                    results.append((h, v_adj, mean, std, None))
                else:
                    worst = +1e9
                    worst_tuple = None
                    for ho in pool_opp:
                        s2 = apply_ban(s1_ban, ho)
                        v_adj, mean, std = estimate_V(
                            s2, N=N_inner, evaluator=evaluator, beta=beta, policy=policy
                        )
                        if v_adj < worst:
                            worst = v_adj
                            worst_tuple = (h, v_adj, mean, std, ho)
                    results.append(worst_tuple)
            else:
                raise ValueError("opp_after_ban must be None/'pick'/'ban'.")


# MCTS estimate score for current state
def estimate_V(state, N, evaluator, beta=0.2, policy="weighted"):
    vals = []
    for _ in range(N):
        R5, D5 = rollout_complete(state, policy=policy)
        p = evaluator(R5, D5, state["first_pick"])
        vals.append(p)
    mean = sum(vals) / len(vals)
    # 风险折扣（可选）：mean - β·std
    std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
    return mean - beta * std, mean, std


class TeacherEvaluator:
    def __init__(self, model_path, map_path, emb_path):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 加载 ID 映射
        with open(map_path, 'r') as f:
            mappings = json.load(f)
        # 确保 key 是 int
        self.hero2idx = {int(k): v for k, v in mappings['hero2idx'].items()}
        self.idx2hero = {int(k): v for k, v in mappings['idx2hero'].items()}
        self.H = len(self.hero2idx)
        self.ALL_HEROES = list(self.hero2idx.keys())

        # 加载模型
        self.model = DeepSetsEvaluatorFP(self.H, d=...).to(
            self.device)  # d 需要从 config 读
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # 加载 MCTS 依赖的向量 (E64)
        # ... (把 E64_torch 相关的加载逻辑搬过来) ...

    @torch.no_grad()
    def predict_final_prob(self, R5, D5, fp):
        # 这是“深度学习”评估器，对应 predict_lineup_prob
        R_idx = torch.tensor([self.hero2idx.get(h, 0) for h in R5], dtype=torch.long)[
            None, :].to(self.device)
        D_idx = torch.tensor([self.hero2idx.get(h, 0) for h in D5], dtype=torch.long)[
            None, :].to(self.device)
        fp_tensor = torch.tensor([int(fp)], dtype=torch.long).to(self.device)

        _, logit = self.model(R_idx, D_idx, fp_tensor)
        return torch.sigmoid(logit).item()

    def estimate_current_V(self, state, N=64, beta=0.2):
        # 这是 MCTS 评估器，对应 estimate_V
        # 把 estimate_V 函数搬到这里作为类方法
        # 确保它调用的 evaluator 是 self.predict_final_prob
        v_adj, mean, std = estimate_V(state, N=N, evaluator=self.predict_final_prob, beta=beta, ...)
        return v_adj  # 返回风险调整分

    def get_teacher_action(self, state, N=32, topk=12):
        # 这是 MCTS 决策器，对应 best_action
        (best_h, ...), memo = best_action(state, evaluator=self.predict_final_prob, N=N, topk=topk, ...)
        return best_h
