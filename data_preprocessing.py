# %% [markdown]
# 数据加载与预处理
# ---

import math
import random
import json
import requests
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from copy import deepcopy
import numpy as np
import pandas as pd
from itertools import combinations

path = Path("data/matches_chr.jsonl")

df = pd.read_json(path, lines=True)

print("Shape:", df.shape)
print("Columns:", list(df.columns)[:25], " ...")
df.head(3)


# 1) 读 JSONL（每行一个比赛）
df_matches = df.copy()

# 2) 基础清洗与类型
df_matches["start_time"] = pd.to_datetime(
    df_matches["start_time"], unit="s", errors="coerce"
)
df_matches["radiant_win"] = df_matches["radiant_win"].astype("boolean")
df_matches["duration"] = pd.to_numeric(df_matches["duration"], errors="coerce").astype(
    "Int32"
)

# 有些行可能 picks_bans 为空/缺失，统一成 []，便于 explode
df_matches["picks_bans"] = df_matches["picks_bans"].apply(
    lambda x: x if isinstance(x, list) else []
)

print("matches shape:", df_matches.shape)
df_matches.head(2)


# %%
# 3) 把每场的 picks_bans 列表「炸开」成行，并展开字典字段
steps = (
    df_matches[["match_id", "start_time", "radiant_win", "picks_bans"]]
    .explode("picks_bans", ignore_index=False)
    .dropna(subset=["picks_bans"])
)

# 展开 is_pick / hero_id / team / order
pb = pd.json_normalize(steps["picks_bans"])
steps = pd.concat(
    [
        steps.drop(columns=["picks_bans"]).reset_index(drop=True),
        pb.reset_index(drop=True),
    ],
    axis=1,
)

# 4) 类型与派生列
steps["is_pick"] = steps["is_pick"].astype("boolean")
steps["hero_id"] = pd.to_numeric(
    steps["hero_id"], errors="coerce").astype("Int16")
steps["team"] = pd.to_numeric(steps["team"], errors="coerce").astype("Int8")
steps["order"] = pd.to_numeric(steps["order"], errors="coerce").astype("Int16")

# OpenDota 约定：team=0 为 Radiant，team=1 为 Dire
steps["side"] = steps["team"].map({0: "radiant", 1: "dire"}).astype("category")
steps["action"] = np.where(steps["is_pick"], "pick", "ban")

steps = steps.sort_values(["match_id", "order"]).reset_index(drop=True)

print("steps shape:", steps.shape)
steps.head(8)


# %%


# 若你手里没有 steps，但有 df_wide，也可以直接用 df_wide 的 r_picks/d_picks。
# 这里以 steps（长表：每步一行，含 is_pick/team/hero_id/order）为准。
assert set(
    ["match_id", "start_time", "radiant_win",
        "is_pick", "team", "hero_id", "order"]
).issubset(steps.columns)


def extract_final_lineup(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("order")
    r = g[(g["is_pick"]) & (g["team"] == 0)]["hero_id"].astype(int).tolist()
    d = g[(g["is_pick"]) & (g["team"] == 1)]["hero_id"].astype(int).tolist()
    return pd.Series(
        {
            "start_time": g["start_time"].iloc[0],
            "radiant_win": bool(g["radiant_win"].iloc[0]),
            "R": r,
            "D": d,
        }
    )


finals = (
    steps.groupby("match_id", as_index=False)
    .apply(extract_final_lineup)
    .reset_index(drop=True)
)

# 只保留真正 5v5 的对局
finals["nR"] = finals["R"].str.len()
finals["nD"] = finals["D"].str.len()
finals = (
    finals[(finals["nR"] == 5) & (finals["nD"] == 5)]
    .drop(columns=["nR", "nD"])
    .reset_index(drop=True)
)

# 去除异常：两边有重复英雄的对局（理论上不该出现）
mask_conflict = finals.apply(lambda r: len(
    set(r["R"]) & set(r["D"])) > 0, axis=1)
finals = finals[~mask_conflict].reset_index(drop=True)

print("finals (5v5 only) shape:", finals.shape)
finals.head(3)


# %%
# 构建英雄词表（连续索引，预留 0 做 PAD——虽然这里不用 PAD，但方便后续统一）
all_heroes = sorted(
    set(h for lst in finals["R"] for h in lst)
    | set(h for lst in finals["D"] for h in lst)
)
hero2idx = {int(h): i + 1 for i, h in enumerate(all_heroes)}  # 1..H
idx2hero = {v: k for k, v in hero2idx.items()}
H = len(hero2idx)
print("H (unique heroes) =", H)


def to_idx_list(lst):
    return [hero2idx[int(h)] for h in lst]


finals["R_idx"] = finals["R"].apply(to_idx_list)
finals["D_idx"] = finals["D"].apply(to_idx_list)

# 展开成固定 10 列（R1..R5, D1..D5）+ 标签 y
out = pd.DataFrame(
    {
        "match_id": finals.index,  # 如需原 match_id，可在上面 groupby 保留
        "start_time": finals["start_time"],
        "y": finals["radiant_win"].astype(int),
    }
)
for i in range(5):
    out[f"R{i+1}"] = finals["R_idx"].apply(lambda x: x[i])
    out[f"D{i+1}"] = finals["D_idx"].apply(lambda x: x[i])

print(out.shape)
out.head(3)


# %%


# steps 必须包含：match_id,start_time,radiant_win,is_pick,team,hero_id,order
assert set(
    ["match_id", "start_time", "radiant_win",
        "is_pick", "team", "hero_id", "order"]
).issubset(steps.columns)


def extract_final_lineup(g: pd.DataFrame) -> pd.Series:
    g = g.sort_values("order")
    R = g[(g["is_pick"]) & (g["team"] == 0)]["hero_id"].astype(int).tolist()
    D = g[(g["is_pick"]) & (g["team"] == 1)]["hero_id"].astype(int).tolist()
    # 先手：第一手 pick 的 team==0 → Radiant 先手
    fp_row = g[g["is_pick"] == True].head(1)
    first_pick = 1 if (not fp_row.empty and int(
        fp_row["team"].iloc[0]) == 0) else 0
    return pd.Series(
        {
            "start_time": g["start_time"].iloc[0],
            "radiant_win": bool(g["radiant_win"].iloc[0]),
            "R": R,
            "D": D,
            "first_pick": first_pick,
        }
    )


finals = (
    steps.groupby("match_id", as_index=False)
    .apply(extract_final_lineup)
    .reset_index(drop=True)
)

# 只保留标准 5v5
finals = finals[
    (finals["R"].str.len() == 5) & (finals["D"].str.len() == 5)
].reset_index(drop=True)
# 去除异常：双方重复英雄
mask_conflict = finals.apply(lambda r: len(
    set(r["R"]) & set(r["D"])) > 0, axis=1)
finals = finals[~mask_conflict].reset_index(drop=True)

print("finals shape:", finals.shape)
finals.head(2)


# %%
# 英雄词表（1..H；0 预留 PAD）
all_heroes = sorted(
    set(h for lst in finals["R"] for h in lst)
    | set(h for lst in finals["D"] for h in lst)
)
hero2idx = {int(h): i + 1 for i, h in enumerate(all_heroes)}
idx2hero = {v: k for k, v in hero2idx.items()}
H = len(hero2idx)
print("H =", H)


def map_ids(lst):
    return [hero2idx[int(h)] for h in lst]


finals["R_idx"] = finals["R"].apply(map_ids)
finals["D_idx"] = finals["D"].apply(map_ids)

out = pd.DataFrame(
    {
        "y": finals["radiant_win"].astype(int),
        "first_pick": finals["first_pick"].astype(int),
    }
)
for i in range(5):
    out[f"R{i+1}"] = finals["R_idx"].str[i]
    out[f"D{i+1}"] = finals["D_idx"].str[i]

# 切分（你说都是同大版本，可直接随机或按时间；这里仍按时间稳定些）
# out = out.join(finals["start_time"]).sort_values("start_time").reset_index(drop=True)
out = out.sample(frac=1, random_state=42).reset_index(drop=True)
i1, i2 = int(0.8 * len(out)), int(0.9 * len(out))
tr, va, te = out.iloc[:i1], out.iloc[i1:i2], out.iloc[i2:]


def pack_xyf(df):
    X = df[[f"R{i}" for i in range(1, 6)] + [f"D{i}" for i in range(1, 6)]].to_numpy(
        np.int64
    )
    y = df["y"].to_numpy(np.int64)
    f = df["first_pick"].to_numpy(np.int64)
    return X, y, f


X_train, y_train, fp_train = pack_xyf(tr)
X_val, y_val, fp_val = pack_xyf(va)
X_test, y_test, fp_test = pack_xyf(te)
print("splits:", X_train.shape, X_val.shape, X_test.shape)


# %%

# 用最终阵容统计协同/对位（索引变成 0..H-1 便于矩阵操作）
ally = np.zeros((H, H), np.float64)
enemy = np.zeros((H, H), np.float64)
for R, D in zip(finals["R_idx"], finals["D_idx"]):
    R0 = [r - 1 for r in R]
    D0 = [d - 1 for d in D]
    for i in R0:
        for j in R0:
            if i != j:
                ally[i, j] += 1
    for i in D0:
        for j in D0:
            if i != j:
                ally[i, j] += 1
    for i in R0:
        for j in D0:
            enemy[i, j] += 1
            enemy[j, i] += 1


def ppmi(M):
    M = M + 1e-8
    row, col, tot = M.sum(1, keepdims=True), M.sum(0, keepdims=True), M.sum()
    PMI = np.log((M * tot) / (row * col))
    return np.maximum(PMI, 0.0)


S = ppmi(ally) - 0.6 * ppmi(enemy)  # α=0.6 可调
S = (S + S.T) / 2.0  # 对称化更稳

k = 64
E64 = TruncatedSVD(n_components=k, random_state=42).fit_transform(S)
E64 = E64 / (np.linalg.norm(E64, axis=1, keepdims=True) + 1e-8)  # 行归一化

# 扩到 d 维（前 64 维为预训练，其余置 0）
d = 128
E_init = np.zeros((H, d), np.float32)
E_init[:, :k] = E64.astype(np.float32)


# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


class LineupDatasetFP(Dataset):
    def __init__(self, X, y, fp):
        self.X = np.asarray(X, np.int64)
        self.y = np.asarray(y, np.int64)
        self.fp = np.asarray(fp, np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        xi = self.X[i]
        return {
            "R": torch.from_numpy(xi[:5]),
            "D": torch.from_numpy(xi[5:]),
            "y": torch.tensor(float(self.y[i])),
            "fp": torch.tensor(int(self.fp[i]), dtype=torch.long),  # 0/1
        }


BATCH = 1024
train_loader = DataLoader(
    LineupDatasetFP(X_train, y_train, fp_train),
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
val_loader = DataLoader(
    LineupDatasetFP(X_val, y_val, fp_val),
    batch_size=BATCH,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)
test_loader = DataLoader(
    LineupDatasetFP(X_test, y_test, fp_test),
    batch_size=BATCH,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)


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


model = DeepSetsEvaluatorFP(H, d=d, p_drop=0.2).to(device)

# 加载共现预训练向量（对齐 1..H；0是PAD）
with torch.no_grad():
    model.enc.emb.weight[1: 1 + H].copy_(torch.from_numpy(E_init))
# （可选）先冻结 2-3 个 epoch：
# model.enc.emb.weight.requires_grad_(False)


# %%
print("pos_rate overall =", float(out["y"].mean()))
print("first_pick=1 pos_rate =", float(
    out[out["first_pick"] == 1]["y"].mean()))
print("first_pick=0 pos_rate =", float(
    out[out["first_pick"] == 0]["y"].mean()))


# %%

H = len(hero2idx)


def to_diff_matrix(X):
    # X: (N,10) 1..H; 前5是R后5是D
    N = X.shape[0]
    M = np.zeros((N, H), dtype=np.int8)
    for i in range(5):
        M[np.arange(N), X[:, i] - 1] += 1
        M[np.arange(N), X[:, 5 + i] - 1] -= 1
    return M


Xtr_lin, Xva_lin, Xte_lin = (
    to_diff_matrix(X_train),
    to_diff_matrix(X_val),
    to_diff_matrix(X_test),
)

lr = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
lr.fit(Xtr_lin, y_train)


def eval_lin(X, y, name):
    p = lr.predict_proba(X)[:, 1]
    print(
        f"[{name}] NLL={log_loss(y, p):.4f}  Brier={brier_score_loss(y, p):.4f}  AUC={roc_auc_score(y, p):.4f}"
    )


eval_lin(Xtr_lin, y_train, "train")
eval_lin(Xva_lin, y_val, "valid")
eval_lin(Xte_lin, y_test, "test")


# %%
# 若把R/D互换、标签取反，AUC应该接近1-原AUC（在同一模型/打分下）

X_swap = np.concatenate([X_test[:, 5:], X_test[:, :5]], axis=1)
y_swap = 1 - y_test
p_orig = lr.predict_proba(Xte_lin)[:, 1]
p_swap = lr.predict_proba(to_diff_matrix(X_swap))[:, 1]
print("symmetry gap (should be ~0):", float(
    np.mean(np.abs(p_orig + p_swap - 1))))


# %%

# 需要 DataFrame finals: 列 ["R","D","radiant_win","first_pick"]
# 其中 R/D 是长度5的英雄ID列表（原始 hero_id），radiant_win 为 0/1
assert {"R", "D", "radiant_win", "first_pick"}.issubset(finals.columns)

# 词表（1..H）
all_heroes = sorted(
    set(h for L in finals["R"]
        for h in L) | set(h for L in finals["D"] for h in L)
)
hero2idx = {
    int(h): i for i, h in enumerate(all_heroes)
}  # 0..H-1 索引（注意与之前1..H不同）
idx2hero = {v: k for k, v in hero2idx.items()}
H = len(hero2idx)

# 计数容器
hero_win = np.zeros(H, np.int64)
hero_cnt = np.zeros(H, np.int64)
pair_win = np.zeros((H, H), np.int64)
pair_cnt = np.zeros((H, H), np.int64)
xwin = np.zeros((H, H), np.int64)
xcnt = np.zeros((H, H), np.int64)  # i 对位 j 时，含 i 方是否获胜
fp_win = np.zeros(2, np.int64)
fp_cnt = np.zeros(2, np.int64)

for _, row in finals.iterrows():
    y = int(row["radiant_win"])
    fp = int(row["first_pick"])
    R = [hero2idx[int(h)] for h in row["R"]]
    D = [hero2idx[int(h)] for h in row["D"]]

    # 单英雄：按各自阵营的胜负记
    for i in R:
        hero_win[i] += y
        hero_cnt[i] += 1
    for j in D:
        hero_win[j] += 1 - y
        hero_cnt[j] += 1

    # 同队二元协同（无序，上三角）
    for i, j in combinations(R, 2):
        pair_win[i, j] += y
        pair_cnt[i, j] += 1
    for i, j in combinations(D, 2):
        pair_win[i, j] += 1 - y
        pair_cnt[i, j] += 1

    # 跨队对位：i∈R 与 j∈D；以及 i∈D 与 j∈R，统一到“i方是否赢”
    for i in R:
        for j in D:
            xwin[i, j] += y
            xcnt[i, j] += 1  # i在R、j在D，Radiant赢=y => i方赢
            xwin[j, i] += 1 - y
            xcnt[j, i] += 1  # 交换阵营，i在D、j在R

    # 先手统计
    fp_win[fp] += y
    fp_cnt[fp] += 1


# %%
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


alpha1, alpha2, alphaX = 5, 3, 3  # 拉普拉斯平滑（可调，但不“训练”）
# 单英雄“基础强度”
p1 = (hero_win + alpha1 * 0.5) / (hero_cnt + alpha1)
phi1 = logit(p1)  # (H,)

# 同队二元：只取上三角，做“残差协同”避免重复计数
p2 = (pair_win + alpha2 * 0.5) / (pair_cnt + alpha2)  # (H,H)
phi2 = np.zeros_like(p2, dtype=np.float64)
mask = pair_cnt > 0
phi2[mask] = logit(
    p2[mask]
)  # 直接用二元对数几率；残差法也可：phi2 -= (phi1[i]+phi1[j])
# 为简洁，我们后面用“平均二元分数”，避免与单项强度叠加过度；也可减去 phi1 的和作为残差

# 跨队对位：i对j时，i方“对位强度”
px = (xwin + alphaX * 0.5) / (xcnt + alphaX)
psiX = logit(px)  # (H,H) 越大越克制；我们在总分里取负号

# 先手偏置（一个常数项）
p_fp = (fp_win + 1) / (fp_cnt + 2)
bias_fp = float(logit(p_fp[1]) - logit(p_fp[0])) / 2.0  # 近似化到 ±bias


# %%
def team_score_single(team):  # 单英雄强度之和 /5
    return float(np.sum(phi1[team]) / 5.0)


def team_score_pair(team):  # 二元协同的平均（C(5,2)=10）
    s = 0.0
    c = 0
    for i, j in combinations(team, 2):
        s += phi2[i, j] if i < j else phi2[j, i]
        c += 1
    return float(s / max(1, c))


def cross_score(R, D):  # 对位（取负号；越小越好），用所有 25 对的均值
    return float(-np.mean(psiX[np.ix_(R, D)]))


# 组合成总分；超参是“配方”，不是训练：可先按 1 : 0.5 : 0.7 试起
W1, W2, WX = 1.0, 0.5, 0.7


def lineup_score(radiant_ids, dire_ids, first_pick):
    R = [hero2idx[int(h)] for h in radiant_ids]
    D = [hero2idx[int(h)] for h in dire_ids]
    sR = W1 * team_score_single(R) + W2 * team_score_pair(R)
    sD = W1 * team_score_single(D) + W2 * team_score_pair(D)
    xc = WX * cross_score(R, D)
    bf = bias_fp if int(first_pick) == 1 else -bias_fp
    return (sR - sD) + xc + bf


# 映射为“概率”近似；一个温度 k（可固定 k=1.0；若想更准，可用验证集拟合 k，这算校准而非训练模型）
def lineup_prob(radiant_ids, dire_ids, first_pick, k=1.0):
    from math import exp

    z = k * lineup_score(radiant_ids, dire_ids, first_pick)
    return 1.0 / (1.0 + np.exp(-z))


# %%
# 你当前训练好的模型 + 温度T
# 假设：model, T, device, hero2idx 已就位；H 是英雄总数（1..H）


@torch.no_grad()
def predict_lineup_prob(model, radiant_ids, dire_ids, first_pick, hero2idx, T=1.0):
    def map_ids(lst):
        return [hero2idx.get(int(h), 0) for h in lst]  # 0=PAD

    R = torch.tensor(map_ids(radiant_ids), dtype=torch.long)[
        None, :].to(device)
    D = torch.tensor(map_ids(dire_ids), dtype=torch.long)[None, :].to(device)
    fp = torch.tensor([int(first_pick)], dtype=torch.long).to(device)
    _, logit = (
        model(R, D, fp) if "fp" in model.forward.__code__.co_varnames else model(R, D)
    )
    return torch.sigmoid(logit / T).item()


# %%

random.seed(42)

ALL_HEROES = list(hero2idx.keys())  # 原始 hero_id 列表（不是索引）
H = len(ALL_HEROES)


def legal_pool(state):
    used = set(state["R"]) | set(state["D"]) | set(state["B"])
    return [h for h in ALL_HEROES if h not in used]


def next_side(state):
    # 用“先手+数量差”决定下一手：谁人数少谁选；相等则看先手
    r, d = len(state["R"]), len(state["D"])
    if r == d:
        return "R" if state["first_pick"] == 1 else "D"
    return "R" if r < d else "D"


# 可选：用你算过的 E64 做“协同减对位”加权抽样，降低方差
E64_torch = None
try:
    import torch

    if "E64" in globals():
        # 构造 1..H 的表
        E_arr = np.zeros((H + 1, E64.shape[1]), np.float32)
        for hid, idx in hero2idx.items():
            E_arr[idx] = E64[hero2idx[hid] - 1]
        E64_torch = torch.from_numpy(E_arr)
except Exception:
    pass


def score_hero_weighted(h, R, D, alpha=0.6):
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


# %%
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


def score_pick(state, hero, evaluator, N=64, beta=0.2, policy="weighted"):
    s2 = apply_pick(state, hero)
    v_adj, mean, std = estimate_V(s2, N, evaluator, beta=beta, policy=policy)
    return v_adj, mean, std


def score_ban(state, hero, evaluator, N=64, beta=0.2, policy="weighted"):
    s2 = apply_ban(state, hero)
    v_adj, mean, std = estimate_V(s2, N, evaluator, beta=beta, policy=policy)
    return v_adj, mean, std


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


# %%
# ===== Route A · Monte-Carlo 补完 + ΔV 奖励 · 最小可跑版 =====

# --- 1) 评估器：用你已有的 lineup_prob；否则给一个占位（恒0.5）
if "lineup_prob" in globals():

    def evaluator(R5, D5, fp):
        return float(lineup_prob(R5, D5, fp, k=1.0))

else:
    print("[WARN] lineup_prob 未定义，使用恒 0.5 的占位评估器。")

    def evaluator(R5, D5, fp):
        return 0.5


# --- 2) 英雄池/向量：从 hero2idx 推断；E64 可选（用于加权采样）
assert "hero2idx" in globals(), "缺少 hero2idx（从你的数据构建的英雄词表）。"
ALL_HEROES = list(hero2idx.keys())
H = len(ALL_HEROES)

E64_torch = None
if "E64" in globals():
    # 构造 1..H 的表，便于快速索引（E64 是 (H, k) 对应 1..H）
    E_arr = np.zeros((max(hero2idx.values()) + 1, E64.shape[1]), np.float32)
    for hid, idx in hero2idx.items():
        E_arr[idx] = E64[hero2idx[hid] - 1]
    E64_torch = torch.from_numpy(E_arr)


# --- 3) Monte-Carlo 工具 ---
def legal_pool(state):
    used = set(state["R"]) | set(state["D"]) | set(state["B"])
    return [h for h in ALL_HEROES if h not in used]


def sample_weight(h, R, D, alpha=0.6):
    """E64 加权：与我方均值点积 - α*与对手均值点积；越大越优先。"""
    if E64_torch is None:
        return 0.0
    Eh = E64_torch[hero2idx[h]]

    def mean_vec(lst):
        if not lst:
            return torch.zeros_like(Eh)
        idx = torch.tensor([hero2idx[x] for x in lst])
        return E64_torch[idx].mean(dim=0)

    sR, sD = mean_vec(R), mean_vec(D)
    return float(torch.dot(Eh, sR) - alpha * torch.dot(Eh, sD))


def sample_one(pool, R, D, policy="weighted"):
    if not pool:
        return None
    if policy == "uniform" or E64_torch is None:
        return random.choice(pool)
    scores = [sample_weight(h, R, D) for h in pool]
    mx = max(scores)
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


def estimate_V(state, N=64, evaluator=evaluator, beta=0.2, policy="weighted"):
    vals = []
    for _ in range(N):
        R5, D5 = rollout_complete(state, policy=policy)
        p = evaluator(R5, D5, state["first_pick"])
        vals.append(p)
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
    return mean - beta * std, mean, std  # 返回：风险调整值、均值、标准差


# --- 4) 动作应用/打分 ---
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
    return s


def candidate_pool(state, topk=None, policy="weighted"):
    pool = legal_pool(state)
    if (
        (topk is None)
        or (len(pool) <= topk)
        or (E64_torch is None)
        or (policy == "uniform")
    ):
        return pool
    scored = sorted(
        [(h, sample_weight(h, state["R"], state["D"])) for h in pool],
        key=lambda x: x[1],
        reverse=True,
    )
    return [h for h, _ in scored[:topk]]


def best_action(
    state, evaluator=evaluator, kind="pick", N=64, beta=0.2, topk=24, policy="weighted"
):
    pool = candidate_pool(state, topk=topk, policy=policy)
    best = None
    best_score = -1.0
    memo = []
    for h in pool:
        s2 = apply_pick(state, h) if kind == "pick" else apply_ban(state, h)
        v_adj, mean, std = estimate_V(
            s2, N=N, evaluator=evaluator, beta=beta, policy=policy
        )
        memo.append((h, v_adj, mean, std))
        if v_adj > best_score:
            best = (h, v_adj, mean, std)
            best_score = v_adj
    memo.sort(key=lambda x: x[1], reverse=True)
    return best, memo


def dense_reward(
    prev_state, next_state, evaluator=evaluator, N=64, beta=0.2, policy="weighted"
):
    v_prev, _, _ = estimate_V(
        prev_state, N=N, evaluator=evaluator, beta=beta, policy=policy
    )
    v_next, _, _ = estimate_V(
        next_state, N=N, evaluator=evaluator, beta=beta, policy=policy
    )
    return v_next - v_prev, v_prev, v_next


# %%
# --- 5) 示例：从空局面给出第一手建议（Pick）
random.seed(42)
N = 64  # ← 你刚才报错的变量
beta = 0.25
policy = "weighted"  # 若没有 E64，将自动退化为 uniform
topk = 24


url = "https://api.opendota.com/api/heroes"
resp = requests.get(url)
heroes_data = resp.json()

with open("data/heroes_data.json", "w", encoding="utf-8") as f:
    json.dump(heroes_data, f, ensure_ascii=False, indent=2)

# 创建一个 {id: 名字} 的字典
hero_id_to_name = {h["id"]: h["localized_name"] for h in heroes_data}


state = {"R": [], "D": [], "B": [], "first_pick": 1, "turn": "R"}
(best_h, best_v, mean, std), memo = best_action(
    state, evaluator, kind="pick", N=N, beta=beta, topk=topk, policy=policy
)
best_hero_name = hero_id_to_name.get(best_h, f"Hero_{best_h}")

print(
    f"推荐先手Pick: {best_h} , {best_hero_name} | 风险调整分={best_v:.4f} (均值={mean:.4f}, std={std:.4f})"
)
print("Top5 候选及分数:", memo[:5])


# %%


def flip_turn(s):
    s2 = {k: (v[:] if isinstance(v, list) else v) for k, v in s.items()}
    s2["turn"] = "D" if s["turn"] == "R" else "R"
    return s2


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

    # 选“最坏情况下最好的”动作
    results.sort(key=lambda x: x[1], reverse=True)
    best = results[0] if results else (None, float("-inf"), None, None, None)
    return best, results


# %%
# 开局：先手方（Radiant）要 Pick，一步 minimax
state = {"R": [], "D": [], "B": [], "first_pick": 1, "turn": "R"}

best, memo = minimax_best_action(
    state,
    evaluator,
    kind="pick",
    N_outer=64,  # 外层（我们动作后直接评估时）采样数
    N_inner=32,  # 内层（对手回应）采样数
    beta=0.25,  # 风险折扣
    topk_self=24,  # 我方只评估前24个候选
    topk_opp=12,  # 对手只考虑前12个回应
    policy="weighted",
)

h, score, mean, std, opp = best

print(
    f"minimax 推荐Pick: {h} | 最坏回应: {opp} | 评分={score:.4f} (mean={mean:.4f}, std={std:.4f})"
)
print("Top5（按minimax得分）:", memo[:5])


# %%
# Save model and artifacts

# 1. 保存模型权重
torch.save(model.state_dict(), "models/deepsets_model.pth")

# 2. 保存 ID 映射表

mappings = {"hero2idx": hero2idx, "idx2hero": idx2hero}
with open("models/hero_mappings.json", "w") as f:
    json.dump(mappings, f)

# 3. 保存预训练向量
# E_init 是 1..H 的，我们存 E64 (0..H-1) 就够了
np.save("models/hero_embeddings.npy", E64)
print("All outputs are saved uner /models/ folder。")
# %%
# real hero id to name mapping
