import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pandas as pd
import json
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy
from metrics.user_fairness_metrics import compute_dp_eo, load_user_gender

print("=" * 55)
print("User Fairness Evaluation (DP and EO)")
print("Same metrics as FairIR paper (Shi et al.)")
print("=" * 55)

CFG = {'emb_dim': 64, 'hidden_dim': 256, 'window': 10,
       'fairness_lambda': 0.1}

# ── Load data ──────────────────────────────────────────────────────────
train  = pd.read_csv('data/train.csv')
test   = pd.read_csv('data/test.csv')
meta   = json.load(open('data/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

# Load user2idx mapping to convert original user ids to new indices
user2idx = {int(k): int(v) for k, v in meta['user2idx'].items()}
idx2user = {v: k for k, v in user2idx.items()}

# Load gender from users.dat
print("\nLoading user gender from data/users.dat...")
raw_gender  = load_user_gender('data/users.dat')

# Map original user ids to new indexed user ids
user_gender = {}
for orig_uid, gender in raw_gender.items():
    if orig_uid in user2idx:
        new_uid = user2idx[orig_uid]
        user_gender[new_uid] = gender

male_count   = sum(1 for g in user_gender.values() if g == 'M')
female_count = sum(1 for g in user_gender.values() if g == 'F')
print(f"Male users  : {male_count}")
print(f"Female users: {female_count}")

# ── Build test items dict ──────────────────────────────────────────────
test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

# ── Load environment and policy ────────────────────────────────────────
env = RecEnv(
    train_path='data/train.csv',
    meta_path='data/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window'],
    fairness_lambda=CFG['fairness_lambda']
)

emb_path = 'data/bpr_item_embeddings.npy'
if os.path.exists(emb_path):
    env.load_pretrained_embeddings(np.load(emb_path))

policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=N_ITEMS,
    hidden_dim=CFG['hidden_dim']
)
policy.load_state_dict(
    torch.load('results/policy_final.pt', map_location='cpu'))
policy.eval()
print("Model loaded.")

def get_item_seq(user_id):
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad    = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

# ── Generate recommendations ───────────────────────────────────────────
print("\nGenerating recommendations...")
recs = {}

with torch.no_grad():
    for uid in test['user_id'].unique():
        env.reset(int(uid))
        topk = []
        for _ in range(10):
            item_seq = get_item_seq(int(uid))
            action   = policy.greedy_action(
                item_seq, env.item_exposure,
                exclude_items=env.get_excluded_items())
            topk.append(action)
            env._session_history.append(action)
        recs[str(uid)] = topk

# ── Compute DP and EO ─────────────────────────────────────────────────
print("\nComputing DP and EO metrics...")
fairness = compute_dp_eo(recs, user_gender, test_items, k=10)

print("\n" + "=" * 55)
print(f"Male users   HR@10 : {fairness['male_HR']:.4f}  (n={fairness['n_male']})")
print(f"Female users HR@10 : {fairness['female_HR']:.4f}  (n={fairness['n_female']})")
print(f"DP                 : {fairness['DP']:.4f}  (lower is fairer)")
print(f"EO                 : {fairness['EO']:.4f}  (lower is fairer)")
print("=" * 55)

# ── Compare with FairIR paper ──────────────────────────────────────────
print("\nComparison with FairIR paper (Table 3, K=10):")
print(f"{'Model':<20} {'NDCG@10':>8} {'DP':>8} {'EO':>8}")
print("-" * 50)
print(f"{'BPR (their)':<20} {'0.2492':>8} {'0.6541':>8} {'0.7158':>8}")
print(f"{'FairIR_BPR':<20} {'0.2777':>8} {'0.5741':>8} {'0.6258':>8}")
print(f"{'FairIR_GCCF':<20} {'0.2878':>8} {'0.5986':>8} {'0.6501':>8}")
print(f"{'Your RL':<20} {'see 100neg':>8} {fairness['DP']:>8.4f} {fairness['EO']:>8.4f}")
print("-" * 50)

# Save
with open('results/user_fairness.json', 'w') as f:
    json.dump({'model': 'RL-CF', **fairness}, f, indent=2)
print("\nSaved to results/user_fairness.json")