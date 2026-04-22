import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pandas as pd
import json
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_dp_eo

print("=" * 60)
print("RL Model — Full Evaluation at K = 5, 10, 20, 30, 40")
print("=" * 60)

CFG = {
    'emb_dim':         64,
    'hidden_dim':      256,
    'window':          10,
    'fairness_lambda': 0.1,
    'n_negatives':     99,
    'k_list':          [5, 10, 20, 30, 40],   # ← all K values
}

# ── Load data ──────────────────────────────────────────────────────────
train  = pd.read_csv('data/train.csv')
val    = pd.read_csv('data/val.csv')
test   = pd.read_csv('data/test.csv')
meta   = json.load(open('data/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

print(f"Users: {N_USERS}  |  Items: {N_ITEMS}")

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
    print("Loaded BPR embeddings")

policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=N_ITEMS,
    hidden_dim=CFG['hidden_dim']
)
policy.load_state_dict(
    torch.load('results/policy_final.pt', map_location='cpu'))
policy.eval()
print("Loaded model: results/policy_final.pt")

# ── Gender for DP/EO ───────────────────────────────────────────────────
user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender  = load_user_gender('data/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

# ── Helpers ────────────────────────────────────────────────────────────
def get_item_seq(user_id):
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad    = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

def hit_at_k(ranked, pos, k):
    return 1.0 if pos in ranked[:k] else 0.0

def ndcg_at_k(ranked, pos, k):
    if pos in ranked[:k]:
        return 1.0 / np.log2(ranked[:k].index(pos) + 2)
    return 0.0

def mrr_score(ranked, pos):
    if pos in ranked:
        return 1.0 / (ranked.index(pos) + 1)
    return 0.0

# ── Evaluation loop ────────────────────────────────────────────────────
print("\nRunning evaluation...")
np.random.seed(42)

results   = {k: {'hits': [], 'ndcgs': []} for k in CFG['k_list']}
mrr_list  = []
recs_dict = {}

with torch.no_grad():
    for uid, pos_item in test_items.items():
        seen       = train_set[uid] | val_set[uid] | {pos_item}
        pool       = list(set(range(N_ITEMS)) - seen)

        if len(pool) < 99:
            continue

        neg_items  = np.random.choice(pool, size=99, replace=False).tolist()
        candidates = [pos_item] + neg_items

        item_seq         = get_item_seq(uid)
        seq_t            = torch.LongTensor(item_seq).unsqueeze(0)
        exp_t            = torch.zeros(N_ITEMS)
        logits, _, _, _  = policy.forward(seq_t, exp_t)
        logits_np        = logits.squeeze(0).numpy()

        ranked = sorted(candidates,
                        key=lambda x: logits_np[x], reverse=True)

        recs_dict[str(uid)] = ranked
        mrr_list.append(mrr_score(ranked, pos_item))

        for k in CFG['k_list']:
            results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
            results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

# ── Print results table ────────────────────────────────────────────────
print(f"\n{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("=" * 55)

rl_results = {}
for k in CFG['k_list']:
    hr   = np.mean(results[k]['hits'])
    ndcg = np.mean(results[k]['ndcgs'])

    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)
    fairness = compute_dp_eo(recs_k, user_gender, test_items, k)

    rl_results[k] = {
        'HR':       round(hr,   4),
        'NDCG':     round(ndcg, 4),
        'DP':       round(fairness['DP'], 4),
        'EO':       round(fairness['EO'], 4),
        'Gini':     round(gini, 4),
        'Coverage': round(cov,  4),
    }
    r = rl_results[k]
    print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")

print(f"\nMRR: {np.mean(mrr_list):.4f}")
print("=" * 55)

# ── Save ───────────────────────────────────────────────────────────────
with open('results/rl_full_evaluation.json', 'w') as f:
    json.dump({str(k): v for k, v in rl_results.items()}, f, indent=2)

print("\nSaved: results/rl_full_evaluation.json")