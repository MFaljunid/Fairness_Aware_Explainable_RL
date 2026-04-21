import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import pickle
import glob
from collections import defaultdict

print("Evaluating BPR at K = 5, 10, 20, 30, 40")

# ── Load BPR model ─────────────────────────────────────────────────────
pkl_files = glob.glob('results/BPR/*.pkl')
assert len(pkl_files) > 0, "No BPR model found"
latest = sorted(pkl_files)[-1]
print(f"Loading: {latest}")

with open(latest, 'rb') as f:
    bpr_model = pickle.load(f)

# ── Load data ──────────────────────────────────────────────────────────
train  = pd.read_csv('data/train.csv')
test   = pd.read_csv('data/test.csv')
meta   = json.load(open('data/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

# ── 100-negative sampling evaluation ──────────────────────────────────
np.random.seed(42)
K_LIST = [5, 10, 20, 30, 40]

results = {k: {'hit': [], 'ndcg': []} for k in K_LIST}
recs_dict = {}

print("Running 100-neg evaluation...")
for uid, pos_item in test_items.items():
    seen          = train_set[uid] | {pos_item}
    pool          = list(set(range(N_ITEMS)) - seen)
    neg_items     = np.random.choice(pool, size=99, replace=False).tolist()
    candidates    = [pos_item] + neg_items

    # Get BPR scores
    try:
        scores = bpr_model.score(uid)
    except Exception:
        continue

    candidate_scores = [(item, scores[item]) for item in candidates]
    ranked = [item for item, score in
              sorted(candidate_scores, key=lambda x: x[1], reverse=True)]

    recs_dict[str(uid)] = ranked[:40]

    for k in K_LIST:
        hit  = 1.0 if pos_item in ranked[:k] else 0.0
        dcg  = 1.0/np.log2(ranked[:k].index(pos_item)+2) if pos_item in ranked[:k] else 0.0
        results[k]['hit'].append(hit)
        results[k]['ndcg'].append(dcg)

# ── Fairness metrics ───────────────────────────────────────────────────
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_dp_eo

user2idx   = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender = load_user_gender('data/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items() if u in user2idx}

# ── Print results ──────────────────────────────────────────────────────
print("\nBPR Results at different K:")
print(f"{'K':<6} {'HR@K':<10} {'NDCG@K':<10} {'DP':<10} {'EO':<10} {'Gini':<10}")
print("-" * 56)

bpr_tradeoff = {}
for k in K_LIST:
    hr   = np.mean(results[k]['hit'])
    ndcg = np.mean(results[k]['ndcg'])

    # Build recs at this k
    recs_k = {uid: items[:k] for uid, items in recs_dict.items()}

    # Fairness
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)

    fairness = compute_dp_eo(recs_k, user_gender, test_items, k)
    dp       = fairness['DP']
    eo       = fairness['EO']

    print(f"K={k:<4} HR={hr:.4f}   NDCG={ndcg:.4f}   DP={dp:.4f}   EO={eo:.4f}   Gini={gini:.4f}")

    bpr_tradeoff[k] = {
        'HR': round(hr, 4), 'NDCG': round(ndcg, 4),
        'DP': round(dp, 4), 'EO':   round(eo, 4),
        'Gini': round(gini, 4), 'Coverage': round(cov, 4)
    }

with open('results/bpr_tradeoff.json', 'w') as f:
    json.dump(bpr_tradeoff, f, indent=2)
print("\nSaved to results/bpr_tradeoff.json")