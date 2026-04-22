import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import pickle
import glob
from collections import defaultdict
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_dp_eo

print("BPR full evaluation with fairness metrics...")

train  = pd.read_csv('data/train.csv')
val    = pd.read_csv('data/val.csv')
test   = pd.read_csv('data/test.csv')
meta   = json.load(open('data/meta.json'))

N_ITEMS = meta['n_items']
K_LIST  = [5, 10, 20, 30, 40]

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

# Load BPR
pkl_files = glob.glob('results/BPR/*.pkl')
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

our_to_cornac_user = {int(k): v for k, v in bpr.uid_map.items()}
our_to_cornac_item = {int(k): v for k, v in bpr.iid_map.items()}

user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender  = load_user_gender('data/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

np.random.seed(42)
results   = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}
recs_dict = {}

for uid, pos_item in test_items.items():
    if uid not in our_to_cornac_user or pos_item not in our_to_cornac_item:
        continue

    cornac_uid = our_to_cornac_user[uid]
    seen       = train_set[uid] | val_set[uid] | {pos_item}
    pool       = [i for i in range(N_ITEMS)
                  if i not in seen and i in our_to_cornac_item]

    if len(pool) < 99:
        continue

    neg_items  = np.random.choice(pool, size=99, replace=False).tolist()
    candidates = [pos_item] + neg_items

    try:
        all_scores = bpr.score(cornac_uid)
    except Exception:
        continue

    candidate_scores = []
    for item in candidates:
        cornac_item = our_to_cornac_item.get(item, -1)
        if 0 <= cornac_item < len(all_scores):
            candidate_scores.append((item, float(all_scores[cornac_item])))
        else:
            candidate_scores.append((item, 0.0))

    ranked = [item for item, score in
              sorted(candidate_scores, key=lambda x: x[1], reverse=True)]
    recs_dict[str(uid)] = ranked

    for k in K_LIST:
        hit  = 1.0 if pos_item in ranked[:k] else 0.0
        ndcg = (1.0/np.log2(ranked[:k].index(pos_item)+2)
                if pos_item in ranked[:k] else 0.0)
        results[k]['hits'].append(hit)
        results[k]['ndcgs'].append(ndcg)

print(f"\n{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("-" * 50)

bpr_results = {}
for k in K_LIST:
    hr   = np.mean(results[k]['hits'])
    ndcg = np.mean(results[k]['ndcgs'])
    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)
    fairness = compute_dp_eo(recs_k, user_gender, test_items, k)
    bpr_results[k] = {
        'HR': round(hr, 4), 'NDCG': round(ndcg, 4),
        'DP': round(fairness['DP'], 4), 'EO': round(fairness['EO'], 4),
        'Gini': round(gini, 4), 'Coverage': round(cov, 4)
    }
    r = bpr_results[k]
    print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")

with open('results/bpr_tradeoff.json', 'w') as f:
    json.dump({str(k): v for k, v in bpr_results.items()}, f, indent=2)
print("\nSaved: results/bpr_tradeoff.json")