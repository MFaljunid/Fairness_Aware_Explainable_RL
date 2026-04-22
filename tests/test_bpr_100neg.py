import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import pickle
import glob
from collections import defaultdict

print("BPR 100-negative sampling evaluation (correct mapping)...")

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

# ── Load BPR ───────────────────────────────────────────────────────────
pkl_files = glob.glob('results/BPR/*.pkl')
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

# ── Check mapping keys ─────────────────────────────────────────────────
sample_uid_key = list(bpr.uid_map.keys())[0]
sample_iid_key = list(bpr.iid_map.keys())[0]
print(f"uid_map key type : {type(sample_uid_key)} — sample: {sample_uid_key}")
print(f"iid_map key type : {type(sample_iid_key)} — sample: {sample_iid_key}")

# ── Build our_id → cornac_index mappings ──────────────────────────────
our_to_cornac_user = {}
for key, cornac_idx in bpr.uid_map.items():
    try:
        our_to_cornac_user[int(key)] = cornac_idx
    except Exception:
        pass

our_to_cornac_item = {}
for key, cornac_idx in bpr.iid_map.items():
    try:
        our_to_cornac_item[int(key)] = cornac_idx
    except Exception:
        pass

print(f"Mapped {len(our_to_cornac_user)} users")
print(f"Mapped {len(our_to_cornac_item)} items")

# ── 100-neg evaluation ─────────────────────────────────────────────────
np.random.seed(42)
results  = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}
skipped  = 0

for uid, pos_item in test_items.items():
    if uid not in our_to_cornac_user:
        skipped += 1
        continue
    if pos_item not in our_to_cornac_item:
        skipped += 1
        continue

    cornac_uid = our_to_cornac_user[uid]

    seen = train_set[uid] | val_set[uid] | {pos_item}
    pool = [i for i in range(N_ITEMS)
            if i not in seen and i in our_to_cornac_item]

    if len(pool) < 99:
        skipped += 1
        continue

    neg_items  = np.random.choice(pool, size=99, replace=False).tolist()
    candidates = [pos_item] + neg_items

    try:
        all_scores = bpr.score(cornac_uid)
    except Exception:
        skipped += 1
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

    for k in K_LIST:
        hit  = 1.0 if pos_item in ranked[:k] else 0.0
        ndcg = (1.0/np.log2(ranked[:k].index(pos_item)+2)
                if pos_item in ranked[:k] else 0.0)
        results[k]['hits'].append(hit)
        results[k]['ndcgs'].append(ndcg)

print(f"Evaluated: {len(test_items) - skipped} users  |  Skipped: {skipped}")
print(f"\n{'K':<6} {'HR@K':>8} {'NDCG@K':>8}")
print("-" * 25)
for k in K_LIST:
    if results[k]['hits']:
        hr   = np.mean(results[k]['hits'])
        ndcg = np.mean(results[k]['ndcgs'])
        print(f"K={k:<4} {hr:>8.4f} {ndcg:>8.4f}")