import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import pickle
import glob
from collections import defaultdict
from rl_model.fairness_reranker import FairnessReranker
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_fairir_dp_eo

DATA_DIR = 'data/ml-1m'
K_LIST   = [10, 20, 30, 40]

train = pd.read_csv(f'{DATA_DIR}/train.csv')
test  = pd.read_csv(f'{DATA_DIR}/test.csv')
meta  = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

test_set = defaultdict(set)
for _, row in test.iterrows():
    test_set[int(row['user_id'])].add(int(row['item_id']))

user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender  = load_user_gender(f'{DATA_DIR}/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

pkl_files = glob.glob('results/BPR/*.pkl')
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

bpr_u2c = {int(k): v for k, v in bpr.uid_map.items()}
bpr_i2c = {int(k): v for k, v in bpr.iid_map.items()}

def compute_hr_ndcg(ranked, relevant, k):
    topk = ranked[:k]
    hits = set(topk) & relevant
    hr   = len(hits) / min(len(relevant), k)
    dcg  = sum(1/np.log2(i+2) for i, item in enumerate(topk)
               if item in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    ndcg = dcg/idcg if idcg > 0 else 0.0
    return hr, ndcg

print("BPR + Fairness Reranker — testing lambda values")
print(f"{'Lambda':<10} {'HR@10':>7} {'NDCG@10':>9} {'DP@10':>7} {'EO@10':>7}")
print("-" * 45)


# Before the evaluation loop, sort users by gender alternating
male_users   = [uid for uid, g in user_gender.items() 
                if g == 'M' and uid in test_set]
female_users = [uid for uid, g in user_gender.items() 
                if g == 'F' and uid in test_set]

# Interleave male and female users
from itertools import zip_longest
ordered_users = []
for m, f in zip_longest(male_users, female_users):
    if m is not None:
        ordered_users.append(m)
    if f is not None:
        ordered_users.append(f)

print(f"Ordered users: {len(male_users)} male, {len(female_users)} female")

for lambda_fair in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    reranker   = FairnessReranker(N_ITEMS, lambda_fair=lambda_fair)
    recs_dict  = {}
    results    = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

    # Use gender-balanced order
    for uid in ordered_users:
        relevant = test_set[uid]
        if uid not in bpr_u2c:
            continue
        cornac_uid = bpr_u2c[uid]
        try:
            scores_raw = bpr.score(cornac_uid)
        except Exception:
            continue

        seen   = train_set[uid]
        scores = np.zeros(N_ITEMS)
        for item in range(N_ITEMS):
            if item in seen:
                continue
            cornac_item = bpr_i2c.get(item, -1)
            if 0 <= cornac_item < len(scores_raw):
                scores[item] = float(scores_raw[cornac_item])

        gender = user_gender.get(uid, 'M')
        ranked = reranker.rerank(scores, gender, seen, k=40)
        recs_dict[str(uid)] = ranked

        for k in K_LIST:
            hr, ndcg = compute_hr_ndcg(ranked, relevant, k)
            results[k]['hits'].append(hr)
            results[k]['ndcgs'].append(ndcg)

    recs_k   = {uid: items[:10] for uid, items in recs_dict.items()}
    fairness = compute_fairir_dp_eo(recs_k, user_gender,
                                     test_set, N_ITEMS, 10)
    hr10   = np.mean(results[10]['hits'])
    ndcg10 = np.mean(results[10]['ndcgs'])
    dp10   = fairness['DP']
    eo10   = fairness['EO']
    print(f"{lambda_fair:<10} {hr10:>7.4f} {ndcg10:>9.4f} "
          f"{dp10:>7.4f} {eo10:>7.4f}")