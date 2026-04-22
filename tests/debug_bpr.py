# Save as tests/debug_bpr.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import pickle
import glob
from collections import defaultdict

train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')
meta  = json.load(open('data/meta.json'))

N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

pkl_files = glob.glob('results/BPR/*.pkl')
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

bpr_u2c = {int(k): v for k, v in bpr.uid_map.items()}
bpr_i2c = {int(k): v for k, v in bpr.iid_map.items()}

# ── Test with user 0 ───────────────────────────────────────────────────
uid      = 0
pos_item = test_items[uid]
seen     = train_set[uid]

print(f"User {uid}")
print(f"Test item (our idx): {pos_item}")
print(f"Cornac user idx:     {bpr_u2c.get(uid, 'NOT FOUND')}")
print(f"Cornac item idx:     {bpr_i2c.get(pos_item, 'NOT FOUND')}")
print(f"Train items count:   {len(seen)}")
print(f"N_ITEMS:             {N_ITEMS}")

# Get scores
cornac_uid = bpr_u2c[uid]
scores     = bpr.score(cornac_uid)
print(f"\nScores array length: {len(scores)}")
print(f"Max score:           {scores.max():.4f}")
print(f"Min score:           {scores.min():.4f}")

# Score of test item
cornac_pos = bpr_i2c[pos_item]
print(f"\nTest item cornac idx:  {cornac_pos}")
print(f"Test item score:       {scores[cornac_pos]:.4f}")

# Build full ranking
all_scores = []
for item in range(N_ITEMS):
    if item in seen:
        continue
    cornac_item = bpr_i2c.get(item, -1)
    if 0 <= cornac_item < len(scores):
        all_scores.append((item, float(scores[cornac_item])))

ranked = [item for item, score in
          sorted(all_scores, key=lambda x: x[1], reverse=True)]

print(f"\nRanked items count:    {len(ranked)}")
print(f"Test item rank:        {ranked.index(pos_item) + 1 if pos_item in ranked else 'NOT IN RANKED'}")
print(f"Top 10 items:          {ranked[:10]}")
print(f"Test item in top 10:   {pos_item in ranked[:10]}")
print(f"Test item in top 20:   {pos_item in ranked[:20]}")
print(f"Test item in top 50:   {pos_item in ranked[:50]}")