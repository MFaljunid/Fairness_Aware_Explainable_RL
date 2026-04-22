import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split

ML1M_DIR = os.path.join('data', 'ml-1m')   # everything in here
os.makedirs(ML1M_DIR, exist_ok=True)

# ── 1. Load raw ratings ───────────────────────────────────────────────
df = pd.read_csv(
    os.path.join(ML1M_DIR, 'ratings.dat'),
    sep='::',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    engine='python'
)
print(f"Raw interactions: {len(df)}")

# ── 2. Keep only ratings > 3 (same as FairIR paper) ──────────────────
df = df[df['rating'] > 3].copy()
print(f"After rating > 3 filter: {len(df)}")

# ── 3. Convert to implicit ────────────────────────────────────────────
df = df[['user_id', 'item_id', 'timestamp']].drop_duplicates(
    subset=['user_id', 'item_id'], keep='last')
df['feedback'] = 1
print(f"After dedup: {len(df)}")

# ── 4. No cold-start filter — FairIR does not apply one ──────────────
n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()
print(f"Users: {n_users} | Items: {n_items}")

# ── 5. Deterministic re-indexing ──────────────────────────────────────
sorted_users = sorted(df['user_id'].unique())
sorted_items = sorted(df['item_id'].unique())
user2idx = {int(u): int(i) for i, u in enumerate(sorted_users)}
item2idx = {int(it): int(i) for i, it in enumerate(sorted_items)}

df['user_id'] = df['user_id'].map(user2idx)
df['item_id'] = df['item_id'].map(item2idx)

n_users = len(sorted_users)
n_items = len(sorted_items)
print(f"Re-indexed — Users: {n_users} | Items: {n_items}")

# ── 6. Random 80/20 split (same as FairIR paper) ─────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── 7. Sanity checks ──────────────────────────────────────────────────
train_users = set(train_df['user_id'])
test_users  = set(test_df['user_id'])
assert test_users.issubset(train_users), "Some test users not in train"
print("PASS: all test users in train")

# ── 8. Save everything inside data/ml-1m/ ────────────────────────────
cols = ['user_id', 'item_id', 'feedback', 'timestamp']
train_df[cols].to_csv(os.path.join(ML1M_DIR, 'train.csv'), index=False)
test_df[cols].to_csv( os.path.join(ML1M_DIR, 'test.csv'),  index=False)

meta = {
    'n_users':  n_users,
    'n_items':  n_items,
    'user2idx': user2idx,
    'item2idx': item2idx,
    'split':    '80/20 random — same as FairIR paper',
    'filter':   'rating > 3',
    'users_dat': os.path.join(ML1M_DIR, 'users.dat'),
}
with open(os.path.join(ML1M_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nAll files saved to {ML1M_DIR}/:")
print(f"  ratings.dat  ← raw input")
print(f"  users.dat    ← gender info for DP/EO")
print(f"  movies.dat   ← movie names for explanation demo")
print(f"  train.csv    ({len(train_df)} rows)")
print(f"  test.csv     ({len(test_df)} rows)")
print(f"  meta.json    (n_users={n_users}, n_items={n_items})")
print(f"\nExpected: ~6040 users, ~3952 items, ~513112 interactions")