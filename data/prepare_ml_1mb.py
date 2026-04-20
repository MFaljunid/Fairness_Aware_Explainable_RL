import pandas as pd
import numpy as np
import os
import json

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# ── 1. Load raw ratings ───────────────────────────────────────────────
df = pd.read_csv(
    os.path.join(DATA_DIR, 'ratings.dat'),
    sep='::',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    engine='python'
)
print(f"Raw interactions      : {len(df)}")

# ── 2. Convert to implicit feedback ──────────────────────────────────
df = (df[['user_id', 'item_id', 'timestamp']]
      .sort_values('timestamp')
      .drop_duplicates(subset=['user_id', 'item_id'], keep='last')
      .copy())
df['feedback'] = 1
print(f"After dedup           : {len(df)}")

# ── 3. Iterative cold-start filtering ────────────────────────────────
MIN_INTERACTIONS = 5
prev_len = -1
iteration = 0
while prev_len != len(df):
    prev_len = len(df)
    iteration += 1
    user_counts = df['user_id'].value_counts()
    item_counts = df['item_id'].value_counts()
    df = df[df['user_id'].isin(user_counts[user_counts >= MIN_INTERACTIONS].index)]
    df = df[df['item_id'].isin(item_counts[item_counts >= MIN_INTERACTIONS].index)]

print(f"After filtering ({iteration} iters): {len(df)} interactions")

# ── 4. Deterministic re-indexing ─────────────────────────────────────
sorted_users = sorted(df['user_id'].unique())
sorted_items = sorted(df['item_id'].unique())

user2idx = {int(u): int(i) for i, u in enumerate(sorted_users)}
item2idx = {int(it): int(i) for i, it in enumerate(sorted_items)}

df['user_id'] = df['user_id'].map(user2idx)
df['item_id'] = df['item_id'].map(item2idx)

n_users = len(sorted_users)
n_items = len(sorted_items)
print(f"Users: {n_users}  |  Items: {n_items}")

# ── 5. Sort before splitting ──────────────────────────────────────────
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# ── 6. Leave-one-out split with validation ────────────────────────────
# Fix: don't use include_groups — instead add split column directly
df['split'] = 'train'

# For each user mark last row as test, second-to-last as val
for uid, group in df.groupby('user_id'):
    idx = group.index.tolist()          # row indices for this user
    df.loc[idx[-1], 'split'] = 'test'
    if len(idx) >= 2:
        df.loc[idx[-2], 'split'] = 'val'

train_df = df[df['split'] == 'train'].drop(columns='split').reset_index(drop=True)
val_df   = df[df['split'] == 'val'].drop(columns='split').reset_index(drop=True)
test_df  = df[df['split'] == 'test'].drop(columns='split').reset_index(drop=True)

print(f"Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

# ── 7. Sanity checks ──────────────────────────────────────────────────
assert len(test_df) == n_users, "Every user must have exactly 1 test item"
assert len(val_df)  == n_users, "Every user must have exactly 1 val item"

train_users = set(train_df['user_id'])
test_users  = set(test_df['user_id'])
assert test_users == train_users, "Some test users missing from train"

train_pairs = set(zip(train_df['user_id'], train_df['item_id']))
test_pairs  = set(zip(test_df['user_id'],  test_df['item_id']))
val_pairs   = set(zip(val_df['user_id'],   val_df['item_id']))
assert len(train_pairs & test_pairs) == 0, "Train/test overlap detected"
assert len(train_pairs & val_pairs)  == 0, "Train/val overlap detected"

print("All sanity checks passed.")

# ── 8. Save ───────────────────────────────────────────────────────────
cols = ['user_id', 'item_id', 'feedback', 'timestamp']
train_df[cols].to_csv(os.path.join(DATA_DIR, 'train.csv'), index=False)
val_df[cols].to_csv(  os.path.join(DATA_DIR, 'val.csv'),   index=False)
test_df[cols].to_csv( os.path.join(DATA_DIR, 'test.csv'),  index=False)

meta = {
    'n_users':          n_users,
    'n_items':          n_items,
    'min_interactions': MIN_INTERACTIONS,
    'user2idx':         user2idx,
    'item2idx':         item2idx,
}
with open(os.path.join(DATA_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\nFiles saved to {DATA_DIR}/:")
print(f"  train.csv  ({len(train_df)} rows)")
print(f"  val.csv    ({len(val_df)} rows)")
print(f"  test.csv   ({len(test_df)} rows)")
print(f"  meta.json  (n_users={n_users}, n_items={n_items})")