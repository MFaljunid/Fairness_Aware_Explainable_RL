import pandas as pd
import numpy as np
import os

os.makedirs('Data', exist_ok=True)

# ── 1. Load raw ratings ──────────────────────────────────────────────
df = pd.read_csv(
    'Data/ratings.dat',
    sep='::',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    engine='python'
)

print(f"Raw data: {len(df)} interactions")

# ── 2. Convert to implicit feedback ──────────────────────────────────
# Drop rating value — any interaction counts as a positive signal
df = df[['user_id', 'item_id', 'timestamp']].drop_duplicates()
df['feedback'] = 1

# ── 3. Filter cold-start users/items (keep users with ≥ 5 interactions)
user_counts = df['user_id'].value_counts()
item_counts = df['item_id'].value_counts()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]
df = df[df['item_id'].isin(item_counts[item_counts >= 5].index)]

print(f"After filtering: {len(df)} interactions")

# ── 4. Re-index users and items to 0-based integers ──────────────────
df['user_id'] = pd.Categorical(df['user_id']).codes
df['item_id'] = pd.Categorical(df['item_id']).codes

n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()
print(f"Users: {n_users}  |  Items: {n_items}")

# ── 5. Leave-one-out split (standard for implicit CF papers) ─────────
# Sort by timestamp so the LAST interaction per user goes to test
df = df.sort_values(['user_id', 'timestamp'])

test_df  = df.groupby('user_id').tail(1)          # 1 item per user → test
train_df = df.drop(test_df.index)                 # everything else → train

print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

# ── 6. Save ───────────────────────────────────────────────────────────
train_df[['user_id', 'item_id', 'feedback', 'timestamp']].to_csv(
    'data/train.csv', index=False)

test_df[['user_id', 'item_id', 'feedback', 'timestamp']].to_csv(
    'data/test.csv', index=False)

# Save metadata — useful later for your RL environment
meta = {'n_users': int(n_users), 'n_items': int(n_items)}
import json
with open('data/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nFiles saved:")
print("  Data/train.csv")
print("  Data/test.csv")
print("  Data/meta.json")