import cornac
import pandas as pd
import numpy as np
import os, json

os.makedirs('data/lastfm', exist_ok=True)

print("Loading LastFM from Cornac...")
data = cornac.datasets.lastfm.load_feedback()

# Convert to DataFrame
rows = []
for uid, iid, rating in data:
    rows.append({'user_id': uid, 'item_id': iid, 'rating': float(rating)})

df = pd.DataFrame(rows)
print(f"Raw interactions: {len(df)}")
print(f"Users: {df['user_id'].nunique()}")
print(f"Items: {df['item_id'].nunique()}")

# Convert to implicit
df['feedback'] = 1
df = df[['user_id', 'item_id', 'feedback']].drop_duplicates()

# Filter cold start
MIN_INTERACTIONS = 5
prev_len = -1
while prev_len != len(df):
    prev_len = len(df)
    uc = df['user_id'].value_counts()
    ic = df['item_id'].value_counts()
    df = df[df['user_id'].isin(uc[uc >= MIN_INTERACTIONS].index)]
    df = df[df['item_id'].isin(ic[ic >= MIN_INTERACTIONS].index)]

print(f"After filtering: {len(df)} interactions")

# Re-index
sorted_users = sorted(df['user_id'].unique())
sorted_items = sorted(df['item_id'].unique())
user2idx = {int(u): int(i) for i, u in enumerate(sorted_users)}
item2idx = {int(it): int(i) for i, it in enumerate(sorted_items)}

df['user_id'] = df['user_id'].map(user2idx)
df['item_id'] = df['item_id'].map(item2idx)

n_users = len(sorted_users)
n_items = len(sorted_items)
print(f"Users: {n_users} | Items: {n_items}")

# Add fake timestamp for split
df['timestamp'] = range(len(df))
df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

# Split
df['split'] = 'train'
for uid, group in df.groupby('user_id'):
    idx = group.index.tolist()
    df.loc[idx[-1], 'split'] = 'test'
    if len(idx) >= 2:
        df.loc[idx[-2], 'split'] = 'val'

train_df = df[df['split'] == 'train'].drop(columns='split')
val_df   = df[df['split'] == 'val'].drop(columns='split')
test_df  = df[df['split'] == 'test'].drop(columns='split')

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Save
cols = ['user_id', 'item_id', 'feedback', 'timestamp']
train_df[cols].to_csv('data/lastfm/train.csv', index=False)
val_df[cols].to_csv(  'data/lastfm/val.csv',   index=False)
test_df[cols].to_csv( 'data/lastfm/test.csv',  index=False)

meta = {
    'n_users':   n_users,
    'n_items':   n_items,
    'user2idx':  user2idx,
    'item2idx':  item2idx,
    'dataset':   'lastfm'
}
with open('data/lastfm/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\nLastFM data saved to data/lastfm/")
print(f"  train.csv  ({len(train_df)} rows)")
print(f"  val.csv    ({len(val_df)} rows)")
print(f"  test.csv   ({len(test_df)} rows)")
print(f"  meta.json  (n_users={n_users}, n_items={n_items})")