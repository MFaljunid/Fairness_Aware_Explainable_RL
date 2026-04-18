import pandas as pd
import json
from collections import defaultdict

# ── Load ──────────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
val   = pd.read_csv('data/val.csv')
test  = pd.read_csv('data/test.csv')
meta  = json.load(open('data/meta.json'))

n_users = meta['n_users']
n_items = meta['n_items']

# ── Build history ─────────────────────────────────────────────────────
user_history = defaultdict(list)
for _, row in train.iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))

# ── Checks ────────────────────────────────────────────────────────────
history_lengths = [len(v) for v in user_history.values()]

print(f"n_users          : {n_users}")
print(f"n_items          : {n_items}")
print(f"Train rows       : {len(train)}")
print(f"Val rows         : {len(val)}")
print(f"Test rows        : {len(test)}")
print(f"Users in history : {len(user_history)}")
print(f"Min interactions : {min(history_lengths)}")
print(f"Max interactions : {max(history_lengths)}")
print(f"Avg interactions : {sum(history_lengths)/len(history_lengths):.1f}")
print(f"Item id range    : 0 to {max(train['item_id'])}")
print(f"User id range    : 0 to {max(train['user_id'])}")
print()

# Check 1: every user has at least 3 interactions in train
assert min(history_lengths) >= 3, "Some users have < 3 train interactions"
print("PASS: all users have >= 3 train interactions")

# Check 2: no item index out of range
assert train['item_id'].max() < n_items, "Item index out of range"
assert val['item_id'].max()   < n_items, "Val item index out of range"
assert test['item_id'].max()  < n_items, "Test item index out of range"
print("PASS: all item ids within range")

# Check 3: no user index out of range
assert train['user_id'].max() < n_users, "User index out of range"
print("PASS: all user ids within range")

# Check 4: exactly one val and one test item per user
assert len(val)  == n_users, f"Expected {n_users} val rows,  got {len(val)}"
assert len(test) == n_users, f"Expected {n_users} test rows, got {len(test)}"
print("PASS: exactly 1 val and 1 test item per user")

# Check 5: no overlap between train and test
train_pairs = set(zip(train['user_id'], train['item_id']))
test_pairs  = set(zip(test['user_id'],  test['item_id']))
val_pairs   = set(zip(val['user_id'],   val['item_id']))
assert len(train_pairs & test_pairs) == 0, "Train/test overlap!"
assert len(train_pairs & val_pairs)  == 0, "Train/val overlap!"
print("PASS: no overlap between train, val, test")

print("\nAll checks passed. History is clean.")
print("Ready for Step 2: State Encoder")