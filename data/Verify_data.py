import pandas as pd
import json

train = pd.read_csv('Data/train.csv')
test  = pd.read_csv('Data/test.csv')
meta  = json.load(open('Data/meta.json'))

# Basic checks
print("=== Train ===")
print(train.shape)
print(train.head(3))

print("\n=== Test ===")
print(test.shape)
print(test.head(3))

# Critical check: every test user must also be in train
test_users  = set(test['user_id'].unique())
train_users = set(train['user_id'].unique())
assert test_users.issubset(train_users), "Some test users not in train!"
print("\nAll test users present in train.")

# Critical check: exactly 1 test item per user
assert (test.groupby('user_id').size() == 1).all(), "Some users have >1 test item!"
print("Exactly 1 test item per user.")

# No overlap between train and test per user
merged = train.merge(test, on=['user_id','item_id'], how='inner')
assert len(merged) == 0, "Train/test overlap detected!"
print("No train/test overlap.")

print(f"\nReady: {meta['n_users']} users, {meta['n_items']} items")