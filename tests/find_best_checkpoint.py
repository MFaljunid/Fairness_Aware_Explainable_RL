import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pandas as pd
import json
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

CFG = {'emb_dim': 64, 'hidden_dim': 256, 'window': 10, 'fairness_lambda': 0.1}

env = RecEnv('data/train.csv', 'data/meta.json',
             emb_dim=CFG['emb_dim'], window=CFG['window'],
             fairness_lambda=CFG['fairness_lambda'])

emb_path = 'data/bpr_item_embeddings.npy'
if os.path.exists(emb_path):
    env.load_pretrained_embeddings(np.load(emb_path))

test  = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')

def get_item_seq(user_id):
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

def ndcg_at_k(recommended, relevant, k):
    dcg  = sum(1/np.log2(i+2) for i, item in enumerate(recommended[:k])
               if item in set(relevant))
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg/idcg if idcg > 0 else 0.0

checkpoints = sorted([
    f for f in os.listdir('results') if f.startswith('policy_ep') and f.endswith('.pt')
])
print(f"Found {len(checkpoints)} checkpoints\n")

test_df     = pd.read_csv('data/test.csv')
user_groups = test_df.groupby('user_id')['item_id'].apply(list)

best_ndcg  = 0.0
best_ckpt  = None
results    = []

for ckpt in checkpoints:
    policy = ActorCriticPolicy(CFG['emb_dim'], env.n_items, CFG['hidden_dim'])
    policy.load_state_dict(
        torch.load(f'results/{ckpt}', map_location='cpu'))
    policy.eval()

    recs = {}
    with torch.no_grad():
        for uid in test['user_id'].unique():
            env.reset(int(uid))
            topk = []
            for _ in range(20):
                item_seq = get_item_seq(int(uid))
                action   = policy.greedy_action(
                    item_seq, env.item_exposure,
                    exclude_items=env.get_excluded_items())
                _, _, done, _ = env.step(action)
                topk.append(action)
                if done:
                    break
            recs[str(uid)] = topk

    ndcg_scores = []
    for uid, relevant_items in user_groups.items():
        uid_str = str(uid)
        if uid_str not in recs:
            continue
        ndcg_scores.append(ndcg_at_k(recs[uid_str], relevant_items, 10))

    avg_ndcg = np.mean(ndcg_scores)
    coverage = len(set(item for topk in recs.values() for item in topk)) / env.n_items

    results.append((ckpt, avg_ndcg, coverage))
    print(f"{ckpt:30s} | NDCG@10: {avg_ndcg:.4f} | Coverage: {coverage:.4f}")

    if avg_ndcg > best_ndcg:
        best_ndcg = avg_ndcg
        best_ckpt = ckpt

print(f"\nBest checkpoint: {best_ckpt}")
print(f"Best NDCG@10  : {best_ndcg:.4f}")

# Copy best checkpoint as policy_final.pt
import shutil
shutil.copy(f'results/{best_ckpt}', 'results/policy_best.pt')
print(f"Saved as results/policy_best.pt")