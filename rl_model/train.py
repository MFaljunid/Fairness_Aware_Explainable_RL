import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from collections import deque

from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

os.makedirs('results', exist_ok=True)

CFG = {
    'emb_dim':          64,
    'hidden_dim':       256,
    'lr':               3e-4,
    'gamma':            0.99,
    'fairness_lambda':  0.1,
    'n_episodes':       5000,
    'max_steps':        20,
    'log_every':        100,
    'save_every':       500,
    'window':           10,
}

env = RecEnv(
    train_path='data/train.csv',
    meta_path='data/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window'],
    fairness_lambda=CFG['fairness_lambda']
)

policy    = ActorCriticPolicy(
    state_dim=CFG['emb_dim'],
    n_items=env.n_items,
    hidden_dim=CFG['hidden_dim']
)
optimizer = optim.Adam(policy.parameters(), lr=CFG['lr'])

episode_rewards = []
recent_rewards  = deque(maxlen=100)

print("Starting RL training...")
print(f"Users: {env.n_users}  |  Items: {env.n_items}")

for episode in range(CFG['n_episodes']):

    user_id    = np.random.randint(0, env.n_users)
    state      = env.reset(user_id)
    seen_items = list(env.user_history[user_id])

    log_probs, values, rewards = [], [], []

    for step in range(CFG['max_steps']):
        action, log_prob, value = policy.select_action(state, exclude_items=seen_items)
        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)   # ← tensor, keeps grad
        values.append(value)         # ← tensor, keeps grad
        rewards.append(reward)       # ← float, fine
        seen_items.append(action)
        state = next_state

        if done:
            break

    # ── Discounted returns ────────────────────────────────────────────
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + CFG['gamma'] * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # ── Stack tensors — preserves grad_fn ────────────────────────────
    log_probs_t = torch.stack(log_probs)   # ← stack, not FloatTensor()
    values_t    = torch.stack(values)      # ← stack, not FloatTensor()
    advantages  = (returns - values_t.detach())

    # ── Losses ────────────────────────────────────────────────────────
    actor_loss  = -(log_probs_t * advantages).mean()
    critic_loss = nn.MSELoss()(values_t, returns)
    loss        = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()    # ← works now
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()

    ep_reward = sum(rewards)
    episode_rewards.append(ep_reward)
    recent_rewards.append(ep_reward)

    if (episode + 1) % CFG['log_every'] == 0:
        avg = np.mean(recent_rewards)
        print(f"Episode {episode+1:>5} | "
              f"Avg reward (last 100): {avg:.4f} | "
              f"Loss: {loss.item():.4f}")

    if (episode + 1) % CFG['save_every'] == 0:
        path = f"results/policy_ep{episode+1}.pt"
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

# ── Final save ────────────────────────────────────────────────────────
torch.save(policy.state_dict(), 'results/policy_final.pt')
np.save('results/episode_rewards.npy', np.array(episode_rewards))
print("\nTraining complete. Model saved to results/policy_final.pt")

# ── Fairness evaluation ───────────────────────────────────────────────
from metrics.fairness_metrics import compute_all
import pandas as pd

print("\nEvaluating fairness on test users...")
test = pd.read_csv('data/test.csv')
policy.eval()
recs = {}

with torch.no_grad():   # ← no gradients needed during evaluation
    for uid in test['user_id'].unique():
        state = env.reset(int(uid))
        seen  = list(env.user_history[int(uid)])
        topk  = []
        for _ in range(10):
            action, _, _ = policy.select_action(state, exclude_items=seen + topk)
            topk.append(action)
        recs[str(uid)] = topk

item_pop = np.bincount(
    pd.read_csv('data/train.csv')['item_id'].values,
    minlength=env.n_items
).astype(float)

metrics = compute_all(recs, env.n_items, item_pop, k=10)
print("\nRL Model Fairness Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

with open('results/rl_fairness.json', 'w') as f:
    json.dump({'model': 'RL-CF', **metrics}, f, indent=2)

# ── Performance evaluation (same as BPR) ─────────────────────────────
print("\nEvaluating RL Performance Metrics...")

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

# Load test data
test_df = pd.read_csv('data/test.csv')
user_groups = test_df.groupby('user_id')['item_id'].apply(list)

ndcg10, ndcg20 = [], []
recall10, recall20 = [], []
precision10 = []

for uid, relevant_items in user_groups.items():
    uid = str(uid)
    if uid not in recs:
        continue

    recommended = recs[uid]

    ndcg10.append(ndcg_at_k(recommended, relevant_items, 10))
    ndcg20.append(ndcg_at_k(recommended, relevant_items, 20))
    recall10.append(recall_at_k(recommended, relevant_items, 10))
    recall20.append(recall_at_k(recommended, relevant_items, 20))
    precision10.append(precision_at_k(recommended, relevant_items, 10))

print("\nRL Performance Metrics:")
print(f"NDCG@10: {np.mean(ndcg10):.4f}")
print(f"NDCG@20: {np.mean(ndcg20):.4f}")
print(f"Precision@10: {np.mean(precision10):.4f}")
print(f"Recall@10: {np.mean(recall10):.4f}")
print(f"Recall@20: {np.mean(recall20):.4f}")

# Save results
perf_results = {
    "model": "RL-CF",
    "NDCG@10": float(np.mean(ndcg10)),
    "NDCG@20": float(np.mean(ndcg20)),
    "Precision@10": float(np.mean(precision10)),
    "Recall@10": float(np.mean(recall10)),
    "Recall@20": float(np.mean(recall20))
}

with open('results/rl_performance.json', 'w') as f:
    json.dump(perf_results, f, indent=2)