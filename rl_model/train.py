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

# ── Config ────────────────────────────────────────────────────────────
CFG = {
    'emb_dim':         64,
    'hidden_dim':      256,
    'lr':              3e-4,
    'gamma':           0.99,
    'fairness_lambda': 0.1,
    'n_episodes':      5000,
    'max_steps':       20,
    'log_every':       100,
    'save_every':      500,
    'window':          10,
    'entropy_coef':    0.01,   # entropy bonus — keeps policy from collapsing
}

# ── Environment ───────────────────────────────────────────────────────
env = RecEnv(
    train_path='data/train.csv',
    meta_path='data/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window'],
    fairness_lambda=CFG['fairness_lambda']
)
# Load pretrained BPR embeddings — gives GRU real signal
emb_path = 'data/bpr_item_embeddings.npy'
if os.path.exists(emb_path):
    env.load_pretrained_embeddings(np.load(emb_path))
else:
    print("WARNING: BPR embeddings not found — using random embeddings")
# ── Policy ────────────────────────────────────────────────────────────
# Fix 1: use emb_dim not state_dim
policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=env.n_items,
    hidden_dim=CFG['hidden_dim']
)
optimizer = optim.Adam(policy.parameters(), lr=CFG['lr'])

episode_rewards = []
recent_rewards  = deque(maxlen=100)

print("Starting RL training...")
print(f"Users: {env.n_users}  |  Items: {env.n_items}")
print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

# ── Helper: get item id sequence for current user ─────────────────────
def get_item_seq(user_id: int) -> np.ndarray:
    """
    Returns last `window` item ids for a user as a numpy int array.
    Shape: (window,)
    Padded with 0 at the front for short histories.
    """
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad    = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

# ── Training loop ─────────────────────────────────────────────────────
for episode in range(CFG['n_episodes']):

    user_id  = np.random.randint(0, env.n_users)
    env.reset(user_id)

    log_probs, values, rewards = [], [], []

    for step in range(CFG['max_steps']):
        # Fix 2: get item id sequence — policy takes ids not embeddings
        item_seq = get_item_seq(user_id)
        excluded = env.get_excluded_items()

        # Fix 3: pass item_exposure to select_action
        action, log_prob, value = policy.select_action(
            item_seq, env.item_exposure, exclude_items=excluded)

        _, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        if done:
            break

    # ── Discounted returns ────────────────────────────────────────────
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + CFG['gamma'] * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)

    # Fix 4: safe normalization — std() is NaN for single-element tensor
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    else:
        returns = returns - returns.mean()

    # ── Stack tensors ─────────────────────────────────────────────────
    log_probs_t = torch.stack(log_probs)
    values_t    = torch.stack(values)
    advantages  = returns - values_t.detach()
    advantages  = torch.clamp(advantages, -5.0, 5.0)  # prevent extreme swings

    # ── Losses ────────────────────────────────────────────────────────
    actor_loss  = -(log_probs_t * advantages).mean()
    critic_loss = nn.MSELoss()(values_t, returns)

    # Entropy bonus — prevents policy collapsing to one item
    item_seq_t  = torch.LongTensor(get_item_seq(user_id)).unsqueeze(0)
    exp_t       = torch.FloatTensor(env.item_exposure)
    logits, _, _, _ = policy.forward(item_seq_t, exp_t)
    probs       = torch.softmax(logits, dim=-1)
    entropy     = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()

    loss = actor_loss + 0.5 * critic_loss - CFG['entropy_coef'] * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()

    ep_reward = sum(rewards)
    episode_rewards.append(ep_reward)
    recent_rewards.append(ep_reward)

    if (episode + 1) % CFG['log_every'] == 0:
        avg = np.mean(recent_rewards)
        print(f"Episode {episode+1:>5} | "
              f"Avg reward: {avg:.4f} | "
              f"Loss: {loss.item():.4f} | "
              f"Entropy: {entropy.item():.4f}")

    if (episode + 1) % CFG['save_every'] == 0:
        path = f"results/policy_ep{episode+1}.pt"
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

# ── Final save ────────────────────────────────────────────────────────
torch.save(policy.state_dict(), 'results/policy_final.pt')
np.save('results/episode_rewards.npy', np.array(episode_rewards))
print("\nTraining complete. Model saved to results/policy_final.pt")

# ── Evaluation ────────────────────────────────────────────────────────
from metrics.fairness_metrics import compute_all
import pandas as pd

print("\nEvaluating on test users...")
test    = pd.read_csv('data/test.csv')
policy.eval()
recs    = {}

with torch.no_grad():
    for uid in test['user_id'].unique():
        env.reset(int(uid))
        topk = []
        for _ in range(20):          # collect 20 so NDCG@20 is valid
            item_seq = get_item_seq(int(uid))
            action   = policy.greedy_action(
                item_seq, env.item_exposure,
                exclude_items=env.get_excluded_items())
            topk.append(action)
            env._session_history.append(action)
        recs[str(uid)] = topk

policy.train()   # restore training mode

# ── Fairness metrics ──────────────────────────────────────────────────
item_pop = np.bincount(
    pd.read_csv('data/train.csv')['item_id'].values,
    minlength=env.n_items
).astype(float)

metrics = compute_all(recs, env.n_items, item_pop, k=10)
print("\nRL Fairness Metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v}")

with open('results/rl_fairness.json', 'w') as f:
    json.dump({'model': 'RL-CF', **metrics}, f, indent=2)

# ── Performance metrics ───────────────────────────────────────────────
print("\nRL Performance Metrics...")

def ndcg_at_k(recommended, relevant, k):
    dcg  = sum(1 / np.log2(i + 2)
               for i, item in enumerate(recommended[:k])
               if item in set(relevant))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0: return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)

def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k

test_df     = pd.read_csv('data/test.csv')
user_groups = test_df.groupby('user_id')['item_id'].apply(list)

ndcg10, ndcg20      = [], []
recall10, recall20  = [], []
precision10         = []

for uid, relevant_items in user_groups.items():
    uid_str = str(uid)
    if uid_str not in recs:
        continue
    recommended = recs[uid_str]
    ndcg10.append(    ndcg_at_k(recommended, relevant_items, 10))
    ndcg20.append(    ndcg_at_k(recommended, relevant_items, 20))
    recall10.append(  recall_at_k(recommended, relevant_items, 10))
    recall20.append(  recall_at_k(recommended, relevant_items, 20))
    precision10.append(precision_at_k(recommended, relevant_items, 10))

print(f"\nRL Performance Metrics:")
print(f"  NDCG@10      : {np.mean(ndcg10):.4f}")
print(f"  NDCG@20      : {np.mean(ndcg20):.4f}")
print(f"  Precision@10 : {np.mean(precision10):.4f}")
print(f"  Recall@10    : {np.mean(recall10):.4f}")
print(f"  Recall@20    : {np.mean(recall20):.4f}")

perf_results = {
    "model":        "RL-CF",
    "NDCG@10":      round(float(np.mean(ndcg10)),      4),
    "NDCG@20":      round(float(np.mean(ndcg20)),      4),
    "Precision@10": round(float(np.mean(precision10)), 4),
    "Recall@10":    round(float(np.mean(recall10)),    4),
    "Recall@20":    round(float(np.mean(recall20)),    4),
}
with open('results/rl_performance.json', 'w') as f:
    json.dump(perf_results, f, indent=2)

print("\nAll results saved to results/")