import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json, os
from collections import deque

from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

os.makedirs('results', exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────
CFG = {
    'emb_dim':          64,
    'hidden_dim':       256,
    'lr':               3e-4,
    'gamma':            0.99,       # discount factor
    'fairness_lambda':  0.1,        # fairness penalty weight in reward
    'n_episodes':       5000,
    'max_steps':        20,
    'log_every':        100,
    'save_every':       500,
    'window':           10,         # history window for state
}

# ── Init environment and policy ───────────────────────────────────────
env = RecEnv(
    train_path='data/train.csv',
    meta_path='data/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window'],
    fairness_lambda=CFG['fairness_lambda']
)

policy = ActorCriticPolicy(
    state_dim=CFG['emb_dim'],
    n_items=env.n_items,
    hidden_dim=CFG['hidden_dim']
)
optimizer = optim.Adam(policy.parameters(), lr=CFG['lr'])

# ── Training loop (REINFORCE with baseline) ───────────────────────────
episode_rewards = []
recent_rewards  = deque(maxlen=100)

print("Starting RL training...")
print(f"Users: {env.n_users}  |  Items: {env.n_items}")

for episode in range(CFG['n_episodes']):

    # Sample a random user each episode
    user_id = np.random.randint(0, env.n_users)
    state   = env.reset(user_id)

    log_probs, values, rewards = [], [], []
    seen_items = list(env.user_history[user_id])

    for step in range(CFG['max_steps']):
        action, log_prob, value = policy.select_action(state, exclude_items=seen_items)
        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        seen_items.append(action)
        state = next_state

        if done:
            break

    # ── Compute discounted returns ────────────────────────────────────
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + CFG['gamma'] * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

    # ── Compute losses ────────────────────────────────────────────────
    log_probs_t = torch.FloatTensor(log_probs)
    values_t    = torch.FloatTensor(values)
    advantages  = returns - values_t.detach()

    actor_loss  = -(log_probs_t * advantages).mean()
    critic_loss = nn.MSELoss()(values_t, returns)
    loss        = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()

    # ── Logging ───────────────────────────────────────────────────────
    ep_reward = sum(rewards)
    episode_rewards.append(ep_reward)
    recent_rewards.append(ep_reward)

    if (episode + 1) % CFG['log_every'] == 0:
        avg = np.mean(recent_rewards)
        print(f"Episode {episode+1:>5} | "
              f"Avg reward (last 100): {avg:.4f} | "
              f"Loss: {loss.item():.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────
    if (episode + 1) % CFG['save_every'] == 0:
        path = f"results/policy_ep{episode+1}.pt"
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

# ── Final save ────────────────────────────────────────────────────────
torch.save(policy.state_dict(), 'results/policy_final.pt')
np.save('results/episode_rewards.npy', np.array(episode_rewards))
print("\nTraining complete. Model saved to results/policy_final.pt")

# ── Quick fairness evaluation after training ──────────────────────────
from metrics.fairness_metrics import compute_all
import pandas as pd

print("\nEvaluating fairness on test users...")
test   = pd.read_csv('data/test.csv')
policy.eval()
recs   = {}

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