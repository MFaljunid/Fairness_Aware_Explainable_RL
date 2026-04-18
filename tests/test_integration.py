import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

print("=" * 50)
print("Integration check — all 4 files together")
print("=" * 50)

CFG = {
    'emb_dim':   64,
    'hidden_dim': 256,
    'window':    10,
    'fairness_lambda': 0.1,
}

# ── 1. Environment ────────────────────────────────────────────────────
print("\n--- Check 1: Environment loads ---")
env = RecEnv(
    train_path='data/train.csv',
    meta_path='data/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window'],
    fairness_lambda=CFG['fairness_lambda']
)
assert env.n_users == 6040
assert env.n_items == 3416
print(f"PASS: env — {env.n_users} users | {env.n_items} items")

# ── 2. Policy ─────────────────────────────────────────────────────────
print("\n--- Check 2: Policy loads ---")
policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=env.n_items,
    hidden_dim=CFG['hidden_dim']
)
total = sum(p.numel() for p in policy.parameters())
print(f"PASS: policy — {total:,} parameters")

# ── 3. get_item_seq helper ────────────────────────────────────────────
print("\n--- Check 3: get_item_seq ---")
def get_item_seq(user_id):
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad    = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

seq = get_item_seq(0)
assert seq.shape == (CFG['window'],), f"Wrong shape: {seq.shape}"
assert seq.dtype == np.int64
print(f"PASS: item seq shape {seq.shape} dtype {seq.dtype}")

# ── 4. One full episode ───────────────────────────────────────────────
print("\n--- Check 4: One full training episode ---")
user_id = 0
env.reset(user_id)
log_probs, values, rewards = [], [], []

for step in range(20):
    item_seq = get_item_seq(user_id)
    excluded = env.get_excluded_items()
    action, log_prob, value = policy.select_action(
        item_seq, env.item_exposure, exclude_items=excluded)
    _, reward, done, info = env.step(action)
    log_probs.append(log_prob)
    values.append(value)
    rewards.append(reward)
    if done:
        break

assert len(rewards) == 20
assert all(r >= -CFG['fairness_lambda'] for r in rewards)
assert log_probs[0].requires_grad
assert values[0].requires_grad
print(f"PASS: episode ran {len(rewards)} steps")
print(f"      total reward : {sum(rewards):.4f}")
print(f"      log_prob grad: {log_probs[0].requires_grad}")
print(f"      value grad   : {values[0].requires_grad}")

# ── 5. Loss computation ───────────────────────────────────────────────
print("\n--- Check 5: Loss computes and backprops ---")
returns = []
G = 0.0
for r in reversed(rewards):
    G = r + 0.99 * G
    returns.insert(0, G)
returns = torch.FloatTensor(returns)
if len(returns) > 1:
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

log_probs_t = torch.stack(log_probs)
values_t    = torch.stack(values)
advantages  = torch.clamp(returns - values_t.detach(), -5.0, 5.0)

actor_loss  = -(log_probs_t * advantages).mean()
critic_loss = torch.nn.MSELoss()(values_t, returns)
loss        = actor_loss + 0.5 * critic_loss

loss.backward()
assert loss.item() == loss.item(), "Loss is NaN"
print(f"PASS: loss = {loss.item():.4f}  (not NaN)")

# ── 6. Greedy action for evaluation ──────────────────────────────────
print("\n--- Check 6: Greedy action ---")
env.reset(0)
item_seq = get_item_seq(0)
action   = policy.greedy_action(
    item_seq, env.item_exposure,
    exclude_items=env.get_excluded_items())
assert isinstance(action, int)
assert 0 <= action < env.n_items
print(f"PASS: greedy action = {action}")

# ── 7. Explain ────────────────────────────────────────────────────────
print("\n--- Check 7: Explain ---")
exp = policy.explain(item_seq, env.item_exposure, action)
assert 'attention'      in exp
assert 'saliency'       in exp
assert 'counterfactual' in exp
print(f"PASS: explanation generated")
print(f"      {exp['attention']['explanation']}")
print(f"      {exp['counterfactual']['explanation']}")

# ── 8. Fairness metrics ───────────────────────────────────────────────
print("\n--- Check 8: Fairness metrics ---")
from metrics.fairness_metrics import compute_all
dummy_recs = {str(i): list(range(10)) for i in range(100)}
metrics    = compute_all(dummy_recs, env.n_items)
assert 'gini'     in metrics
assert 'coverage' in metrics
print(f"PASS: fairness metrics — {list(metrics.keys())}")

print("\n" + "=" * 50)
print("All integration checks passed.")
print("Ready to run: python rl_model/train.py")
print("=" * 50)