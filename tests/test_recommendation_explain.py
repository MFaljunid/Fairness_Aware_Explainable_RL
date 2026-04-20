import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pandas as pd
import json
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

print("=" * 60)
print("Live Recommendation + Explanation Demo")
print("=" * 60)

CFG = {'emb_dim': 64, 'hidden_dim': 256, 'window': 10,
       'fairness_lambda': 0.1}

# ── Load movies.dat for real movie names ──────────────────────────────
movies = {}
with open('data/movies.dat', 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('::')
        if len(parts) >= 2:
            movies[int(parts[0])] = parts[1]   # id → title

meta     = json.load(open('data/meta.json'))
item2idx = {int(k): int(v) for k, v in meta['item2idx'].items()}
idx2item = {v: k for k, v in item2idx.items()}   # new idx → original id

def get_movie_name(item_idx):
    orig_id = idx2item.get(item_idx, -1)
    return movies.get(orig_id, f"Movie_{item_idx}")

# ── Load environment ───────────────────────────────────────────────────
env = RecEnv('data/train.csv', 'data/meta.json',
             emb_dim=CFG['emb_dim'], window=CFG['window'],
             fairness_lambda=CFG['fairness_lambda'])

emb_path = 'data/bpr_item_embeddings.npy'
if os.path.exists(emb_path):
    env.load_pretrained_embeddings(np.load(emb_path))

# ── Load policy ────────────────────────────────────────────────────────
policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=env.n_items,
    hidden_dim=CFG['hidden_dim']
)

# Load best available model
for model_path in ['results/policy_final.pt',
                   'results/policy_pretrained.pt']:
    if os.path.exists(model_path):
        policy.load_state_dict(
            torch.load(model_path, map_location='cpu'))
        print(f"Loaded model: {model_path}")
        break

policy.eval()

# ── Helper ─────────────────────────────────────────────────────────────
def get_item_seq(user_id):
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad    = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

# ── Demo for 3 users ───────────────────────────────────────────────────
demo_users = [0, 100, 500]

for user_id in demo_users:
    print(f"\n{'='*60}")
    print(f"USER {user_id}")
    print(f"{'='*60}")

    # Show watch history
    history    = env._gt_history[user_id]
    recent_10  = history[-10:]
    print(f"\nRecent watch history (last 10):")
    for i, item_idx in enumerate(recent_10):
        print(f"  {i+1:2}. {get_movie_name(item_idx)}")

    # Get recommendation
    env.reset(user_id)
    item_seq = get_item_seq(user_id)

    with torch.no_grad():
        action = policy.greedy_action(
            item_seq,
            env.item_exposure,
            exclude_items=env.get_excluded_items()
        )

    print(f"\nRecommended movie:")
    print(f"  >>> {get_movie_name(action)} <<<")

    # Get explanation
    explanation = policy.explain(
        item_seq,
        env.item_exposure,
        action=action
    )

    # Attention explanation
    print(f"\nWhy this movie? (Attention explanation):")
    attn_items   = explanation['attention']['top_history_items']
    attn_weights = explanation['attention']['top_weights']
    for item_idx, weight in zip(attn_items[:3], attn_weights[:3]):
        print(f"  Because you watched: {get_movie_name(item_idx)}"
              f"  (influence: {weight:.3f})")

    # Saliency explanation
    saliency  = explanation['saliency']['scores']
    best_pos  = explanation['saliency']['most_important_pos']
    if best_pos < len(recent_10):
        print(f"\nMost influential history position:")
        print(f"  Position {best_pos} → {get_movie_name(recent_10[best_pos])}"
              f"  (saliency: {saliency[best_pos]:.4f})")

    # Counterfactual explanation
    alt_items = explanation['counterfactual']['alternatives']
    alt_probs = explanation['counterfactual']['alt_probs']
    print(f"\nCounterfactual (what if this movie was not available?):")
    print(f"  Next best: {get_movie_name(alt_items[0])}"
          f"  (prob: {alt_probs[0]:.4f})")
    print(f"  Then:      {get_movie_name(alt_items[1])}"
          f"  (prob: {alt_probs[1]:.4f})")

    print(f"\nChosen movie probability: {explanation['chosen_prob']:.6f}")

print(f"\n{'='*60}")
print("Demo complete.")
print(f"{'='*60}")