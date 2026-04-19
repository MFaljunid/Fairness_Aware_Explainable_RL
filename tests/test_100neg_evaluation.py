import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import pandas as pd
import json
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

print("=" * 55)
print("100-Negative Sampling Evaluation")
print("Same protocol as most CF papers")
print("=" * 55)

# ── Config ─────────────────────────────────────────────────────────────
CFG = {
    'emb_dim':    64,
    'hidden_dim': 256,
    'window':     10,
    'fairness_lambda': 0.1,
    'n_negatives': 99,    # 99 negatives + 1 positive = 100 candidates
    'k_list':     [5, 10, 20],
}

# ── Load data ──────────────────────────────────────────────────────────
train  = pd.read_csv('data/train.csv')
test   = pd.read_csv('data/test.csv')
meta   = json.load(open('data/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

# ── Build interaction sets ─────────────────────────────────────────────
# Used to sample TRUE negatives — items user has never interacted with
train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

print(f"\nUsers: {N_USERS}  |  Items: {N_ITEMS}")
print(f"Candidates per user: {CFG['n_negatives'] + 1} "
      f"(1 positive + {CFG['n_negatives']} negatives)")

# ── Load environment and policy ────────────────────────────────────────
env = RecEnv(
    train_path='data/train.csv',
    meta_path='data/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window'],
    fairness_lambda=CFG['fairness_lambda']
)

# Load BPR embeddings if available
emb_path = 'data/bpr_item_embeddings.npy'
if os.path.exists(emb_path):
    env.load_pretrained_embeddings(np.load(emb_path))
    print("Loaded BPR embeddings")
else:
    print("WARNING: using random embeddings")

policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=N_ITEMS,
    hidden_dim=CFG['hidden_dim']
)

# Load trained model
model_path = 'results/policy_final.pt'
assert os.path.exists(model_path), \
    f"No trained model found at {model_path} — run train.py first"
policy.load_state_dict(torch.load(model_path, map_location='cpu'))
policy.eval()
print(f"Loaded model from {model_path}")

# ── Helper ─────────────────────────────────────────────────────────────
def get_item_seq(user_id: int) -> np.ndarray:
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad    = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

# ── Metric functions ───────────────────────────────────────────────────
def hit_at_k(ranked_items, positive_item, k):
    """1 if positive item is in top-k, else 0."""
    return 1.0 if positive_item in ranked_items[:k] else 0.0

def ndcg_at_k(ranked_items, positive_item, k):
    """NDCG with single positive item."""
    if positive_item in ranked_items[:k]:
        rank = ranked_items[:k].index(positive_item)
        return 1.0 / np.log2(rank + 2)
    return 0.0

def mrr(ranked_items, positive_item):
    """Mean Reciprocal Rank."""
    if positive_item in ranked_items:
        rank = ranked_items.index(positive_item)
        return 1.0 / (rank + 1)
    return 0.0

# ── 100-Negative Sampling Evaluation ──────────────────────────────────
print("\nRunning evaluation...")

results_per_k = {k: {'hit': [], 'ndcg': []} for k in CFG['k_list']}
mrr_scores    = []
np.random.seed(42)   # reproducible negative sampling

with torch.no_grad():
    for uid, pos_item in test_items.items():

        # Sample 99 true negatives — items this user never interacted with
        all_items       = set(range(N_ITEMS))
        seen_items      = train_set[uid] | {pos_item}
        candidate_pool  = list(all_items - seen_items)

        # Sample exactly n_negatives
        neg_items = np.random.choice(
            candidate_pool,
            size=min(CFG['n_negatives'], len(candidate_pool)),
            replace=False
        ).tolist()

        # Candidate set: 1 positive + 99 negatives
        candidates = [pos_item] + neg_items   # length = 100

        # Get model scores for all 100 candidates
        item_seq  = get_item_seq(uid)
        seq_t     = torch.LongTensor(item_seq).unsqueeze(0)
        exp_t = torch.zeros(N_ITEMS)

        logits, _, _, _ = policy.forward(seq_t, exp_t)
        logits_np       = logits.squeeze(0).numpy()

        # Score only the 100 candidates
        candidate_scores = [(item, logits_np[item]) for item in candidates]

        # Rank by score descending
        ranked = [item for item, score in
                  sorted(candidate_scores, key=lambda x: x[1], reverse=True)]

        # Compute metrics
        mrr_scores.append(mrr(ranked, pos_item))

        for k in CFG['k_list']:
            results_per_k[k]['hit'].append( hit_at_k(ranked, pos_item, k))
            results_per_k[k]['ndcg'].append(ndcg_at_k(ranked, pos_item, k))

# ── Print results ──────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'Metric':<20} {'Value':>10}")
print("-" * 35)

for k in CFG['k_list']:
    hr   = np.mean(results_per_k[k]['hit'])
    ndcg = np.mean(results_per_k[k]['ndcg'])
    print(f"Hit Rate@{k:<11} {hr:>10.4f}")
    print(f"NDCG@{k:<15}    {ndcg:>10.4f}")
    print("-" * 35)

print(f"{'MRR':<20} {np.mean(mrr_scores):>10.4f}")
print("=" * 55)

# ── Compare with BPR if available ─────────────────────────────────────
print("\nContext — typical 100-neg results on MovieLens-1M:")
print("  BPR       NDCG@10 ~ 0.45–0.65")
print("  LightGCN  NDCG@10 ~ 0.55–0.75")
print("  Your RL   NDCG@10 = see above")

# ── Save results ───────────────────────────────────────────────────────
save_results = {
    'protocol':   '100-negative-sampling',
    'n_users':    N_USERS,
    'n_negatives': CFG['n_negatives'],
}
for k in CFG['k_list']:
    save_results[f'HR@{k}']   = round(float(np.mean(results_per_k[k]['hit'])),  4)
    save_results[f'NDCG@{k}'] = round(float(np.mean(results_per_k[k]['ndcg'])), 4)
save_results['MRR'] = round(float(np.mean(mrr_scores)), 4)

with open('results/rl_100neg_evaluation.json', 'w') as f:
    json.dump(save_results, f, indent=2)

print(f"\nResults saved to results/rl_100neg_evaluation.json")