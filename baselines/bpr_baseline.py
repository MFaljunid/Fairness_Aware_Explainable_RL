import pandas as pd
import numpy as np
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import BPR
from cornac.metrics import NDCG, Recall, Precision
import json, os
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.makedirs('results', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
meta  = json.load(open('data/meta.json'))

data = list(zip(
    train['user_id'].astype(str),
    train['item_id'].astype(str),
    train['feedback'].astype(float)
))

# ── Eval setup ────────────────────────────────────────────────────────
eval_method = RatioSplit(
    data,
    test_size=0.1,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    seed=42
)

# ── Model ─────────────────────────────────────────────────────────────
bpr = BPR(
    k=64,
    max_iter=200,
    learning_rate=0.01,
    lambda_reg=0.001,
    seed=42,
    verbose=True
)

# ── Run experiment ────────────────────────────────────────────────────
exp = cornac.Experiment(
    eval_method=eval_method,
    models=[bpr],
    metrics=[NDCG(k=10), NDCG(k=20), Recall(k=10), Recall(k=20), Precision(k=10)],
    user_based=True,
    save_dir='results/'
)
exp.run()

# ── Save top-K recommendations ────────────────────────────────────────
def get_topk_recs(model, eval_method, k=10):
    recs = {}
    for uid in eval_method.test_set.uid_map:
        u_idx = eval_method.test_set.uid_map[uid]
        scores = model.score(u_idx)
        topk   = np.argsort(scores)[::-1][:k]
        recs[uid] = topk.tolist()
    return recs

bpr_recs = get_topk_recs(bpr, eval_method)
with open('results/bpr_recs.json', 'w') as f:
    json.dump(bpr_recs, f)

# ── Compute and save metrics ──────────────────────────────────────────
from metrics.fairness_metrics import gini_coefficient, compute_exposure, coverage

exposure = compute_exposure(bpr_recs, meta['n_items'])
gini     = gini_coefficient(exposure)
cov      = coverage(bpr_recs, meta['n_items'])

print(f"\nBPR Fairness Metrics:")
print(f"  Gini coefficient : {gini:.4f}  (lower = fairer)")
print(f"  Catalog coverage : {cov:.4f}  (higher = more diverse)")

results = {'model': 'BPR', 'gini': gini, 'coverage': cov}
with open('results/bpr_fairness.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nBPR baseline complete. Results saved to results/")