import pandas as pd
import numpy as np
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import LightGCN
from cornac.metrics import NDCG, Recall, Precision
import json, os

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
lightgcn = LightGCN(
    num_epochs=1000,
    learning_rate=0.001,
    emb_size=64,       # same as BPR for fair comparison
    num_layers=3,
    lambda_reg=1e-4,
    batch_size=1024,
    seed=42,
    verbose=True
)

# ── Run experiment ────────────────────────────────────────────────────
exp = cornac.Experiment(
    eval_method=eval_method,
    models=[lightgcn],
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

lgcn_recs = get_topk_recs(lightgcn, eval_method)
with open('results/lightgcn_recs.json', 'w') as f:
    json.dump(lgcn_recs, f)

# ── Fairness metrics ──────────────────────────────────────────────────
from metrics.fairness_metrics import gini_coefficient, compute_exposure, coverage

exposure = compute_exposure(lgcn_recs, meta['n_items'])
gini     = gini_coefficient(exposure)
cov      = coverage(lgcn_recs, meta['n_items'])

print(f"\nLightGCN Fairness Metrics:")
print(f"  Gini coefficient : {gini:.4f}")
print(f"  Catalog coverage : {cov:.4f}")

results = {'model': 'LightGCN', 'gini': gini, 'coverage': cov}
with open('results/lightgcn_fairness.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nLightGCN baseline complete. Results saved to results/")