import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
import pickle
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy
from metrics.fairness_metrics import compute_exposure, gini_coefficient, coverage
from metrics.user_fairness_metrics import load_user_gender, compute_dp_eo

os.makedirs('results/figures', exist_ok=True)

print("=" * 60)
print("FULL RANKING Evaluation — same as FairIR paper")
print("K = 10, 20, 30, 40")
print("=" * 60)

K_LIST  = [10, 20, 30, 40]
EMB_DIM = 64
HIDDEN  = 256
WINDOW  = 10

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
val   = pd.read_csv('data/val.csv')
test  = pd.read_csv('data/test.csv')
meta  = json.load(open('data/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

test_items = {}
for _, row in test.iterrows():
    test_items[int(row['user_id'])] = int(row['item_id'])

# Gender for DP/EO
user2idx    = {int(k): int(v) for k, v in meta['user2idx'].items()}
raw_gender  = load_user_gender('data/users.dat')
user_gender = {user2idx[u]: g for u, g in raw_gender.items()
               if u in user2idx}

# ── Helper functions ───────────────────────────────────────────────────
def hit_at_k(ranked, pos, k):
    return 1.0 if pos in ranked[:k] else 0.0

def ndcg_at_k(ranked, pos, k):
    if pos in ranked[:k]:
        return 1.0 / np.log2(ranked[:k].index(pos) + 2)
    return 0.0

def compute_all_metrics(recs_dict, k):
    recs_k   = {uid: items[:k] for uid, items in recs_dict.items()}
    exposure = compute_exposure(recs_k, N_ITEMS, k)
    gini     = gini_coefficient(exposure)
    cov      = coverage(recs_k, N_ITEMS, k)
    fairness = compute_dp_eo(recs_k, user_gender, test_items, k)
    return {
        'Gini':     round(gini, 4),
        'Coverage': round(cov,  4),
        'DP':       round(fairness['DP'], 4),
        'EO':       round(fairness['EO'], 4),
    }

ALL_RESULTS = {}

# ══════════════════════════════════════════════════════════════════════
# MODEL 1: BPR — full ranking
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating BPR (full ranking) ---")
pkl_files = glob.glob('results/BPR/*.pkl')
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr = pickle.load(f)

bpr_u2c = {int(k): v for k, v in bpr.uid_map.items()}
bpr_i2c = {int(k): v for k, v in bpr.iid_map.items()}

bpr_recs    = {}
bpr_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

for uid, pos_item in test_items.items():
    if uid not in bpr_u2c:
        continue

    cornac_uid = bpr_u2c[uid]
    try:
        scores = bpr.score(cornac_uid)
    except Exception:
        continue

    # Full ranking — score ALL items except seen ones
    seen = train_set[uid]

    # Get scores for all items
    all_scores = []
    for item in range(N_ITEMS):
        if item in seen:
            continue
        cornac_item = bpr_i2c.get(item, -1)
        if 0 <= cornac_item < len(scores):
            all_scores.append((item, float(scores[cornac_item])))
        else:
            all_scores.append((item, 0.0))

    ranked = [item for item, score in
              sorted(all_scores, key=lambda x: x[1], reverse=True)]

    bpr_recs[str(uid)] = ranked

    for k in K_LIST:
        bpr_results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
        bpr_results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

bpr_final = {}
print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("-" * 55)
for k in K_LIST:
    hr   = np.mean(bpr_results[k]['hits'])
    ndcg = np.mean(bpr_results[k]['ndcgs'])
    fair = compute_all_metrics(bpr_recs, k)
    bpr_final[k] = {'HR': round(hr,4), 'NDCG': round(ndcg,4), **fair}
    r = bpr_final[k]
    print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
ALL_RESULTS['BPR'] = bpr_final

# ══════════════════════════════════════════════════════════════════════
# MODEL 2: LightGCN — full ranking
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating LightGCN (full ranking) ---")
lgcn_pkl = glob.glob('results/LightGCN/*.pkl')

if lgcn_pkl:
    with open(sorted(lgcn_pkl)[-1], 'rb') as f:
        lgcn = pickle.load(f)

    lgcn_u2c = {int(k): v for k, v in lgcn.uid_map.items()}
    lgcn_i2c = {int(k): v for k, v in lgcn.iid_map.items()}

    lgcn_recs    = {}
    lgcn_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

    for uid, pos_item in test_items.items():
        if uid not in lgcn_u2c:
            continue
        cornac_uid = lgcn_u2c[uid]
        try:
            scores = lgcn.score(cornac_uid)
        except Exception:
            continue

        seen = train_set[uid] | val_set[uid]
        all_scores = []
        for item in range(N_ITEMS):
            if item in seen:
                continue
            cornac_item = lgcn_i2c.get(item, -1)
            if 0 <= cornac_item < len(scores):
                all_scores.append((item, float(scores[cornac_item])))
            else:
                all_scores.append((item, 0.0))

        ranked = [item for item, score in
                  sorted(all_scores, key=lambda x: x[1], reverse=True)]

        lgcn_recs[str(uid)] = ranked
        for k in K_LIST:
            lgcn_results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
            lgcn_results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

    lgcn_final = {}
    print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
          f"{'Gini':>7} {'Cov':>7}")
    print("-" * 55)
    for k in K_LIST:
        hr   = np.mean(lgcn_results[k]['hits'])
        ndcg = np.mean(lgcn_results[k]['ndcgs'])
        fair = compute_all_metrics(lgcn_recs, k)
        lgcn_final[k] = {'HR': round(hr,4), 'NDCG': round(ndcg,4), **fair}
        r = lgcn_final[k]
        print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
              f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
              f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
    ALL_RESULTS['LightGCN'] = lgcn_final

# ══════════════════════════════════════════════════════════════════════
# MODEL 3: Your RL model — full ranking
# ══════════════════════════════════════════════════════════════════════
print("\n--- Evaluating Your RL Model (full ranking) ---")

env = RecEnv('data/train.csv', 'data/meta.json',
             emb_dim=EMB_DIM, window=WINDOW)
if os.path.exists('data/bpr_item_embeddings.npy'):
    env.load_pretrained_embeddings(np.load('data/bpr_item_embeddings.npy'))

policy = ActorCriticPolicy(emb_dim=EMB_DIM, n_items=N_ITEMS,
                            hidden_dim=HIDDEN)
policy.load_state_dict(
    torch.load('results/policy_final.pt', map_location='cpu'))
policy.eval()

def get_item_seq(uid):
    history = env._gt_history[uid]
    recent  = history[-WINDOW:]
    if len(recent) < WINDOW:
        pad    = [0] * (WINDOW - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

rl_recs    = {}
rl_results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

with torch.no_grad():
    for uid, pos_item in test_items.items():
        seen  = train_set[uid] | val_set[uid]
        seq_t = torch.LongTensor(get_item_seq(uid)).unsqueeze(0)
        exp_t = torch.zeros(N_ITEMS)
        logits, _, _, _ = policy.forward(seq_t, exp_t)
        scores = logits.squeeze(0).numpy()

        # Full ranking — all unseen items
        all_scores = [(item, float(scores[item]))
                      for item in range(N_ITEMS)
                      if item not in seen]

        ranked = [item for item, score in
                  sorted(all_scores, key=lambda x: x[1], reverse=True)]

        rl_recs[str(uid)] = ranked
        for k in K_LIST:
            rl_results[k]['hits'].append(hit_at_k(ranked, pos_item, k))
            rl_results[k]['ndcgs'].append(ndcg_at_k(ranked, pos_item, k))

rl_final = {}
print(f"{'K':<5} {'HR':>7} {'NDCG':>7} {'DP':>7} {'EO':>7} "
      f"{'Gini':>7} {'Cov':>7}")
print("-" * 55)
for k in K_LIST:
    hr   = np.mean(rl_results[k]['hits'])
    ndcg = np.mean(rl_results[k]['ndcgs'])
    fair = compute_all_metrics(rl_recs, k)
    rl_final[k] = {'HR': round(hr,4), 'NDCG': round(ndcg,4), **fair}
    r = rl_final[k]
    print(f"K={k:<3} {r['HR']:>7.4f} {r['NDCG']:>7.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
ALL_RESULTS['Our RL'] = rl_final

# ── Save ───────────────────────────────────────────────────────────────
with open('results/all_models_fullranking.json', 'w') as f:
    json.dump({m: {str(k): v for k, v in res.items()}
               for m, res in ALL_RESULTS.items()},
              f, indent=2)

# ── Summary table ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL TABLE — Full Ranking (same protocol as FairIR paper)")
print("=" * 65)
print(f"{'Model':<12} {'HR@10':>7} {'NDCG@10':>9} "
      f"{'DP@10':>7} {'EO@10':>7} {'Gini':>7} {'Cov':>7}")
print("-" * 65)
for model, results in ALL_RESULTS.items():
    r = results[10]
    print(f"{model:<12} {r['HR']:>7.4f} {r['NDCG']:>9.4f} "
          f"{r['DP']:>7.4f} {r['EO']:>7.4f} "
          f"{r['Gini']:>7.4f} {r['Coverage']:>7.4f}")
print("=" * 65)

# ── Trade-off figure ───────────────────────────────────────────────────
COLORS  = {'BPR': '#4C72B0', 'LightGCN': '#55A868', 'Our RL': '#E8A838'}
MARKERS = {'BPR': 'o',       'LightGCN': 's',        'Our RL': '*'}
SIZES   = {'BPR': 80,        'LightGCN': 80,          'Our RL': 200}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Accuracy-Fairness Trade-off — MovieLens-1M (Full Ranking)\n'
             'Points at K = 10, 20, 30, 40',
             fontsize=13, fontweight='bold')

configs = [
    ('HR',   'DP',   'HR (higher=better)',   'DP (lower=better)',   'DP vs HR'),
    ('NDCG', 'EO',   'NDCG (higher=better)', 'EO (lower=better)',   'EO vs NDCG'),
]

for ax, (xm, ym, xlabel, ylabel, title) in zip(axes, configs):
    for model, results in ALL_RESULTS.items():
        xs = [results[k][xm] for k in K_LIST]
        ys = [results[k][ym] for k in K_LIST]

        ax.plot(xs, ys, color=COLORS[model],
                linewidth=2, alpha=0.8, zorder=3)
        ax.scatter(xs, ys, color=COLORS[model],
                   s=SIZES[model], marker=MARKERS[model],
                   zorder=5, edgecolors='black', linewidth=0.5)

        for k, x, y in zip(K_LIST, xs, ys):
            ax.annotate(f'K={k}', xy=(x, y),
                        xytext=(6, 4),
                        textcoords='offset points',
                        fontsize=8, color=COLORS[model],
                        fontweight='bold')

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

handles = [
    mlines.Line2D([], [], color=COLORS[m],
                  marker=MARKERS[m],
                  markersize=10 if m == 'Our RL' else 7,
                  linewidth=2, label=m)
    for m in ALL_RESULTS.keys()
]
fig.legend(handles=handles, loc='lower center',
           ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.06))

plt.tight_layout()
plt.savefig('results/figures/tradeoff_fullranking.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: results/figures/tradeoff_fullranking.png")
print("Done.")