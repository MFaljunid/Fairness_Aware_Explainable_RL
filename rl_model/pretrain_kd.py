import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import json
import pickle
import glob
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

print("=" * 55)
print("Phase 1: Supervised Pre-training with BPR Distillation")
print("=" * 55)

DATA_DIR = 'data/ml-1m'

CFG = {
    'emb_dim':    128,
    'hidden_dim': 512,
    'window':     10,
    'lr':         1e-3,
    'epochs':     50,
    'batch_size': 256,
    'temperature': 2.0,   # KD temperature
    'alpha':       0.5,   # balance BPR loss vs KD loss
}

# ── Load data ──────────────────────────────────────────────────────────
train   = pd.read_csv(f'{DATA_DIR}/train.csv')
meta    = json.load(open(f'{DATA_DIR}/meta.json'))
N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

# ── Load BPR teacher model ─────────────────────────────────────────────
pkl_files = glob.glob('results/BPR/*.pkl')
assert len(pkl_files) > 0, "Run bpr_baseline.py first"
with open(sorted(pkl_files)[-1], 'rb') as f:
    bpr_teacher = pickle.load(f)

bpr_u2c = {int(k): v for k, v in bpr_teacher.uid_map.items()}
bpr_i2c = {int(k): v for k, v in bpr_teacher.iid_map.items()}

print(f"BPR teacher loaded: {len(bpr_u2c)} users, {len(bpr_i2c)} items")

# ── Environment ────────────────────────────────────────────────────────
env = RecEnv(
    train_path=f'{DATA_DIR}/train.csv',
    meta_path=f'{DATA_DIR}/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window']
)

bpr_embeddings = np.load(f'{DATA_DIR}/bpr_item_embeddings.npy')
if bpr_embeddings.shape[0] < N_ITEMS:
    pad = np.zeros((N_ITEMS - bpr_embeddings.shape[0],
                    bpr_embeddings.shape[1]), dtype=np.float32)
    bpr_embeddings = np.vstack([bpr_embeddings, pad])
env.load_pretrained_embeddings(bpr_embeddings)

# ── Policy ─────────────────────────────────────────────────────────────
policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=N_ITEMS,
    hidden_dim=CFG['hidden_dim']
)
with torch.no_grad():
    bpr_t    = torch.FloatTensor(bpr_embeddings)
    norms    = bpr_t.norm(dim=1, keepdim=True) + 1e-9
    bpr_norm = bpr_t / norms
    policy.item_emb.weight.data.copy_(bpr_norm)
    policy.item_emb.weight.data[0].zero_()

optimizer = optim.Adam(policy.parameters(), lr=CFG['lr'])

# ── Build training data ────────────────────────────────────────────────
user_items = defaultdict(list)
for _, row in train.iterrows():
    user_items[int(row['user_id'])].append(int(row['item_id']))

def get_item_seq(user_id):
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad    = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(
        torch.sigmoid(pos_scores - neg_scores) + 1e-9).mean()

# ── Precompute ALL BPR scores once ────────────────────────────────────
print("Precomputing BPR scores for all users (one-time)...")
bpr_scores_cache = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)

for uid in range(N_USERS):
    if uid in bpr_u2c:
        try:
            raw = bpr_teacher.score(bpr_u2c[uid])
            for item in range(N_ITEMS):
                ci = bpr_i2c.get(item, -1)
                if 0 <= ci < len(raw):
                    bpr_scores_cache[uid, item] = float(raw[ci])
        except Exception:
            pass

print(f"BPR scores cached: {bpr_scores_cache.shape}")
bpr_scores_tensor = torch.FloatTensor(bpr_scores_cache)

# ── Pre-training with Knowledge Distillation ───────────────────────────
print(f"Training on {len(train)} interactions")
print(f"Epochs: {CFG['epochs']}  |  Batch: {CFG['batch_size']}")
print(f"KD temperature: {CFG['temperature']}  |  Alpha: {CFG['alpha']}")

pos_pairs = list(zip(
    train['user_id'].astype(int),
    train['item_id'].astype(int)
))

T = CFG['temperature']

for epoch in range(CFG['epochs']):
    np.random.shuffle(pos_pairs)
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(pos_pairs), CFG['batch_size']):
        batch = pos_pairs[i:i + CFG['batch_size']]
        if len(batch) < 2:
            continue

        uids      = []
        user_seqs = []
        pos_items = []
        neg_items = []

        for uid, pos_item in batch:
            uids.append(uid)
            seq = get_item_seq(uid)
            user_seqs.append(seq)
            pos_items.append(pos_item)

            seen = set(user_items[uid])
            neg  = np.random.randint(0, N_ITEMS)
            while neg in seen:
                neg = np.random.randint(0, N_ITEMS)
            neg_items.append(neg)

        seq_t = torch.LongTensor(np.array(user_seqs))
        exp_t = torch.zeros(N_ITEMS)
        pos_t = torch.LongTensor(pos_items)
        neg_t = torch.LongTensor(neg_items)

        logits, _, _, _ = policy.forward(seq_t, exp_t)

        # ── BPR pairwise loss ──────────────────────────────────────────
        pos_scores = logits.gather(1, pos_t.unsqueeze(1)).squeeze(1)
        neg_scores = logits.gather(1, neg_t.unsqueeze(1)).squeeze(1)
        loss_bpr   = bpr_loss(pos_scores, neg_scores)

        # ── Knowledge distillation loss ────────────────────────────────
        # Get BPR teacher scores
        teacher_scores = bpr_scores_tensor[uids]


        # Soft targets from teacher
        teacher_soft = F.softmax(teacher_scores / T, dim=-1)
        student_soft = F.log_softmax(logits / T, dim=-1)

        loss_kd = F.kl_div(student_soft, teacher_soft,
                            reduction='batchmean') * (T * T)

        # ── Combined loss ──────────────────────────────────────────────
        loss = CFG['alpha'] * loss_bpr + (1 - CFG['alpha']) * loss_kd

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:>3}/{CFG['epochs']} | Loss: {avg_loss:.4f}")

# ── Save ───────────────────────────────────────────────────────────────
os.makedirs('results/ml-1m', exist_ok=True)
torch.save(policy.state_dict(), 'results/ml-1m/policy_pretrained.pt')
print(f"\nSaved: results/ml-1m/policy_pretrained.pt")

# ── Quick evaluation ───────────────────────────────────────────────────
print("\nQuick 100-neg evaluation...")
test_df         = pd.read_csv(f'{DATA_DIR}/test.csv')
test_items_dict = dict(zip(
    test_df['user_id'].astype(int),
    test_df['item_id'].astype(int)
))

policy.eval()
hits         = []
np.random.seed(42)
sample_users = list(test_items_dict.keys())[:1000]

with torch.no_grad():
    for uid in sample_users:
        pos_item   = test_items_dict[uid]
        seen       = set(user_items[uid]) | {pos_item}
        pool       = [i for i in range(N_ITEMS) if i not in seen]
        if len(pool) < 99:
            continue
        neg_items  = np.random.choice(pool, size=99, replace=False).tolist()
        candidates = [pos_item] + neg_items
        seq_t      = torch.LongTensor(get_item_seq(uid)).unsqueeze(0)
        exp_t      = torch.zeros(N_ITEMS)
        logits, _, _, _ = policy.forward(seq_t, exp_t)
        scores     = logits.squeeze(0).numpy()
        ranked     = sorted(candidates, key=lambda x: scores[x], reverse=True)
        hits.append(1.0 if pos_item in ranked[:10] else 0.0)

print(f"HR@10 after KD pre-training: {np.mean(hits):.4f}")
print("\nReady for Phase 2: RL Fine-tuning")