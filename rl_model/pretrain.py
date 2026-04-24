import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import json
from collections import defaultdict
from rl_model.environment import RecEnv
from rl_model.policy import ActorCriticPolicy

print("=" * 50)
print("Phase 1: Supervised Pre-training")
print("=" * 50)

DATA_DIR = 'data/ml-1m'

CFG = {
    'emb_dim':    64,
    'hidden_dim': 256,
    'window':     10,
    'lr':         1e-3,
    'epochs':     50,
    'batch_size': 512,
}

# ── Load data ──────────────────────────────────────────────────────────
train   = pd.read_csv(f'{DATA_DIR}/train.csv')
meta    = json.load(open(f'{DATA_DIR}/meta.json'))
N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

# ── Environment ────────────────────────────────────────────────────────
env = RecEnv(
    train_path=f'{DATA_DIR}/train.csv',
    meta_path=f'{DATA_DIR}/meta.json',
    emb_dim=CFG['emb_dim'],
    window=CFG['window']
)

bpr_embeddings = np.load(f'{DATA_DIR}/bpr_item_embeddings.npy')
if bpr_embeddings.shape[0] < N_ITEMS:
    print(f"Padding embeddings from {bpr_embeddings.shape[0]} to {N_ITEMS}")
    pad = np.zeros((N_ITEMS - bpr_embeddings.shape[0],
                    bpr_embeddings.shape[1]), dtype=np.float32)
    bpr_embeddings = np.vstack([bpr_embeddings, pad])
    print(f"Embeddings padded to {bpr_embeddings.shape}")
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

# ── Pre-training loop ──────────────────────────────────────────────────
print(f"Training on {len(train)} interactions")
print(f"Epochs: {CFG['epochs']}  |  Batch size: {CFG['batch_size']}")
print(f"Users: {N_USERS}  |  Items: {N_ITEMS}")

pos_pairs = list(zip(
    train['user_id'].astype(int),
    train['item_id'].astype(int)
))

for epoch in range(CFG['epochs']):
    np.random.shuffle(pos_pairs)
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(pos_pairs), CFG['batch_size']):
        batch = pos_pairs[i:i + CFG['batch_size']]
        if len(batch) < 2:
            continue

        user_seqs = []
        pos_items = []
        neg_items = []

        for uid, pos_item in batch:
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
        pos_scores = logits.gather(1, pos_t.unsqueeze(1)).squeeze(1)
        neg_scores = logits.gather(1, neg_t.unsqueeze(1)).squeeze(1)

        loss = bpr_loss(pos_scores, neg_scores)
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
os.makedirs('results', exist_ok=True)
torch.save(policy.state_dict(), 'results/policy_pretrained.pt')
print(f"\nPre-trained model saved to results/policy_pretrained.pt")

# ── Quick evaluation ───────────────────────────────────────────────────
print("\nQuick 100-neg evaluation after pre-training...")

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

        seq_t           = torch.LongTensor(get_item_seq(uid)).unsqueeze(0)
        exp_t           = torch.zeros(N_ITEMS)
        logits, _, _, _ = policy.forward(seq_t, exp_t)
        scores          = logits.squeeze(0).numpy()

        ranked = sorted(candidates, key=lambda x: scores[x], reverse=True)
        hits.append(1.0 if pos_item in ranked[:10] else 0.0)

print(f"HR@10 after pre-training (1000 users): {np.mean(hits):.4f}")
print("\nReady for Phase 2: RL Fine-tuning")