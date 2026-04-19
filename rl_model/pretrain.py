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

CFG = {
    'emb_dim':    64,
    'hidden_dim': 256,
    'window':     10,
    'lr':         1e-3,
    'epochs':     50,
    'batch_size': 512,
    'neg_samples': 4,   # negative samples per positive
}

# ── Load data ──────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
meta  = json.load(open('data/meta.json'))
N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

# ── Environment ────────────────────────────────────────────────────────
env = RecEnv('data/train.csv', 'data/meta.json',
             emb_dim=CFG['emb_dim'], window=CFG['window'])

bpr_embeddings = np.load('data/bpr_item_embeddings.npy')
env.load_pretrained_embeddings(bpr_embeddings)

# ── Policy ─────────────────────────────────────────────────────────────
policy = ActorCriticPolicy(
    emb_dim=CFG['emb_dim'],
    n_items=N_ITEMS,
    hidden_dim=CFG['hidden_dim']
)

# Initialize from BPR
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

all_items = list(range(N_ITEMS))

def get_item_seq(user_id):
    history = env._gt_history[user_id]
    recent  = history[-CFG['window']:]
    if len(recent) < CFG['window']:
        pad = [0] * (CFG['window'] - len(recent))
        recent = pad + recent
    return np.array(recent, dtype=np.int64)

# ── BPR Loss ───────────────────────────────────────────────────────────
def bpr_loss(pos_scores, neg_scores):
    """
    BPR pairwise ranking loss.
    pos_scores should be higher than neg_scores.
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9).mean()

# ── Pre-training loop ──────────────────────────────────────────────────
print(f"Training on {len(train)} interactions")
print(f"Epochs: {CFG['epochs']}  |  Batch size: {CFG['batch_size']}")

# Build all positive (user, item) pairs
pos_pairs = list(zip(train['user_id'].astype(int), train['item_id'].astype(int)))

for epoch in range(CFG['epochs']):
    np.random.shuffle(pos_pairs)
    total_loss = 0.0
    n_batches  = 0

    for i in range(0, len(pos_pairs), CFG['batch_size']):
        batch = pos_pairs[i:i + CFG['batch_size']]
        if len(batch) < 2:
            continue

        # Build tensors
        user_seqs  = []
        pos_items  = []
        neg_items  = []

        for uid, pos_item in batch:
            seq = get_item_seq(uid)
            user_seqs.append(seq)
            pos_items.append(pos_item)

            # Sample negative item not in user history
            seen = set(user_items[uid])
            neg  = np.random.randint(0, N_ITEMS)
            while neg in seen:
                neg = np.random.randint(0, N_ITEMS)
            neg_items.append(neg)

        seq_t   = torch.LongTensor(np.array(user_seqs))   # (B, window)
        exp_t   = torch.zeros(N_ITEMS)                    # zero exposure during pretraining
        pos_t   = torch.LongTensor(pos_items)
        neg_t   = torch.LongTensor(neg_items)

        # Forward pass
        logits, _, _, _ = policy.forward(seq_t, exp_t)   # (B, N_ITEMS)

        # Get scores for positive and negative items
        pos_scores = logits.gather(1, pos_t.unsqueeze(1)).squeeze(1)  # (B,)
        neg_scores = logits.gather(1, neg_t.unsqueeze(1)).squeeze(1)  # (B,)

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

# ── Save pretrained model ──────────────────────────────────────────────
torch.save(policy.state_dict(), 'results/policy_pretrained.pt')
print(f"\nPre-trained model saved to results/policy_pretrained.pt")

# ── Quick evaluation ───────────────────────────────────────────────────
print("\nQuick 100-neg evaluation after pre-training...")

test_df    = pd.read_csv('data/test.csv')
test_items_dict = dict(zip(test_df['user_id'].astype(int),
                           test_df['item_id'].astype(int)))
policy.eval()

hits = []
np.random.seed(42)

with torch.no_grad():
    for uid, pos_item in list(test_items_dict.items())[:1000]:
        seen       = set(user_items[uid]) | {pos_item}
        pool       = [i for i in range(N_ITEMS) if i not in seen]
        neg_items  = np.random.choice(pool, size=99, replace=False).tolist()
        candidates = [pos_item] + neg_items

        seq_t      = torch.LongTensor(get_item_seq(uid)).unsqueeze(0)
        exp_t      = torch.zeros(N_ITEMS)
        logits, _, _, _ = policy.forward(seq_t, exp_t)
        scores     = logits.squeeze(0).numpy()

        ranked = sorted(candidates, key=lambda x: scores[x], reverse=True)
        hits.append(1.0 if pos_item in ranked[:10] else 0.0)

print(f"HR@10 after pre-training (1000 users): {np.mean(hits):.4f}")
print("\nReady for Phase 2: RL Fine-tuning")