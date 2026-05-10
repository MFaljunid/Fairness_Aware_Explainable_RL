import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
from collections import defaultdict

DATA_DIR = 'data/ml-1m'
train   = pd.read_csv(f'{DATA_DIR}/train.csv')
val     = pd.read_csv(f'{DATA_DIR}/val.csv')
test    = pd.read_csv(f'{DATA_DIR}/test.csv')
meta    = json.load(open(f'{DATA_DIR}/meta.json'))
N_ITEMS = meta['n_items']
WINDOW  = 20

train_set = defaultdict(set)
for _, row in train.iterrows():
    train_set[int(row['user_id'])].add(int(row['item_id']))

val_set = defaultdict(set)
for _, row in val.iterrows():
    val_set[int(row['user_id'])].add(int(row['item_id']))

user_history = defaultdict(list)
for _, row in train.sort_values('timestamp').iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))

test_items = dict(zip(
    test['user_id'].astype(int),
    test['item_id'].astype(int)
))

class SASRec(nn.Module):
    def __init__(self, n_items, emb_dim, hidden_dim,
                 num_heads, num_layers, window, dropout):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(window + 1,  emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.layers   = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=num_heads,
                dim_feedforward=hidden_dim, dropout=dropout,
                batch_first=True, norm_first=True,
            ) for _ in range(num_layers)
        ])
        self.norm   = nn.LayerNorm(emb_dim)
        self.window = window

    def forward(self, seq):
        B, W = seq.shape
        positions = torch.arange(1, W+1, device=seq.device).unsqueeze(0)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(W, device=seq.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.norm(x)

    def predict(self, seq, item_indices):
        out   = self.forward(seq)
        state = out[:, -1, :]
        item_embs = self.item_emb(item_indices)
        return torch.bmm(item_embs, state.unsqueeze(-1)).squeeze(-1)

model = SASRec(N_ITEMS, 64, 256, 4, 2, WINDOW, 0.2)
model.load_state_dict(torch.load('results/SASRec/sasrec_best.pt',
                                  map_location='cpu'))
model.eval()

np.random.seed(42)
hits  = []
ndcgs = []

K_LIST = [5, 10, 20, 30, 40]
results = {k: {'hits': [], 'ndcgs': []} for k in K_LIST}

with torch.no_grad():
    for uid, pos_item in test_items.items():
        seen = train_set[uid] | val_set[uid] | {pos_item}
        pool = [i for i in range(1, N_ITEMS+1) if i not in seen]
        if len(pool) < 99:
            continue
        negs       = np.random.choice(pool, 99, replace=False).tolist()
        candidates = [pos_item] + negs

        history = user_history[uid][-WINDOW:]
        if len(history) < WINDOW:
            history = [0]*(WINDOW - len(history)) + history

        seq_t  = torch.LongTensor(history).unsqueeze(0)
        cand_t = torch.LongTensor(candidates).unsqueeze(0)
        scores = model.predict(seq_t, cand_t).squeeze(0)

        ranked = sorted(range(len(candidates)),
                        key=lambda i: scores[i].item(), reverse=True)
        ranked_items = [candidates[i] for i in ranked]

        for k in K_LIST:
            hit  = 1.0 if pos_item in ranked_items[:k] else 0.0
            if pos_item in ranked_items[:k]:
                ndcg = 1.0 / np.log2(ranked_items[:k].index(pos_item) + 2)
            else:
                ndcg = 0.0
            results[k]['hits'].append(hit)
            results[k]['ndcgs'].append(ndcg)

print(f"\nSASRec Full Evaluation (all {len(test_items)} users):")
print(f"{'K':<5} {'HR':>7} {'NDCG':>7}")
print("=" * 25)
for k in K_LIST:
    hr   = np.mean(results[k]['hits'])
    ndcg = np.mean(results[k]['ndcgs'])
    print(f"K={k:<3} {hr:>7.4f} {ndcg:>7.4f}")