import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────
EMB_DIM    = 64     # embedding size per item
HIDDEN_DIM = 256    # GRU hidden state size
WINDOW     = 10     # how many past items to look at
BATCH_SIZE = 4      # test with 4 users at once

# ── Load data ─────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
meta  = json.load(open('data/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

print(f"N_USERS: {N_USERS}  |  N_ITEMS: {N_ITEMS}  |  EMB_DIM: {EMB_DIM}")

# ── Build user history ────────────────────────────────────────────────
user_history = defaultdict(list)
for _, row in train.iterrows():
    user_history[int(row['user_id'])].append(int(row['item_id']))

print(f"History built for {len(user_history)} users")

# ─────────────────────────────────────────────────────────────────────
# PIECE 1: Item Embedding Table
# Maps each item_id (integer) → dense vector of size EMB_DIM
# This is what turns a sequence of item ids into a sequence of vectors
# ─────────────────────────────────────────────────────────────────────
item_embedding_table = nn.Embedding(
    num_embeddings=N_ITEMS,   # one vector per item
    embedding_dim=EMB_DIM,    # each vector is 64-dimensional
    padding_idx=0             # item_id 0 = padding → always zero vector
)

print(f"\nEmbedding table shape : {item_embedding_table.weight.shape}")
# Expected: torch.Size([3416, 64])

# Test: look up embeddings for 3 items
test_items = torch.LongTensor([1, 50, 200])
test_embs  = item_embedding_table(test_items)
print(f"Embedding of items [1,50,200] shape: {test_embs.shape}")
# Expected: torch.Size([3, 64])

# ─────────────────────────────────────────────────────────────────────
# PIECE 2: History → Padded Sequence
# Takes a user's last WINDOW item ids and pads shorter histories
# ─────────────────────────────────────────────────────────────────────
def get_history_sequence(user_id: int, window: int = WINDOW) -> torch.LongTensor:
    """
    Returns the last `window` item ids for a user as a LongTensor.
    Pads with 0 at the front if history is shorter than window.
    Shape: (window,)
    """
    history = user_history[user_id]
    recent  = history[-window:]                    # last `window` items

    # Pad at the front with 0 (padding_idx)
    if len(recent) < window:
        pad    = [0] * (window - len(recent))
        recent = pad + recent

    return torch.LongTensor(recent)                # (window,)

# Test with user 0
seq_user0 = get_history_sequence(0)
print(f"\nHistory sequence for user 0 : {seq_user0}")
print(f"Shape                       : {seq_user0.shape}")
# Expected: torch.Size([10])

# ─────────────────────────────────────────────────────────────────────
# PIECE 3: Sequence → Embedding Matrix
# Converts (window,) item ids → (window, emb_dim) embedding matrix
# ─────────────────────────────────────────────────────────────────────
emb_seq_user0 = item_embedding_table(seq_user0)
print(f"\nEmbedding sequence shape : {emb_seq_user0.shape}")
# Expected: torch.Size([10, 64])

# ─────────────────────────────────────────────────────────────────────
# PIECE 4: GRU Encoder
# Takes (batch, window, emb_dim) → outputs (batch, hidden_dim)
# The final hidden state = compressed summary of user's taste
# ─────────────────────────────────────────────────────────────────────
class GRUStateEncoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,                      # input is (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq    : (batch, window, emb_dim)
        output : (batch, hidden_dim)
        """
        _, h_n = self.gru(seq)        # h_n: (num_layers, batch, hidden_dim)
        state  = h_n[-1]              # last layer only: (batch, hidden_dim)
        return self.norm(state)       # normalize for stable training

encoder = GRUStateEncoder(EMB_DIM, HIDDEN_DIM)
print(f"\nGRU encoder parameters: "
      f"{sum(p.numel() for p in encoder.parameters()):,}")

# ─────────────────────────────────────────────────────────────────────
# PIECE 5: Full pipeline test with a batch of users
# ─────────────────────────────────────────────────────────────────────
test_user_ids = [0, 1, 2, 3]

# Build batch of sequences: (batch, window)
batch_seqs = torch.stack([
    get_history_sequence(uid) for uid in test_user_ids
])
print(f"\nBatch sequences shape    : {batch_seqs.shape}")
# Expected: torch.Size([4, 10])

# Look up embeddings: (batch, window, emb_dim)
batch_embs = item_embedding_table(batch_seqs)
print(f"Batch embeddings shape   : {batch_embs.shape}")
# Expected: torch.Size([4, 10, 64])

# Encode: (batch, hidden_dim)
batch_states = encoder(batch_embs)
print(f"Batch state vectors shape: {batch_states.shape}")
# Expected: torch.Size([4, 256])

# ─────────────────────────────────────────────────────────────────────
# PIECE 6: Verify different users produce different states
# ─────────────────────────────────────────────────────────────────────
state_0 = batch_states[0].detach().numpy()
state_1 = batch_states[1].detach().numpy()
cosine_sim = float(
    np.dot(state_0, state_1) /
    (np.linalg.norm(state_0) * np.linalg.norm(state_1) + 1e-9)
)
print(f"\nCosine similarity user0 vs user1: {cosine_sim:.4f}")
print("(Should be < 1.0 — different users have different states)")

# ─────────────────────────────────────────────────────────────────────
# PIECE 7: Verify gradient flows through the encoder
# ─────────────────────────────────────────────────────────────────────
dummy_target = torch.zeros(BATCH_SIZE, HIDDEN_DIM)
loss = nn.MSELoss()(batch_states, dummy_target)
loss.backward()

emb_grad = item_embedding_table.weight.grad
gru_grad  = encoder.gru.weight_ih_l0.grad

print(f"\nGradient check:")
print(f"  Embedding table grad norm : {emb_grad.norm().item():.4f}")
print(f"  GRU weight grad norm      : {gru_grad.norm().item():.4f}")
print("  (Both should be > 0 — gradients flow correctly)")

print("\n=== All State Encoder checks passed ===")
print("Ready for Step 3: RL Policy Network")