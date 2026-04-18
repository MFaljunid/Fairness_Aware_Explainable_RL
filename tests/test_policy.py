import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EMB_DIM    = 64
HIDDEN_DIM = 256
WINDOW     = 10
N_ITEMS    = 3416

print("=" * 50)
print("STEP 3: RL Policy Network (Actor-Critic)")
print("=" * 50)

class GRUStateEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru  = nn.GRU(emb_dim, hidden_dim, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq):
        _, h_n = self.gru(seq)
        return self.norm(h_n[-1])


class FairnessConstraintLayer(nn.Module):
    """
    YOUR NOVELTY — Fairness Constraint Layer.

    Subtracts a learned exposure penalty from actor logits.
    Over-exposed items get lower logits → lower probability of selection.
    The penalty strength alpha is learned during training.
    Per-user sensitivity is also learned via user_proj.
    """
    def __init__(self, hidden_dim, n_items):
        super().__init__()
        self.n_items  = n_items
        self.alpha    = nn.Parameter(torch.ones(1))       # learned penalty scale
        self.user_proj = nn.Linear(hidden_dim, 1)         # user fairness sensitivity

    def forward(self, logits, state, item_exposure):
        # Normalize exposure to [0,1]
        exp_norm = item_exposure / (item_exposure.max() + 1e-9)   # (n_items,)

        # Penalty strength scaled by learned alpha
        fairness_penalty = exp_norm * self.alpha.abs()             # (n_items,)

        # Per-user sensitivity in (0,1)
        user_sensitivity = torch.sigmoid(self.user_proj(state))    # (batch, 1)

        # Subtract penalty from logits — over-exposed items suppressed
        adjusted = logits - user_sensitivity * fairness_penalty.unsqueeze(0)

        return adjusted


class ActorCriticPolicy(nn.Module):
    def __init__(self, emb_dim, n_items, hidden_dim=256, num_gru_layers=2):
        super().__init__()
        self.n_items    = n_items
        self.emb_dim    = emb_dim
        self.hidden_dim = hidden_dim

        self.item_emb       = nn.Embedding(n_items, emb_dim, padding_idx=0)
        self.encoder        = GRUStateEncoder(emb_dim, hidden_dim, num_gru_layers)
        self.trunk          = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.actor          = nn.Linear(hidden_dim, n_items)
        self.critic         = nn.Linear(hidden_dim, 1)
        self.fairness_layer = FairnessConstraintLayer(hidden_dim, n_items)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        # Fairness layer uses xavier — orthogonal breaks it
        nn.init.xavier_uniform_(self.fairness_layer.user_proj.weight)
        nn.init.constant_(self.fairness_layer.user_proj.bias, 0.0)

    def forward(self, item_seq, item_exposure):
        emb         = self.item_emb(item_seq)
        state       = self.encoder(emb)
        features    = self.trunk(state)
        logits      = self.actor(features)
        value       = self.critic(features).squeeze(-1)
        fair_logits = self.fairness_layer(logits, features, item_exposure)
        return fair_logits, value, state

    def select_action(self, item_seq, item_exposure, exclude_items=None):
        seq_t = torch.LongTensor(item_seq).unsqueeze(0)
        exp_t = torch.FloatTensor(item_exposure)
        logits, value, _ = self.forward(seq_t, exp_t)
        logits = logits.clone()
        if exclude_items and len(exclude_items) > 0:
            logits[0, exclude_items] = -1e9
        probs    = F.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(), value.squeeze()

    def greedy_action(self, item_seq, item_exposure, exclude_items=None):
        training = self.training
        self.eval()
        with torch.no_grad():
            seq_t = torch.LongTensor(item_seq).unsqueeze(0)
            exp_t = torch.FloatTensor(item_exposure)
            logits, _, _ = self.forward(seq_t, exp_t)
            logits = logits.clone()
            if exclude_items and len(exclude_items) > 0:
                logits[0, exclude_items] = -1e9
            action = logits.argmax(dim=-1).item()
        if training:
            self.train()
        return action


# ── Tests ─────────────────────────────────────────────────────────────
policy = ActorCriticPolicy(EMB_DIM, N_ITEMS, HIDDEN_DIM)
print(f"\nTotal parameters: {sum(p.numel() for p in policy.parameters()):,}")

# Test 1: shapes
print("\n--- Test 1: Forward pass shapes ---")
batch_seqs = torch.randint(1, N_ITEMS, (4, WINDOW))
exposure   = torch.zeros(N_ITEMS)
logits, value, state = policy.forward(batch_seqs, exposure)
assert logits.shape == (4, N_ITEMS)
assert value.shape  == (4,)
assert state.shape  == (4, HIDDEN_DIM)
print("PASS: all shapes correct")

# Test 2: fairness layer actually suppresses over-exposed items
print("\n--- Test 2: Fairness suppression ---")
exposure_high       = torch.zeros(N_ITEMS)
exposure_high[0]    = 1000.0
exposure_low        = torch.zeros(N_ITEMS)
seq = torch.randint(1, N_ITEMS, (1, WINDOW))

with torch.no_grad():
    logits_fair, _, _ = policy.forward(seq, exposure_high)
    logits_base, _, _ = policy.forward(seq, exposure_low)

probs_fair = F.softmax(logits_fair, dim=-1)[0]
probs_base = F.softmax(logits_base, dim=-1)[0]

prob_high = probs_fair[0].item()
prob_low  = probs_base[0].item()
reduction = (prob_low - prob_high) / (prob_low + 1e-9) * 100

print(f"Item 0 prob — zero exposure : {prob_low:.6f}")
print(f"Item 0 prob — high exposure : {prob_high:.6f}")
print(f"Reduction                   : {reduction:.1f}%")
assert prob_high < prob_low, "Fairness layer must suppress over-exposed items"
assert reduction > 10.0,     "Suppression must be meaningful (>10%)"
print("PASS: fairness layer meaningfully suppresses over-exposed items")

# Test 3: select_action
print("\n--- Test 3: select_action ---")
seq_np = np.random.randint(1, N_ITEMS, (WINDOW,))
exp_np = np.zeros(N_ITEMS)
action, log_prob, val = policy.select_action(seq_np, exp_np, exclude_items=[0,1,2])
assert isinstance(action, int)
assert log_prob.requires_grad
assert val.requires_grad
assert action not in [0, 1, 2]
print(f"Action: {action}  |  log_prob: {log_prob.item():.4f}  |  value: {val.item():.4f}")
print("PASS: select_action correct")

# Test 4: greedy is deterministic
print("\n--- Test 4: greedy_action deterministic ---")
a1 = policy.greedy_action(seq_np, exp_np)
a2 = policy.greedy_action(seq_np, exp_np)
assert a1 == a2
print(f"Same action both calls: {a1}")
print("PASS: deterministic")

# Test 5: gradients flow everywhere including fairness layer
print("\n--- Test 5: Gradient flow ---")
policy.train()
batch_seqs = torch.randint(1, N_ITEMS, (4, WINDOW))
exposure = torch.rand(N_ITEMS) * 100.0
logits, value, _ = policy.forward(batch_seqs, exposure)
loss = logits.mean() + value.mean()
loss.backward()

grads = {
    'embedding':       policy.item_emb.weight.grad.norm().item(),
    'GRU':             policy.encoder.gru.weight_ih_l0.grad.norm().item(),
    'actor':           policy.actor.weight.grad.norm().item(),
    'fairness_alpha':  policy.fairness_layer.alpha.grad.norm().item(),
    'fairness_proj':   policy.fairness_layer.user_proj.weight.grad.norm().item(),
}
for name, g in grads.items():
    print(f"  {name:20s} grad norm: {g:.4f}")
assert all(g > 0 for g in grads.values()), "All gradients must be > 0"
print("PASS: gradients flow through all layers")

# Test 6: learned alpha changes during training
print("\n--- Test 6: Alpha is learnable ---")
alpha_before = policy.fairness_layer.alpha.item()
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
for _ in range(5):
    seq   = torch.randint(1, N_ITEMS, (8, WINDOW))
    exp   = torch.rand(N_ITEMS) * 10
    l, v, _ = policy.forward(seq, exp)
    loss  = l.mean() + v.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
alpha_after = policy.fairness_layer.alpha.item()
print(f"Alpha before training: {alpha_before:.4f}")
print(f"Alpha after 5 steps : {alpha_after:.4f}")
assert alpha_before != alpha_after, "Alpha must update during training"
print("PASS: alpha is learnable")

print("\n" + "=" * 50)
print("All Step 3 tests passed.")
print("Ready for Step 4: Environment + Reward Signal")
print("=" * 50)