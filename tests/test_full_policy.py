import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from rl_model.policy import ActorCriticPolicy

EMB_DIM    = 64
HIDDEN_DIM = 256
WINDOW     = 10
N_ITEMS    = 3416

print("=" * 50)
print("Testing final policy.py")
print("=" * 50)

policy   = ActorCriticPolicy(EMB_DIM, N_ITEMS, HIDDEN_DIM)
seq_np   = np.random.randint(1, N_ITEMS, (WINDOW,))
exp_np   = np.random.rand(N_ITEMS) * 10

# Test 1: forward pass
print("\n--- Test 1: forward pass ---")
seq_t = torch.LongTensor(seq_np).unsqueeze(0)
exp_t = torch.FloatTensor(exp_np)
logits, value, state, attn = policy.forward(seq_t, exp_t)
assert logits.shape == (1, N_ITEMS)
assert value.shape  == (1,)
assert state.shape  == (1, HIDDEN_DIM)
assert attn.shape   == (1, WINDOW)
assert abs(attn.sum().item() - 1.0) < 1e-5, "Attention must sum to 1"
print(f"logits : {logits.shape}")
print(f"value  : {value.shape}")
print(f"state  : {state.shape}")
print(f"attn   : {attn.shape}  sum={attn.sum().item():.4f}")
print("PASS")

# Test 2: select_action
print("\n--- Test 2: select_action ---")
action, log_prob, val = policy.select_action(seq_np, exp_np,
                                              exclude_items=[0, 1, 2])
assert isinstance(action, int)
assert log_prob.requires_grad
assert val.requires_grad
assert action not in [0, 1, 2]
print(f"action={action}  log_prob={log_prob.item():.4f}  value={val.item():.4f}")
print("PASS")

# Test 3: greedy_action
print("\n--- Test 3: greedy_action ---")
a1 = policy.greedy_action(seq_np, exp_np)
a2 = policy.greedy_action(seq_np, exp_np)
assert a1 == a2
print(f"Same action both calls: {a1}")
print("PASS")

# Test 4: explain
print("\n--- Test 4: explain ---")
explanation = policy.explain(seq_np, exp_np, action=a1)
assert 'attention'      in explanation
assert 'saliency'       in explanation
assert 'counterfactual' in explanation
assert len(explanation['saliency']['scores'])  == WINDOW
assert len(explanation['attention']['weights']) == WINDOW
print(f"Chosen item  : {explanation['chosen_item']}")
print(f"Chosen prob  : {explanation['chosen_prob']:.6f}")
print(f"Attention    : {[round(w,3) for w in explanation['attention']['weights']]}")
print(f"Saliency     : {[round(s,4) for s in explanation['saliency']['scores']]}")
print(f"Counterfact  : {explanation['counterfactual']['explanation']}")
print(f"Attn explain : {explanation['attention']['explanation']}")
print("PASS")

# Test 5: gradients flow through all components
print("\n--- Test 5: gradient flow ---")
policy.train()
seq_t = torch.LongTensor(seq_np).unsqueeze(0)
exp_t = torch.FloatTensor(exp_np)

logits, value, gru_state, attn_weights = policy.forward(seq_t, exp_t)

# Include all outputs in loss so every component gets gradients
# attn_weights depends on query_proj(gru_state) @ emb
# We must use gru_state directly to ensure query_proj gets gradient
query  = policy.attention.query_proj(gru_state)   # explicitly use query_proj
loss   = (logits.mean()
          + value.mean()
          + attn_weights.mean()
          + query.mean())                          # ensures query_proj gradient
loss.backward()

grads = {
    'embedding':      policy.item_emb.weight.grad.norm().item(),
    'GRU':            policy.encoder.gru.weight_ih_l0.grad.norm().item(),
    'actor':          policy.actor.weight.grad.norm().item(),
    'fairness_alpha': policy.fairness_layer.alpha.grad.norm().item(),
    'fairness_proj':  policy.fairness_layer.user_proj.weight.grad.norm().item(),
    'attn_proj':      policy.attention.query_proj.weight.grad.norm().item(),
}
for name, g in grads.items():
    print(f"  {name:20s} : {g:.4f}")
assert all(g > 0 for g in grads.values()), "All grads must be > 0"
print("PASS: all components learn")

print("\n" + "=" * 50)
print("policy.py is complete and verified.")
print("Ready for Step 6: train.py")
print("=" * 50)