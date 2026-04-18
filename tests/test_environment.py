import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import json
from collections import defaultdict

print("=" * 50)
print("STEP 4: Environment + Reward Signal")
print("=" * 50)

# ── Config ────────────────────────────────────────────────────────────
EMB_DIM  = 64
WINDOW   = 10
FAIRNESS_LAMBDA = 0.1

# ── Load data ─────────────────────────────────────────────────────────
train = pd.read_csv('data/train.csv')
meta  = json.load(open('data/meta.json'))

N_USERS = meta['n_users']
N_ITEMS = meta['n_items']

print(f"\nN_USERS: {N_USERS}  |  N_ITEMS: {N_ITEMS}")

# ─────────────────────────────────────────────────────────────────────
# PIECE 1: Build immutable ground-truth history
# This is the user's real interaction history from train.csv
# We NEVER modify this — it is ground truth only
# ─────────────────────────────────────────────────────────────────────
print("\n--- Piece 1: Ground-truth history ---")

_gt_history = defaultdict(list)
for _, row in train.iterrows():
    _gt_history[int(row['user_id'])].append(int(row['item_id']))

# Build set version for O(1) lookup
_gt_set = defaultdict(set)
for u, items in _gt_history.items():
    _gt_set[u] = set(items)

# Verify
assert len(_gt_history) == N_USERS, "Missing users in history"
assert all(len(v) >= 3 for v in _gt_history.values()), "Some users have < 3 items"
print(f"History built for {len(_gt_history)} users")
print(f"Sample user 0 history (last 5): {_gt_history[0][-5:]}")
print("PASS: ground-truth history correct")

# ─────────────────────────────────────────────────────────────────────
# PIECE 2: Item embeddings
# Random for now — will be replaced with BPR embeddings later
# ─────────────────────────────────────────────────────────────────────
print("\n--- Piece 2: Item embeddings ---")

np.random.seed(42)
item_embeddings = np.random.randn(N_ITEMS, EMB_DIM).astype(np.float32)

# Unit normalize
norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-9
item_embeddings /= norms

assert item_embeddings.shape == (N_ITEMS, EMB_DIM)
assert abs(np.linalg.norm(item_embeddings[0]) - 1.0) < 1e-5, "Not normalized"
print(f"Embeddings shape : {item_embeddings.shape}")
print(f"Norm of item 0   : {np.linalg.norm(item_embeddings[0]):.4f}  (should be 1.0)")
print("PASS: embeddings correct")

# ─────────────────────────────────────────────────────────────────────
# PIECE 3: _get_state()
# Returns last WINDOW items as a padded sequence (window, emb_dim)
# This feeds directly into the GRU encoder
# ─────────────────────────────────────────────────────────────────────
print("\n--- Piece 3: _get_state() ---")

def _get_state(user_id: int) -> np.ndarray:
    history = _gt_history[user_id]
    if len(history) == 0:
        return np.zeros((WINDOW, EMB_DIM), dtype=np.float32)
    recent = history[-WINDOW:]
    seq    = item_embeddings[recent]
    if len(seq) < WINDOW:
        pad = np.zeros((WINDOW - len(seq), EMB_DIM), dtype=np.float32)
        seq = np.vstack([pad, seq])
    return seq.astype(np.float32)

# Test
state_u0 = _get_state(0)
state_u1 = _get_state(1)

assert state_u0.shape == (WINDOW, EMB_DIM), f"Wrong shape: {state_u0.shape}"
assert state_u1.shape == (WINDOW, EMB_DIM)
assert not np.array_equal(state_u0, state_u1), "Different users must have different states"
print(f"State shape for user 0 : {state_u0.shape}")
print(f"State shape for user 1 : {state_u1.shape}")
print(f"States are different   : {not np.array_equal(state_u0, state_u1)}")
print("PASS: _get_state() correct")

# ─────────────────────────────────────────────────────────────────────
# PIECE 4: Relevance reward
# +1 if recommended item is in user's ground-truth history
# 0 otherwise
# Binary signal — clean and unambiguous
# ─────────────────────────────────────────────────────────────────────
print("\n--- Piece 4: Relevance reward ---")

def _relevance_reward(user_id: int, item: int) -> float:
    return 1.0 if item in _gt_set[user_id] else 0.0

# Test with known items
user_0_items   = list(_gt_set[0])
known_item     = user_0_items[0]       # item we know user 0 interacted with
unknown_item   = N_ITEMS - 1           # last item — very unlikely in history

r_known   = _relevance_reward(0, known_item)
r_unknown = _relevance_reward(0, unknown_item)

print(f"Reward for known item {known_item}  : {r_known}")
print(f"Reward for unknown item {unknown_item}: {r_unknown}")
assert r_known   == 1.0, "Known item must give reward 1.0"
assert r_unknown == 0.0, "Unknown item must give reward 0.0"
print("PASS: relevance reward correct")

# ─────────────────────────────────────────────────────────────────────
# PIECE 5: Fairness penalty
# Penalizes items that are over-exposed relative to uniform exposure
# Normalized to [0, 1] — same scale as relevance reward
# ─────────────────────────────────────────────────────────────────────
print("\n--- Piece 5: Fairness penalty ---")

def _fairness_penalty(item: int,
                      item_exposure: np.ndarray,
                      total_recs: int) -> float:
    if total_recs == 0:
        return 0.0
    expected = total_recs / N_ITEMS
    actual   = float(item_exposure[item])
    excess   = max(0.0, actual - expected)
    return min(1.0, excess / (expected + 1e-9))

# Test 1: no recommendations yet → penalty = 0
item_exposure = np.zeros(N_ITEMS, dtype=np.float32)
penalty_cold  = _fairness_penalty(0, item_exposure, total_recs=0)
assert penalty_cold == 0.0
print(f"Penalty with 0 recs        : {penalty_cold}")

# Test 2: item recommended 100x, others 0x → high penalty
item_exposure_biased       = np.zeros(N_ITEMS, dtype=np.float32)
item_exposure_biased[0]    = 100.0
penalty_overexposed        = _fairness_penalty(0, item_exposure_biased, total_recs=100)
penalty_normal             = _fairness_penalty(1, item_exposure_biased, total_recs=100)
print(f"Penalty for over-exposed item : {penalty_overexposed:.4f}  (should be 1.0)")
print(f"Penalty for unexposed item    : {penalty_normal:.4f}   (should be 0.0)")
assert penalty_overexposed == 1.0, "Over-exposed item must have max penalty"
assert penalty_normal      == 0.0, "Unexposed item must have zero penalty"
print("PASS: fairness penalty correct")

# ─────────────────────────────────────────────────────────────────────
# PIECE 6: Combined reward
# R = R_relevance - lambda * R_fairness
# ─────────────────────────────────────────────────────────────────────
print("\n--- Piece 6: Combined reward ---")

def _compute_reward(user_id: int, item: int,
                    item_exposure: np.ndarray,
                    total_recs: int) -> float:
    r_rel  = _relevance_reward(user_id, item)
    r_fair = _fairness_penalty(item, item_exposure, total_recs)
    return r_rel - FAIRNESS_LAMBDA * r_fair

# Scenario A: relevant + fair item
item_exposure = np.zeros(N_ITEMS, dtype=np.float32)
user_0_item   = list(_gt_set[0])[0]
reward_good   = _compute_reward(0, user_0_item, item_exposure, total_recs=10)

# Scenario B: relevant but over-exposed item
item_exposure_biased         = np.zeros(N_ITEMS, dtype=np.float32)
item_exposure_biased[user_0_item] = 1000.0
reward_overexposed = _compute_reward(0, user_0_item, item_exposure_biased, total_recs=1000)

# Scenario C: irrelevant item
irrelevant_item = N_ITEMS - 1
reward_irrelevant = _compute_reward(0, irrelevant_item, item_exposure, total_recs=10)

print(f"Reward (relevant + fair)       : {reward_good:.4f}   ← should be ~1.0")
print(f"Reward (relevant + overexposed): {reward_overexposed:.4f}   ← should be < 1.0")
print(f"Reward (irrelevant)            : {reward_irrelevant:.4f}   ← should be 0.0")
assert reward_good > reward_overexposed, "Fair item must beat overexposed item"
assert reward_good > reward_irrelevant,  "Relevant item must beat irrelevant item"
assert reward_irrelevant == 0.0,         "Irrelevant item must give 0 reward"
print("PASS: combined reward correct")

# ─────────────────────────────────────────────────────────────────────
# PIECE 7: Full episode simulation
# Simulates one complete episode: reset → 20 steps → done
# ─────────────────────────────────────────────────────────────────────
print("\n--- Piece 7: Full episode simulation ---")

MAX_STEPS = 20
user_id   = 0

# Reset
current_user     = user_id
current_step     = 0
session_history  = []
item_exposure    = np.zeros(N_ITEMS, dtype=np.float32)
total_recs       = 0
state            = _get_state(current_user)

assert state.shape == (WINDOW, EMB_DIM)

episode_rewards = []
episode_hits    = 0

for step in range(MAX_STEPS):
    # Pick a random action (not yet using policy)
    excluded = session_history.copy()
    candidates = [i for i in range(N_ITEMS) if i not in set(excluded)]
    action = np.random.choice(candidates)

    # Compute reward
    reward = _compute_reward(current_user, action,
                             item_exposure, total_recs)

    # Update state
    session_history.append(action)
    item_exposure[action] += 1
    total_recs += 1
    current_step += 1
    done  = current_step >= MAX_STEPS
    state = _get_state(current_user)

    episode_rewards.append(reward)
    if _relevance_reward(current_user, action) == 1.0:
        episode_hits += 1

print(f"Episode length  : {len(episode_rewards)} steps")
print(f"Total reward    : {sum(episode_rewards):.4f}")
print(f"Relevant hits   : {episode_hits} / {MAX_STEPS}")
print(f"Hit rate        : {episode_hits/MAX_STEPS*100:.1f}%")
print(f"Reward range    : [{min(episode_rewards):.4f}, {max(episode_rewards):.4f}]")
assert len(episode_rewards) == MAX_STEPS
assert len(set(session_history)) == MAX_STEPS, "No duplicate recommendations"
assert min(episode_rewards) >= -FAIRNESS_LAMBDA, "Reward below minimum"
print("PASS: full episode runs correctly")

print("\n" + "=" * 50)
print("All Step 4 tests passed.")
print("Ready to write environment.py")
print("=" * 50)

print("\n--- Piece 8: Final environment.py ---")

from rl_model.environment import RecEnv

env = RecEnv(
    train_path='data/train.csv',
    meta_path='data/meta.json',
    emb_dim=EMB_DIM,
    window=WINDOW,
    fairness_lambda=FAIRNESS_LAMBDA
)

# Test reset
state = env.reset(user_id=0)
assert state.shape == (WINDOW, EMB_DIM), f"Wrong state shape: {state.shape}"
print(f"State shape after reset : {state.shape}")

# Test step
action              = 100
next_state, reward, done, info = env.step(action)
assert next_state.shape == (WINDOW, EMB_DIM)
assert isinstance(reward, float)
assert isinstance(done, bool)
assert 'is_hit' in info
print(f"Step result — reward: {reward:.4f} | done: {done} | is_hit: {info['is_hit']}")

# Test full episode
state    = env.reset()
total_r  = 0.0
hits     = 0
for _ in range(env.max_steps):
    excluded            = env.get_excluded_items()
    action              = np.random.choice(
        [i for i in range(N_ITEMS) if i not in set(excluded)])
    state, reward, done, info = env.step(action)
    total_r += reward
    if info['is_hit']:
        hits += 1
    if done:
        break

print(f"Full episode — total reward: {total_r:.4f} | hits: {hits}/{env.max_steps}")
assert len(env.get_excluded_items()) == env.max_steps
print("PASS: environment.py works end-to-end")

print("\n" + "=" * 50)
print("Step 4 complete. environment.py is ready.")
print("Ready for Step 5: Explainability Module")
print("=" * 50)