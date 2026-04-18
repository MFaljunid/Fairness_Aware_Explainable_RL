import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np
import glob

# ── Find the latest saved BPR model ──────────────────────────────────
pkl_files = glob.glob('results/BPR/*.pkl')
assert len(pkl_files) > 0, "No BPR model found — run bpr_baseline.py first"

latest = sorted(pkl_files)[-1]
print(f"Loading BPR model from: {latest}")

with open(latest, 'rb') as f:
    bpr_model = pickle.load(f)

# ── Extract item embeddings ───────────────────────────────────────────
# Cornac BPR stores item factors as i_factors
item_embeddings = bpr_model.i_factors   # shape: (n_items, k)
user_embeddings = bpr_model.u_factors   # shape: (n_users, k)

print(f"Item embeddings shape : {item_embeddings.shape}")
print(f"User embeddings shape : {user_embeddings.shape}")
print(f"Item emb sample norms : {np.linalg.norm(item_embeddings[:5], axis=1).round(4)}")

# ── Save ──────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
np.save('data/bpr_item_embeddings.npy', item_embeddings)
np.save('data/bpr_user_embeddings.npy', user_embeddings)

print(f"\nSaved:")
print(f"  data/bpr_item_embeddings.npy  {item_embeddings.shape}")
print(f"  data/bpr_user_embeddings.npy  {user_embeddings.shape}")
print("\nReady to load into RL environment.")