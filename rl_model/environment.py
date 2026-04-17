import numpy as np
import pandas as pd
import json
from collections import defaultdict
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class RecEnv:
    def __init__(self, train_path: str, meta_path: str,
                 emb_dim: int = 64, window: int = 10,
                 fairness_lambda: float = 0.1, k: int = 10):

        self.train           = pd.read_csv(train_path)
        self.meta            = json.load(open(meta_path))
        self.n_users         = self.meta['n_users']
        self.n_items         = self.meta['n_items']
        self.emb_dim         = emb_dim
        self.window          = window
        self.k               = k
        self.fairness_lambda = fairness_lambda

        # Build IMMUTABLE ground-truth history — never modify this
        self._gt_history = defaultdict(list)
        for _, row in self.train.iterrows():
            self._gt_history[int(row['user_id'])].append(int(row['item_id']))

        # Build ground-truth set for fast O(1) lookup
        self._gt_set = {u: set(items) for u, items in self._gt_history.items()}

        # Item embeddings — random for now, replace with BPR embeddings later
        np.random.seed(42)
        self.item_embeddings = np.random.randn(self.n_items, emb_dim).astype(np.float32)
        self.item_embeddings /= (np.linalg.norm(
            self.item_embeddings, axis=1, keepdims=True) + 1e-9)

        # Episode-level exposure — reset every episode
        self.item_exposure  = np.zeros(self.n_items, dtype=np.float32)
        self.total_recs     = 0

        # Global exposure across all episodes — for fairness evaluation only
        self.global_exposure = np.zeros(self.n_items, dtype=np.float32)

        self.current_user    = None
        self.current_step    = 0
        self.max_steps       = 20
        self._session_history = []   # items seen THIS episode only

    # ── Core RL interface ─────────────────────────────────────────────

    def reset(self, user_id: int = None):
        if user_id is None:
            user_id = np.random.randint(0, self.n_users)

        self.current_user     = user_id
        self.current_step     = 0
        self._session_history = []

        # Reset episode-level exposure tracking
        self.item_exposure = np.zeros(self.n_items, dtype=np.float32)
        self.total_recs    = 0

        return self._get_state()

    def step(self, action: int):
        assert self.current_user is not None, "Call reset() first"

        reward = self._compute_reward(action)

        # Update SESSION history only — never touch ground-truth history
        self._session_history.append(action)

        # Update episode-level exposure
        self.item_exposure[action] += 1
        self.total_recs += 1

        # Update global exposure for evaluation
        self.global_exposure[action] += 1

        self.current_step += 1
        done       = self.current_step >= self.max_steps
        next_state = self._get_state()
        info       = {
            'user':  self.current_user,
            'item':  action,
            'step':  self.current_step,
            'reward': reward
        }
        return next_state, reward, done, info

    # ── State ─────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        State = mean of last `window` items from ground-truth history only.
        Session items are NOT included — avoids history pollution.
        """
        history = self._gt_history[self.current_user]
        if len(history) == 0:
            return np.zeros(self.emb_dim, dtype=np.float32)
        recent = history[-self.window:]
        return self.item_embeddings[recent].mean(axis=0)

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self, item: int) -> float:
        """
        R = R_relevance - lambda * R_fairness_penalty

        Keeps relevance and fairness on similar scales.
        """
        r_relevance = self._relevance_reward(item)
        r_fairness  = self._fairness_penalty(item)
        return r_relevance - self.fairness_lambda * r_fairness

    def _relevance_reward(self, item: int) -> float:
        """
        Binary reward: +1 if item is in user's ground-truth history.
        No similarity fallback — random embeddings produce noise, not signal.
        """
        return 1.0 if item in self._gt_set[self.current_user] else 0.0

    def _fairness_penalty(self, item: int) -> float:
        """
        Penalize items that are over-exposed relative to uniform exposure.
        Normalized to [0, 1] range so it stays on same scale as relevance.
        """
        if self.total_recs == 0:
            return 0.0
        expected = self.total_recs / self.n_items
        actual   = float(self.item_exposure[item])
        excess   = max(0.0, actual - expected)
        # Normalize: divide by expected so penalty is scale-free
        return min(1.0, excess / (expected + 1e-9))

    # ── Utilities ─────────────────────────────────────────────────────

    def load_pretrained_embeddings(self, embeddings: np.ndarray):
        assert embeddings.shape == (self.n_items, self.emb_dim), \
            f"Expected ({self.n_items}, {self.emb_dim}), got {embeddings.shape}"
        self.item_embeddings = embeddings.astype(np.float32)

    def get_excluded_items(self) -> list:
        """Items to exclude from action selection (already seen this session)."""
        return self._session_history.copy()

    def get_state_dim(self) -> int:
        return self.emb_dim

    def get_action_dim(self) -> int:
        return self.n_items