import numpy as np
import pandas as pd
import json
from collections import defaultdict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class RecEnv:
    """
    Recommendation environment for RL.

    State  : history of last `window` item embeddings for a user (GRU input)
    Action : item index to recommend (0 .. n_items-1)
    Reward : +1 click, 0 no click, with optional fairness penalty
    """

    def __init__(self, train_path: str, meta_path: str,
                 emb_dim: int = 64, window: int = 10,
                 fairness_lambda: float = 0.5, k: int = 10):

        self.train      = pd.read_csv(train_path)
        self.meta       = json.load(open(meta_path))
        self.n_users    = self.meta['n_users']
        self.n_items    = self.meta['n_items']
        self.emb_dim    = emb_dim
        self.window     = window
        self.k          = k
        self.fairness_lambda = fairness_lambda

        # Build user interaction history: {user_id: [item_id, ...]}
        self.user_history = defaultdict(list)
        for _, row in self.train.iterrows():
            self.user_history[int(row['user_id'])].append(int(row['item_id']))

        # Random item embeddings — replace with pretrained BPR embeddings later
        self.item_embeddings = np.random.randn(self.n_items, emb_dim).astype(np.float32)
        self.item_embeddings /= np.linalg.norm(
            self.item_embeddings, axis=1, keepdims=True) + 1e-9

        # Track item exposure for fairness reward
        self.item_exposure  = np.zeros(self.n_items, dtype=np.float32)
        self.total_recs     = 0

        self.current_user   = None
        self.current_step   = 0
        self.max_steps      = 20     # max recommendations per episode

    # ── Core RL interface ─────────────────────────────────────────────

    def reset(self, user_id: int = None):
        """Start a new episode for a user. Returns initial state."""
        if user_id is None:
            user_id = np.random.randint(0, self.n_users)

        self.current_user = user_id
        self.current_step = 0
        self.session_items = []      # items recommended this episode
        return self._get_state()

    def step(self, action: int):
        """
        Take action (recommend item `action`).
        Returns: (next_state, reward, done, info)
        """
        assert self.current_user is not None, "Call reset() first"

        reward = self._compute_reward(action)

        # Update history and exposure
        self.user_history[self.current_user].append(action)
        self.item_exposure[action] += 1
        self.total_recs += 1
        self.session_items.append(action)
        self.current_step += 1

        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        info = {'user': self.current_user, 'item': action, 'step': self.current_step}

        return next_state, reward, done, info

    # ── State ─────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        State = mean of last `window` item embeddings.
        Shape: (emb_dim,)
        """
        history = self.user_history[self.current_user]
        if len(history) == 0:
            return np.zeros(self.emb_dim, dtype=np.float32)
        recent = history[-self.window:]
        return self.item_embeddings[recent].mean(axis=0)

    # ── Reward ────────────────────────────────────────────────────────

    # def _compute_reward(self, item: int) -> float:
    #     """
    #     Relevance reward + fairness penalty.

    #     R = R_relevance - lambda * R_fairness_violation
    #     """
    #     r_relevance = self._relevance_reward(item)
    #     r_fairness  = self._fairness_penalty(item)
    #     return r_relevance - self.fairness_lambda * r_fairness

    def _compute_reward(self, item: int) -> float:
        r_relevance = self._relevance_reward(item)
        r_fairness  = self._fairness_penalty(item)

        # 🔥 NEW: popularity penalty
        if self.total_recs == 0:
            pop_penalty = 0.0
        else:
            pop_penalty = self.item_exposure[item] / (self.total_recs + 1)

        return r_relevance - self.fairness_lambda * (r_fairness + pop_penalty)
    def _relevance_reward(self, item: int) -> float:
        """
        Simulate click signal from implicit feedback.
        +1 if item is in user's held-out interactions, else 0.
        In a real system this would be a live click signal.
        """
        if item in self.user_history[self.current_user]:
            return 1.0
        # Partial reward: item is similar to user history
        state = self._get_state()
        similarity = float(np.dot(state, self.item_embeddings[item]))
        return max(0.0, similarity)

    def _fairness_penalty(self, item: int) -> float:
        """
        Penalize recommending over-exposed items.
        Uses Inverse Propensity Scoring (IPS) logic:
        penalty is proportional to how over-exposed this item already is.
        """
        if self.total_recs == 0:
            return 0.0
        expected_exposure = self.total_recs / self.n_items
        actual_exposure   = self.item_exposure[item]
        # return max(0.0, actual_exposure - expected_exposure) / (self.total_recs + 1)
        return actual_exposure / (self.total_recs + 1)
    # ── Utilities ─────────────────────────────────────────────────────

    def load_pretrained_embeddings(self, embeddings: np.ndarray):
        """Load embeddings from a pretrained BPR or LightGCN model."""
        assert embeddings.shape == (self.n_items, self.emb_dim)
        self.item_embeddings = embeddings.astype(np.float32)

    def get_state_dim(self) -> int:
        return self.emb_dim

    def get_action_dim(self) -> int:
        return self.n_items