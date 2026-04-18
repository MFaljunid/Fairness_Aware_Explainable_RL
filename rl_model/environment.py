import numpy as np
import pandas as pd
import json
from collections import defaultdict
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class RecEnv:
    """
    Recommendation Environment for RL training.

    Connects to the data pipeline from preprocessing.py and provides
    the standard RL interface: reset() → step() → reward.

    State  : sequence of last `window` item embeddings → (window, emb_dim)
             fed directly into GRU encoder in policy.py
    Action : item index to recommend (0 .. n_items-1)
    Reward : R_relevance - lambda * R_fairness
    """

    def __init__(self, train_path: str, meta_path: str,
                 emb_dim: int = 64, window: int = 10,
                 fairness_lambda: float = 0.1, k: int = 10):

        # ── Load data ─────────────────────────────────────────────────
        self.train           = pd.read_csv(train_path)
        self.meta            = json.load(open(meta_path))
        self.n_users         = self.meta['n_users']
        self.n_items         = self.meta['n_items']
        self.emb_dim         = emb_dim
        self.window          = window
        self.k               = k
        self.fairness_lambda = fairness_lambda

        # ── Build immutable ground-truth history ──────────────────────
        # Never modify _gt_history or _gt_set during training
        self._gt_history = defaultdict(list)
        for _, row in self.train.iterrows():
            self._gt_history[int(row['user_id'])].append(int(row['item_id']))

        # Set version for O(1) lookup in relevance reward
        self._gt_set = defaultdict(set)
        for u, items in self._gt_history.items():
            self._gt_set[u] = set(items)

        # ── Item embeddings ───────────────────────────────────────────
        # Random init — replace with BPR embeddings via
        # load_pretrained_embeddings() after baselines are trained
        np.random.seed(42)
        self.item_embeddings = np.random.randn(
            self.n_items, emb_dim).astype(np.float32)
        self._normalize_embeddings()

        # ── Episode state ─────────────────────────────────────────────
        self.current_user     = None
        self.current_step     = 0
        self.max_steps        = 20
        self._session_history = []       # items recommended THIS episode only
        self.item_exposure    = np.zeros(self.n_items, dtype=np.float32)
        self.total_recs       = 0

        print(f"RecEnv ready — users: {self.n_users} | "
              f"items: {self.n_items} | "
              f"emb_dim: {self.emb_dim}")

    # ── Normalization ─────────────────────────────────────────────────

    def _normalize_embeddings(self):
        """Unit-normalize item embeddings in place."""
        norms = np.linalg.norm(
            self.item_embeddings, axis=1, keepdims=True) + 1e-9
        self.item_embeddings /= norms

    # ── Core RL interface ─────────────────────────────────────────────

    def reset(self, user_id: int = None) -> np.ndarray:
        """
        Start a new episode for a user.

        Parameters
        ----------
        user_id : int or None
            If None, sample a random user.

        Returns
        -------
        state : np.ndarray of shape (window, emb_dim)
        """
        if user_id is None:
            user_id = np.random.randint(0, self.n_users)

        self.current_user     = int(user_id)
        self.current_step     = 0
        self._session_history = []
        self.item_exposure    = np.zeros(self.n_items, dtype=np.float32)
        self.total_recs       = 0

        return self._get_state()

    def step(self, action: int):
        """
        Recommend item `action` and observe reward.

        Parameters
        ----------
        action : int — item index chosen by the policy

        Returns
        -------
        next_state : np.ndarray (window, emb_dim)
        reward     : float
        done       : bool
        info       : dict
        """
        assert self.current_user is not None, "Call reset() before step()"
        assert 0 <= action < self.n_items,    f"Invalid action {action}"

        reward = self._compute_reward(action)

        # Update session — never touch _gt_history
        self._session_history.append(action)
        self.item_exposure[action] += 1
        self.total_recs += 1
        self.current_step += 1

        done       = self.current_step >= self.max_steps
        next_state = self._get_state()

        info = {
            'user':       self.current_user,
            'item':       action,
            'step':       self.current_step,
            'reward':     reward,
            'is_hit':     self._relevance_reward(action) == 1.0,
        }

        return next_state, reward, done, info

    # ── State ─────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Returns the last `window` items from ground-truth history
        as a padded embedding sequence.

        Shape: (window, emb_dim) — fed directly into GRU encoder.
        Padding (zeros) at the front for users with short histories.
        Session items are NOT included — ground truth only.
        """
        history = self._gt_history[self.current_user]

        if len(history) == 0:
            return np.zeros((self.window, self.emb_dim), dtype=np.float32)

        recent = history[-self.window:]
        seq    = self.item_embeddings[recent]

        if len(seq) < self.window:
            pad = np.zeros(
                (self.window - len(seq), self.emb_dim), dtype=np.float32)
            seq = np.vstack([pad, seq])

        return seq.astype(np.float32)

    # ── Reward ────────────────────────────────────────────────────────

    def _compute_reward(self, item: int) -> float:
        """
        R = R_relevance - lambda * R_fairness

        Both components are in [0, 1] so the total is in
        [-lambda, 1.0] which is a stable range for RL training.
        """
        r_rel  = self._relevance_reward(item)
        r_fair = self._fairness_penalty(item)
        return r_rel - self.fairness_lambda * r_fair

    def _relevance_reward(self, item: int) -> float:
        """
        +1 if item is in user's ground-truth training interactions.
         0 otherwise.

        Binary and unambiguous — no similarity heuristics.
        Cold-start users (not in _gt_set) safely return 0.
        """
        return 1.0 if item in self._gt_set[self.current_user] else 0.0

    def _fairness_penalty(self, item: int) -> float:
        """
        Penalize items that are over-exposed relative to uniform.

        Normalized to [0, 1] so it stays on the same scale
        as the relevance reward.
        """
        if self.total_recs == 0:
            return 0.0
        expected = self.total_recs / self.n_items
        actual   = float(self.item_exposure[item])
        excess   = max(0.0, actual - expected)
        return min(1.0, excess / (expected + 1e-9))

    # ── Utilities ─────────────────────────────────────────────────────

    def load_pretrained_embeddings(self, embeddings: np.ndarray):
        """
        Load pretrained item embeddings from BPR or LightGCN.
        Automatically re-normalizes after loading.

        Parameters
        ----------
        embeddings : np.ndarray of shape (n_items, emb_dim)
        """
        assert embeddings.shape[0] == self.n_items, \
            f"Expected {self.n_items} items, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == self.emb_dim, \
            f"Expected emb_dim={self.emb_dim}, got {embeddings.shape[1]}"
        self.item_embeddings = embeddings.astype(np.float32)
        self._normalize_embeddings()
        print(f"Loaded pretrained embeddings: {embeddings.shape}")

    def get_excluded_items(self) -> list:
        """
        Returns items already recommended this episode.
        Pass to policy.select_action(exclude_items=...) to avoid
        recommending the same item twice.
        """
        return self._session_history.copy()

    def get_user_history(self, user_id: int) -> list:
        """Returns ground-truth item history for a user."""
        return self._gt_history[user_id].copy()

    def get_state_shape(self) -> tuple:
        """Returns (window, emb_dim) — shape of one state tensor."""
        return (self.window, self.emb_dim)

    def get_n_items(self) -> int:
        return self.n_items

    def get_n_users(self) -> int:
        return self.n_users