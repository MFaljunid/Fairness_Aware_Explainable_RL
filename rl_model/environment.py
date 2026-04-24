import numpy as np
import pandas as pd
import json
from collections import defaultdict
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class RecEnv:
    """
    Recommendation Environment for RL training.
    State  : sequence of last `window` item embeddings → (window, emb_dim)
    Action : item index to recommend (0 .. n_items-1)
    Reward : R_relevance - lambda * R_fairness (item + gender aware)
    """

    def __init__(self, train_path: str, meta_path: str,
                 emb_dim: int = 64, window: int = 10,
                 fairness_lambda: float = 0.1, k: int = 10,
                 users_dat_path: str = None):

        # ── Load data ──────────────────────────────────────────────────
        self.train           = pd.read_csv(train_path)
        self.meta            = json.load(open(meta_path))
        self.n_users         = self.meta['n_users']
        self.n_items         = self.meta['n_items']
        self.emb_dim         = emb_dim
        self.window          = window
        self.k               = k
        self.fairness_lambda = fairness_lambda

        # ── Build ground-truth history ─────────────────────────────────
        self._gt_history = defaultdict(list)
        for _, row in self.train.iterrows():
            self._gt_history[int(row['user_id'])].append(int(row['item_id']))

        self._gt_set = defaultdict(set)
        for u, items in self._gt_history.items():
            self._gt_set[u] = set(items)

        # ── Item embeddings ────────────────────────────────────────────
        np.random.seed(42)
        self.item_embeddings = np.random.randn(
            self.n_items, emb_dim).astype(np.float32)
        self._normalize_embeddings()

        # ── Gender-aware fairness tracking ─────────────────────────────
        self.user_gender = {}
        if users_dat_path and os.path.exists(users_dat_path):
            user2idx = {int(k): int(v)
                        for k, v in self.meta['user2idx'].items()}
            with open(users_dat_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 2:
                        uid = int(parts[0])
                        if uid in user2idx:
                            self.user_gender[user2idx[uid]] = parts[1]
            print(f"Loaded gender for {len(self.user_gender)} users")

        # Per-item exposure by gender
        self.male_item_exposure   = np.zeros(self.n_items, dtype=np.float32)
        self.female_item_exposure = np.zeros(self.n_items, dtype=np.float32)

        # ── Episode state ──────────────────────────────────────────────
        self.current_user      = None
        self.current_step      = 0
        self.max_steps         = 20
        self._session_history  = []
        self.item_exposure     = np.zeros(self.n_items, dtype=np.float32)
        self.total_recs        = 0
        self.male_total_recs   = 0
        self.female_total_recs = 0

        print(f"RecEnv ready — users: {self.n_users} | "
              f"items: {self.n_items} | "
              f"emb_dim: {self.emb_dim}")

    # ── Normalization ──────────────────────────────────────────────────

    def _normalize_embeddings(self):
        """Unit-normalize item embeddings in place."""
        norms = np.linalg.norm(
            self.item_embeddings, axis=1, keepdims=True) + 1e-9
        self.item_embeddings /= norms

    # ── Core RL interface ──────────────────────────────────────────────

    def reset(self, user_id: int = None) -> np.ndarray:
        if user_id is None:
            user_id = np.random.randint(0, self.n_users)

        self.current_user     = int(user_id)
        self.current_step     = 0
        self._session_history = []
        self.item_exposure    = np.zeros(self.n_items, dtype=np.float32)
        self.total_recs       = 0

        return self._get_state()

    def step(self, action: int):
        assert self.current_user is not None, "Call reset() before step()"
        assert 0 <= action < self.n_items,    f"Invalid action {action}"

        reward = self._compute_reward(action)

        # Update session
        self._session_history.append(action)
        self.item_exposure[action] += 1
        self.total_recs            += 1
        self.current_step          += 1

        # Update gender-aware exposure
        gender = self.user_gender.get(self.current_user, 'M')
        if gender == 'M':
            self.male_item_exposure[action] += 1
            self.male_total_recs            += 1
        else:
            self.female_item_exposure[action] += 1
            self.female_total_recs            += 1

        done       = self.current_step >= self.max_steps
        next_state = self._get_state()

        info = {
            'user':   self.current_user,
            'item':   action,
            'step':   self.current_step,
            'reward': reward,
            'is_hit': self._relevance_reward(action) == 1.0,
        }

        return next_state, reward, done, info

    # ── State ──────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
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

    # ── Reward ─────────────────────────────────────────────────────────

    def _compute_reward(self, item: int) -> float:
        r_rel  = self._relevance_reward(item)
        r_fair = self._fairness_penalty(item)
        return r_rel - self.fairness_lambda * r_fair

    def _relevance_reward(self, item: int) -> float:
        if item in self._gt_set[self.current_user]:
            return 1.0
        state     = self._get_state()
        state_vec = state.mean(axis=0)
        sim       = float(np.dot(state_vec, self.item_embeddings[item]))
        return max(0.0, sim)

    def _fairness_penalty(self, item: int) -> float:
        # Component 1: item over-exposure (weight 0.3)
        if self.total_recs == 0:
            item_penalty = 0.0
        else:
            expected     = self.total_recs / self.n_items
            actual       = float(self.item_exposure[item])
            excess       = max(0.0, actual - expected)
            item_penalty = min(1.0, excess / (expected + 1e-9))

        # Component 2: gender disparity (weight 0.7)
        male_c         = float(self.male_item_exposure[item])
        female_c       = float(self.female_item_exposure[item])
        denom          = male_c + female_c
        gender_penalty = abs(male_c - female_c) / denom if denom > 0 else 0.0

        return 0.3 * item_penalty + 0.7 * gender_penalty
    # ── Utilities ──────────────────────────────────────────────────────

    def load_pretrained_embeddings(self, embeddings: np.ndarray):
        assert embeddings.shape[0] == self.n_items, \
            f"Expected {self.n_items} items, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == self.emb_dim, \
            f"Expected emb_dim={self.emb_dim}, got {embeddings.shape[1]}"
        self.item_embeddings = embeddings.astype(np.float32)
        self._normalize_embeddings()
        print(f"Loaded pretrained embeddings: {embeddings.shape}")

    def get_excluded_items(self) -> list:
        return self._session_history.copy()

    def get_user_history(self, user_id: int) -> list:
        return self._gt_history[user_id].copy()

    def get_state_shape(self) -> tuple:
        return (self.window, self.emb_dim)

    def get_n_items(self) -> int:
        return self.n_items

    def get_n_users(self) -> int:
        return self.n_users