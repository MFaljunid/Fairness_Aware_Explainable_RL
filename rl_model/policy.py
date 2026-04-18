import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUStateEncoder(nn.Module):
    """
    Step 2 — Encodes user interaction sequence into a state vector.
    Captures temporal preference patterns.

    Input  : (batch, window, emb_dim)
    Output : (batch, hidden_dim)
    """
    def __init__(self, emb_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm       = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(seq)
        return self.norm(h_n[-1])


class FairnessConstraintLayer(nn.Module):
    """
    Step 3 — YOUR NOVELTY.

    Subtracts a learned exposure penalty from actor logits
    BEFORE the action distribution is computed.

    Unlike reward shaping:
      - Acts on logits directly — fairness is IN the policy
      - Penalty strength alpha is LEARNED
      - Per-user sensitivity is LEARNED via user_proj
    """
    def __init__(self, hidden_dim: int, n_items: int):
        super().__init__()
        self.n_items   = n_items
        self.alpha     = nn.Parameter(torch.ones(1))
        self.user_proj = nn.Linear(hidden_dim, 1)

    def forward(self, logits: torch.Tensor,
                state: torch.Tensor,
                item_exposure: torch.Tensor) -> torch.Tensor:
        exp_norm         = item_exposure / (item_exposure.max() + 1e-9)
        fairness_penalty = exp_norm * self.alpha.abs()
        user_sensitivity = torch.sigmoid(self.user_proj(state))
        return logits - user_sensitivity * fairness_penalty.unsqueeze(0)


class AttentionExplainer(nn.Module):
    """
    Step 5 — Attention-based explainability.

    Answers: Which past interactions drove this recommendation?

    Input  : gru_state (batch, hidden_dim)
             history_embs (batch, window, emb_dim)
    Output : attention_weights (batch, window)
    """
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, emb_dim)
        self.scale      = emb_dim ** 0.5

    def forward(self, gru_state: torch.Tensor,
                history_embs: torch.Tensor) -> torch.Tensor:
        query   = self.query_proj(gru_state).unsqueeze(2)
        scores  = torch.bmm(history_embs, query).squeeze(2) / self.scale
        return F.softmax(scores, dim=-1)

    def explain(self, gru_state: torch.Tensor,
                history_embs: torch.Tensor,
                history_item_ids: list) -> dict:
        weights    = self.forward(gru_state, history_embs)
        weights_np = weights.squeeze(0).detach().numpy()
        top_idx    = np.argsort(weights_np)[::-1]
        top_items  = [history_item_ids[i] for i in top_idx
                      if i < len(history_item_ids)]
        top_w      = weights_np[top_idx].tolist()
        return {
            'attention_weights':     weights_np.tolist(),
            'top_history_items':     top_items[:5],
            'top_history_weights':   top_w[:5],
            'most_influential_item': top_items[0],
            'explanation': (
                f"Recommended because of your interactions with "
                f"items {top_items[:3]} "
                f"(influence: {[round(w, 3) for w in top_w[:3]]})"
            )
        }


class ActorCriticPolicy(nn.Module):
    """
    Full policy combining all contributions:

      GRUStateEncoder        — temporal user modelling
      FairnessConstraintLayer — fairness IN the policy (novelty)
      AttentionExplainer     — interpretable recommendations
    """
    def __init__(self, emb_dim: int, n_items: int,
                 hidden_dim: int = 256, num_gru_layers: int = 2):
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
        self.attention      = AttentionExplainer(emb_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.xavier_uniform_(self.fairness_layer.user_proj.weight)
        nn.init.constant_(self.fairness_layer.user_proj.bias, 0.0)

    def forward(self, item_seq: torch.Tensor,
                item_exposure: torch.Tensor) -> tuple:
        """
        item_seq      : (batch, window)   LongTensor — item ids
        item_exposure : (n_items,)        FloatTensor — exposure counts

        Returns
        -------
        fair_logits  : (batch, n_items)
        value        : (batch,)
        gru_state    : (batch, hidden_dim)
        attn_weights : (batch, window)
        """
        emb          = self.item_emb(item_seq)
        gru_state    = self.encoder(emb)
        features     = self.trunk(gru_state)
        logits       = self.actor(features)
        value        = self.critic(features).squeeze(-1)
        fair_logits  = self.fairness_layer(logits, features, item_exposure)
        attn_weights = self.attention(gru_state, emb)
        return fair_logits, value, gru_state, attn_weights

    def select_action(self, item_seq: np.ndarray,
                      item_exposure: np.ndarray,
                      exclude_items: list = None) -> tuple:
        """
        Sample action during training rollout.

        Parameters
        ----------
        item_seq      : (window,) numpy int array
        item_exposure : (n_items,) numpy float array

        Returns: action (int), log_prob (tensor), value (tensor)
        """
        seq_t = torch.LongTensor(item_seq).unsqueeze(0)
        exp_t = torch.FloatTensor(item_exposure)

        logits, value, _, _ = self.forward(seq_t, exp_t)
        logits = logits.clone()

        if exclude_items and len(exclude_items) > 0:
            logits[0, exclude_items] = -1e9

        probs    = F.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(), value.squeeze()

    def greedy_action(self, item_seq: np.ndarray,
                      item_exposure: np.ndarray,
                      exclude_items: list = None) -> int:
        """Deterministic action for evaluation."""
        training = self.training
        self.eval()
        with torch.no_grad():
            seq_t = torch.LongTensor(item_seq).unsqueeze(0)
            exp_t = torch.FloatTensor(item_exposure)
            logits, _, _, _ = self.forward(seq_t, exp_t)
            logits = logits.clone()
            if exclude_items and len(exclude_items) > 0:
                logits[0, exclude_items] = -1e9
            action = logits.argmax(dim=-1).item()
        if training:
            self.train()
        return action

    def explain(self, item_seq: np.ndarray,
                item_exposure: np.ndarray,
                action: int,
                exclude_items: list = None,
                top_k: int = 5) -> dict:
        """
        Full explanation combining attention, saliency, counterfactual.
        """
        seq_t = torch.LongTensor(item_seq).unsqueeze(0)
        exp_t = torch.FloatTensor(item_exposure)

        # Grad w.r.t. embeddings for saliency
        emb_t        = self.item_emb(seq_t).detach().requires_grad_(True)
        gru_state    = self.encoder(emb_t)
        features     = self.trunk(gru_state)
        logits       = self.actor(features)
        fair_logits  = self.fairness_layer(logits, features, exp_t)
        attn_weights = self.attention(gru_state, emb_t)

        fair_logits_masked = fair_logits.clone()
        if exclude_items and len(exclude_items) > 0:
            fair_logits_masked[0, exclude_items] = -1e9

        probs_np = F.softmax(fair_logits_masked.detach(), dim=-1).numpy()[0]

        # 1. Attention
        history_ids = item_seq.tolist()
        attn_exp    = self.attention.explain(
            gru_state.detach(), emb_t.detach(), history_ids)

        # 2. Gradient saliency
        if emb_t.grad is not None:
            emb_t.grad.zero_()
        fair_logits_masked[0, action].backward()
        saliency = emb_t.grad.abs().mean(dim=-1).squeeze().detach().numpy()

        # 3. Counterfactual
        probs_cf         = probs_np.copy()
        probs_cf[action] = 0.0
        top_alt          = np.argsort(probs_cf)[::-1][:top_k]

        return {
            'chosen_item': action,
            'chosen_prob': float(probs_np[action]),
            'attention': {
                'weights':           attn_exp['attention_weights'],
                'top_history_items': attn_exp['top_history_items'],
                'top_weights':       attn_exp['top_history_weights'],
                'explanation':       attn_exp['explanation'],
            },
            'saliency': {
                'scores':             saliency.tolist(),
                'most_important_pos': int(np.argmax(saliency)),
                'explanation': (
                    f"Position {int(np.argmax(saliency))} in history "
                    f"(0=oldest, {len(item_seq)-1}=most recent) "
                    f"had the most influence"
                )
            },
            'counterfactual': {
                'alternatives': top_alt.tolist(),
                'alt_probs':    probs_np[top_alt].tolist(),
                'explanation': (
                    f"If item {action} was not available, "
                    f"item {top_alt[0]} would have been recommended"
                )
            }
        }