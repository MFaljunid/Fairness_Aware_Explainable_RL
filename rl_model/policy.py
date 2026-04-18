import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GRUStateEncoder(nn.Module):
    """Encodes user interaction history into a fixed-size state vector."""

    def __init__(self, item_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gru        = nn.GRU(item_emb_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (batch, seq_len, emb_dim)
        _, h_n = self.gru(history)
        return h_n.squeeze(0)   # (batch, hidden_dim)


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic policy for fair recommendation.

    Actor  : state → logits over all items → Categorical distribution
    Critic : state → scalar V(s) for advantage estimation
    """

    def __init__(self, state_dim: int, n_items: int, hidden_dim: int = 256):
        super().__init__()
        self.n_items    = n_items
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim

        # Shared trunk — processes state into features
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: one logit per item
        self.actor  = nn.Linear(hidden_dim, n_items)

        # Critic head: scalar state value
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init — helps Actor-Critic converge faster."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor output layer uses smaller gain so initial policy is near-uniform
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, state: torch.Tensor):
        """
        state : (batch, state_dim)
        returns: logits (batch, n_items),  value (batch,)   ← 1-D, not 2-D
        """
        features = self.trunk(state)
        logits   = self.actor(features)
        value    = self.critic(features).squeeze(-1)   # (batch,) not (batch,1)
        return logits, value

    def select_action(self, state: np.ndarray,
                      exclude_items: list = None) -> tuple:
        """
        Sample one action from the policy during a rollout.

        Returns
        -------
        action   : int               — item index to recommend
        log_prob : torch.Tensor      — scalar tensor WITH grad_fn (for training)
        value    : torch.Tensor      — scalar tensor WITH grad_fn (for training)
        """
        state_t        = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)
        logits, value  = self.forward(state_t)                  # (1, n_items), (1,)

        # Clone before masking — avoids in-place mutation of the grad graph
        logits = logits.clone()
        if exclude_items and len(exclude_items) > 0:
            logits[0, exclude_items] = -1e9

        probs    = F.softmax(logits, dim=-1)                    # (1, n_items)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()                                # (1,)
        log_prob = dist.log_prob(action)                        # (1,) with grad_fn

        return (
            action.item(),           # plain int — safe array index
            log_prob.squeeze(),      # 0-dim tensor WITH grad_fn
            value.squeeze()          # 0-dim tensor WITH grad_fn
        )

    def greedy_action(self, state: np.ndarray,
                      exclude_items: list = None) -> int:
        """
        Pick the highest-scoring item deterministically.
        Use this during EVALUATION only — no gradients needed.
        """
        with torch.no_grad():
            state_t       = torch.FloatTensor(state).unsqueeze(0)
            logits, _     = self.forward(state_t)
            logits        = logits.clone()
            if exclude_items and len(exclude_items) > 0:
                logits[0, exclude_items] = -1e9
            return logits.argmax(dim=-1).item()

    def explain(self, state: np.ndarray, action: int,
                top_k: int = 5) -> dict:
        """
        Counterfactual explanation for a recommendation.

        Answers: 'Why item X and not item Y?'

        Returns
        -------
        dict with:
          chosen_item       : int
          chosen_prob       : float
          top_alternatives  : list[int]   — next best items
          alt_probs         : list[float]
          feature_importance: list[float] — which state dims drove the choice
          state_value       : float       — how good this state was rated
        """
        # Single forward pass — compute everything from one graph
        state_t = torch.FloatTensor(state).unsqueeze(0).requires_grad_(True)
        logits, value = self.forward(state_t)
        probs_t       = F.softmax(logits, dim=-1)              # (1, n_items)

        probs_np  = probs_t.detach().numpy()[0]
        top_items = np.argsort(probs_np)[::-1][:top_k]

        # Gradient of chosen item's logit w.r.t. state — feature importance
        # Zero any existing grads first
        if state_t.grad is not None:
            state_t.grad.zero_()

        logits[0, action].backward(retain_graph=False)
        importance = state_t.grad.abs().squeeze().detach().numpy()

        return {
            'chosen_item':        action,
            'chosen_prob':        float(probs_np[action]),
            'top_alternatives':   top_items.tolist(),
            'alt_probs':          probs_np[top_items].tolist(),
            'feature_importance': importance.tolist(),
            'state_value':        float(value.detach().item()),
        }