import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GRUStateEncoder(nn.Module):
    """Encodes user interaction history into a fixed-size state vector."""

    def __init__(self, item_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(item_emb_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (batch, seq_len, emb_dim)
        _, h_n = self.gru(history)
        return h_n.squeeze(0)   # (batch, hidden_dim)


class ActorCriticPolicy(nn.Module):
    """
    Actor-Critic policy network for recommendation.

    Actor  : scores all items → softmax → recommendation probability
    Critic : estimates state value V(s) for advantage computation
    """

    def __init__(self, state_dim: int, n_items: int, hidden_dim: int = 256):
        super().__init__()
        self.n_items = n_items

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head: outputs logit per item
        self.actor = nn.Linear(hidden_dim, n_items)

        # Critic head: outputs scalar state value
        self.critic = nn.Linear(hidden_dim, 1)

        # Attention weights for explainability (stored during forward pass)
        self.attention_weights = None

    def forward(self, state: torch.Tensor):
        """
        state: (batch, state_dim)
        Returns: action_logits (batch, n_items), state_value (batch, 1)
        """
        features = self.trunk(state)
        logits   = self.actor(features)
        value    = self.critic(features)
        return logits, value

    def select_action(self, state: np.ndarray,
                      exclude_items: list = None) -> tuple:
        """
        Sample an action from the policy.
        Returns: (action, log_prob, state_value)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(state_t)

        # Mask already-seen items so the agent can't re-recommend
        if exclude_items:
            logits[0, exclude_items] = -1e9

        probs    = F.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        return (action.item(),
                log_prob.item(),
                value.item())

    def explain(self, state: np.ndarray, action: int,
                item_embeddings: np.ndarray, top_k: int = 5) -> dict:
        """
        Generate a simple counterfactual explanation:
        'Why was item X recommended instead of item Y?'

        Returns a dict with:
          - chosen_item
          - top_alternatives: items that were close to being chosen
          - feature_importance: which dimensions of the state drove the choice
        """
        state_t  = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(state_t)
        probs    = F.softmax(logits, dim=-1).detach().numpy()[0]

        top_items = np.argsort(probs)[::-1][:top_k]

        # Feature importance via input gradient
        state_t.requires_grad_(True)
        logits2, _ = self.forward(state_t)
        logits2[0, action].backward()
        importance = state_t.grad.abs().squeeze().detach().numpy()

        return {
            'chosen_item':      action,
            'chosen_prob':      float(probs[action]),
            'top_alternatives': top_items.tolist(),
            'alt_probs':        probs[top_items].tolist(),
            'feature_importance': importance.tolist(),
            'state_value':      float(value.item())
        }