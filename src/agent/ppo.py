"""
PPO (Proximal Policy Optimization) actor-critic agent for graph-based routing.

Implements clipped PPO with:
- GAT-based graph encoder
- Actor head with action masking
- Value head for advantage estimation
- GAE for advantage computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .gat_encoder import GATEncoder


class ActorCritic(nn.Module):
    """Actor-Critic network with GATv2 encoder.

    Architecture:
        GATEncoder → graph_embedding (2*hidden_dim)
        Actor:  graph_embedding + node_embedding[current] → Linear(256, action_dim) + softmax + mask
        Critic: graph_embedding → Linear(256, 1)
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        max_action_dim: int = 10,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
        use_mlp_encoder: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_action_dim = max_action_dim
        self.use_mlp_encoder = use_mlp_encoder

        if use_mlp_encoder:
            # MLP encoder (for ablation study)
            self.encoder = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
            )
            encoder_out_dim = 2 * hidden_dim  # mean || max
        else:
            # GAT encoder
            self.encoder = GATEncoder(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
            encoder_out_dim = 2 * hidden_dim

        # Actor: graph_embed + current_node_embed → action logits
        actor_in_dim = encoder_out_dim + hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(actor_in_dim, 256),
            nn.ELU(),
            nn.Linear(256, max_action_dim),
        )

        # Critic: graph_embed → value
        self.critic = nn.Sequential(
            nn.Linear(encoder_out_dim, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        action_mask: torch.Tensor,
        current_node: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value.

        Args:
            node_features: [N, node_feat_dim]
            edge_index: [2, E]
            edge_features: [E, edge_feat_dim]
            action_mask: [B, max_action_dim]
            current_node: [B] node indices
            batch: [N] batch assignment

        Returns:
            (action_probs, values): [B, max_action_dim], [B, 1]
        """
        if self.use_mlp_encoder:
            h = self.encoder(node_features)  # [N, hidden_dim]
            # Simple mean/max pool per graph
            if batch is None:
                batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            num_g = int(batch.max().item()) + 1
            mean_p = torch.zeros(num_g, self.hidden_dim, device=h.device)
            mean_p.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
            counts = torch.zeros(num_g, device=h.device)
            counts.scatter_add_(0, batch, torch.ones(h.size(0), device=h.device))
            mean_p /= counts.unsqueeze(1).clamp(min=1)
            max_p = torch.full((num_g, self.hidden_dim), float('-inf'), device=h.device)
            max_p.scatter_reduce_(0, batch.unsqueeze(1).expand_as(h), h, reduce='amax')
            graph_emb = torch.cat([mean_p, max_p], dim=-1)
            node_emb = h
        else:
            node_emb, graph_emb = self.encoder(
                node_features, edge_index, edge_features, batch
            )

        # Get current node embeddings
        # current_node indices are into the local graph, offset by batch
        current_emb = node_emb[current_node]  # [B, hidden_dim]

        # Actor
        actor_input = torch.cat([graph_emb, current_emb], dim=-1)
        logits = self.actor(actor_input)  # [B, max_action_dim]

        # Apply action mask — guard against all-zero masks
        has_valid = action_mask.sum(dim=-1, keepdim=True) > 0  # [B, 1]
        safe_mask = torch.where(has_valid, action_mask, torch.ones_like(action_mask))
        logits = logits.masked_fill(safe_mask == 0, float('-inf'))
        action_probs = F_torch.softmax(logits, dim=-1)

        # NaN guard: replace any NaN probs with uniform
        nan_mask = torch.isnan(action_probs).any(dim=-1, keepdim=True)
        if nan_mask.any():
            uniform = torch.ones_like(action_probs) / action_probs.size(-1)
            action_probs = torch.where(nan_mask, uniform, action_probs)

        # Critic
        values = self.critic(graph_emb)  # [B, 1]

        return action_probs, values

    def get_action(
        self,
        obs: Dict[str, Any],
        deterministic: bool = False,
        device: torch.device = torch.device('cpu'),
    ) -> Tuple[int, float, float]:
        """Select an action given an observation dict.

        Args:
            obs: Observation dict from environment.
            deterministic: If True, select argmax action.
            device: Computation device.

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            nf = torch.FloatTensor(obs['node_features']).unsqueeze(0).to(device)
            ei = torch.LongTensor(obs['edge_index']).to(device)
            ef = torch.FloatTensor(obs['edge_features']).to(device)
            am = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)
            cn = torch.LongTensor([obs['current_node']]).to(device)

            # Need to handle batch dimension for single graph
            nf = nf.squeeze(0)  # [N, feat_dim]

            probs, value = self.forward(nf, ei, ef, am, cn, batch=None)

            # Ensure valid probability distribution
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=-1, keepdim=True)

            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action_t = dist.sample()
                action = action_t.item()

            log_prob = torch.log(probs[0, action] + 1e-10).item()
            val = value.item()

        return action, log_prob, val

    def evaluate_actions(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        action_mask: torch.Tensor,
        current_node: torch.Tensor,
        actions: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            (log_probs, values, entropy)
        """
        probs, values = self.forward(
            node_features, edge_index, edge_features, action_mask, current_node, batch
        )

        # Clamp for numerical stability
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """Buffer for storing rollout data during PPO training.

    Memory-bounded: stores at most max_size transitions.
    """

    def __init__(self, max_size: int = 4096):
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.observations: List[Dict] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def add(self, obs: Dict, action: int, log_prob: float,
            reward: float, value: float, done: bool):
        if len(self.observations) >= self.max_size:
            # Ring-buffer behavior: drop oldest
            self.observations.pop(0)
            self.actions.pop(0)
            self.log_probs.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.dones.pop(0)

        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    @property
    def size(self) -> int:
        return len(self.observations)

    def compute_gae(self, gamma: float = 0.99, gae_lambda: float = 0.95,
                    last_value: float = 0.0):
        """Compute Generalised Advantage Estimation."""
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 1.0
            else:
                next_value = self.values[t + 1]
                next_done = 1.0 - float(self.dones[t + 1])

            delta = (self.rewards[t]
                     + gamma * next_value * next_done
                     - self.values[t])
            last_gae = delta + gamma * gae_lambda * next_done * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int = 512):
        """Yield random mini-batches of transitions."""
        n = len(self.observations)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            yield batch_idx


class PPOAgent:
    """PPO training agent.

    Handles the training loop, GAE computation, and policy updates.
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        max_action_dim: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        batch_size: int = 512,
        rollout_length: int = 256,
        buffer_size: int = 4096,
        use_mlp: bool = False,
        device: str = 'auto',
    ):
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.rollout_length = rollout_length

        # Model
        self.model = ActorCritic(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            max_action_dim=max_action_dim,
            use_mlp_encoder=use_mlp,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Rollout buffer
        self.buffer = RolloutBuffer(max_size=buffer_size)

        # Training statistics
        self.total_steps = 0
        self.total_updates = 0

    def select_action(self, obs: Dict, deterministic: bool = False) -> Tuple[int, float, float]:
        """Select an action using the current policy."""
        return self.model.get_action(obs, deterministic=deterministic, device=self.device)

    def store_transition(self, obs: Dict, action: int, log_prob: float,
                         reward: float, value: float, done: bool):
        """Store a transition in the rollout buffer."""
        self.buffer.add(obs, action, log_prob, reward, value, done)
        self.total_steps += 1

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """Perform PPO update using buffered data.

        Returns:
            Dict of loss statistics.
        """
        self.buffer.compute_gae(self.gamma, self.gae_lambda, last_value)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.num_epochs):
            for batch_idx in self.buffer.get_batches(self.batch_size):
                batch_obs = [self.buffer.observations[i] for i in batch_idx]
                batch_actions = torch.LongTensor(
                    [self.buffer.actions[i] for i in batch_idx]
                ).to(self.device)
                batch_old_logp = torch.FloatTensor(
                    [self.buffer.log_probs[i] for i in batch_idx]
                ).to(self.device)
                batch_advantages = torch.FloatTensor(
                    self.buffer.advantages[batch_idx]
                ).to(self.device)
                batch_returns = torch.FloatTensor(
                    self.buffer.returns[batch_idx]
                ).to(self.device)

                # Normalise advantages
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                    batch_advantages.std() + 1e-8
                )

                # Collate graph observations into a batch
                (nf, ei, ef, am, cn, batch_vec) = self._collate_obs(batch_obs)

                # Evaluate actions
                new_logp, values, entropy = self.model.evaluate_actions(
                    nf, ei, ef, am, cn, batch_actions, batch_vec
                )

                # PPO clipped objective
                ratio = torch.exp(new_logp - batch_old_logp)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F_torch.mse_loss(values, batch_returns)

                # Total loss
                loss = (policy_loss
                        + self.value_loss_coeff * value_loss
                        - self.entropy_coeff * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        self.total_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        n_updates = max(n_updates, 1)
        return {
            'policy_loss': total_policy_loss / n_updates,
            'value_loss': total_value_loss / n_updates,
            'entropy': total_entropy / n_updates,
        }

    def _collate_obs(self, obs_list: List[Dict]) -> Tuple:
        """Collate a list of graph observations into a batch.

        Returns:
            (node_features, edge_index, edge_features, action_mask,
             current_nodes, batch_vector)
        """
        all_nf = []
        all_ei_src = []
        all_ei_dst = []
        all_ef = []
        all_am = []
        all_cn = []
        all_batch = []

        node_offset = 0
        for i, obs in enumerate(obs_list):
            nf = obs['node_features']
            ei = obs['edge_index']
            ef = obs['edge_features']
            am = obs['action_mask']
            cn = obs['current_node']

            n_nodes = nf.shape[0]
            all_nf.append(nf)
            all_ei_src.append(ei[0] + node_offset)
            all_ei_dst.append(ei[1] + node_offset)
            all_ef.append(ef)
            all_am.append(am)
            all_cn.append(int(cn) + node_offset)
            all_batch.append(np.full(n_nodes, i, dtype=np.int64))

            node_offset += n_nodes

        nf = torch.FloatTensor(np.concatenate(all_nf, axis=0)).to(self.device)
        ei = torch.LongTensor(np.array([
            np.concatenate(all_ei_src),
            np.concatenate(all_ei_dst),
        ])).to(self.device)
        ef = torch.FloatTensor(np.concatenate(all_ef, axis=0)).to(self.device)
        am = torch.FloatTensor(np.stack(all_am)).to(self.device)
        cn = torch.LongTensor(all_cn).to(self.device)
        batch = torch.LongTensor(np.concatenate(all_batch)).to(self.device)

        return nf, ei, ef, am, cn, batch

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_updates': self.total_updates,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_updates = checkpoint.get('total_updates', 0)
