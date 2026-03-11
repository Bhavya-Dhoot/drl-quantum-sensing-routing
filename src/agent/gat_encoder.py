"""
Graph Attention Network (GAT) encoder for quantum network state.

Implements a 3-layer GATv2 encoder that processes graph-structured
observations from the quantum network and produces node/graph embeddings
for the actor-critic policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GATLayer(nn.Module):
    """Single Graph Attention layer (GATv2-style).

    Uses edge features and multi-head attention.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 4,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_dim: Input node feature dimension.
            out_dim: Output per-head dimension.
            edge_dim: Edge feature dimension.
            heads: Number of attention heads.
            concat: If True, concatenate heads; else average.
            dropout: Dropout rate.
        """
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.concat = concat

        # GATv2: apply attention after concatenation
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, heads * out_dim, bias=False)
        self.attn = nn.Parameter(torch.empty(1, heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.attn)

        self.bias = nn.Parameter(torch.zeros(heads * out_dim if concat else out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_dim].
            edge_index: Edge indices [2, E].
            edge_attr: Edge features [E, edge_dim] (optional).

        Returns:
            Updated node features [N, heads*out_dim] or [N, out_dim].
        """
        N = x.size(0)
        E = edge_index.size(1)

        # Transform nodes
        h = self.W(x).view(N, self.heads, self.out_dim)  # [N, H, D]

        # Handle empty edge case
        if E == 0:
            out = h
            if self.concat:
                out = out.view(N, self.heads * self.out_dim) + self.bias
            else:
                out = out.mean(dim=1) + self.bias
            return out

        src, dst = edge_index[0], edge_index[1]

        # Source and destination features for each edge
        h_src = h[src]  # [E, H, D]
        h_dst = h[dst]  # [E, H, D]

        # Edge features contribution
        if edge_attr is not None and edge_attr.size(0) == E:
            e = self.W_edge(edge_attr).view(-1, self.heads, self.out_dim)  # [E, H, D]
            h_src = h_src + e

        # GATv2 attention: a^T · LeakyReLU([h_src || h_dst])
        cat = torch.cat([h_src, h_dst], dim=-1)  # [E, H, 2D]
        attn_scores = (self.leaky_relu(cat) * self.attn).sum(dim=-1)  # [E, H]

        # Per-destination-node max for numerical stability (log-sum-exp trick)
        attn_max = torch.full((N, self.heads), float('-inf'), device=x.device, dtype=x.dtype)
        attn_max.scatter_reduce_(
            0, dst.unsqueeze(1).expand_as(attn_scores), attn_scores, reduce='amax'
        )
        attn_scores = attn_scores - attn_max[dst]  # subtract per-node max
        attn_exp = torch.exp(attn_scores)  # [E, H]

        # Scatter sum for normalisation
        attn_sum = torch.zeros(N, self.heads, device=x.device, dtype=x.dtype)
        attn_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(attn_exp), attn_exp)
        attn_norm = attn_sum[dst] + 1e-10  # [E, H]

        alpha = attn_exp / attn_norm  # [E, H]
        alpha = self.dropout(alpha)

        # Weighted aggregation
        msg = h_src * alpha.unsqueeze(-1)  # [E, H, D]
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        if self.concat:
            out = out.view(N, self.heads * self.out_dim) + self.bias
        else:
            out = out.mean(dim=1) + self.bias

        return out


class GATEncoder(nn.Module):
    """3-layer GAT encoder for quantum network graph state.

    Architecture:
        Layer 1: node_feat_dim → hidden_dim (multi-head, concat)
        Layer 2: hidden_dim → hidden_dim (multi-head, concat)
        Layer 3: hidden_dim → hidden_dim (multi-head, average)
        Readout: mean_pool || max_pool → 2 * hidden_dim
    """

    def __init__(
        self,
        node_feat_dim: int = 6,
        edge_feat_dim: int = 4,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        """
        Args:
            node_feat_dim: Input node feature dimension.
            edge_feat_dim: Edge feature dimension.
            hidden_dim: Hidden dimension.
            num_heads: Number of attention heads per layer.
            num_layers: Number of GAT layers.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_dim = node_feat_dim
            else:
                in_dim = hidden_dim

            concat = (i < num_layers - 1)
            per_head_dim = hidden_dim // num_heads

            self.layers.append(GATLayer(
                in_dim=in_dim,
                out_dim=per_head_dim,
                edge_dim=edge_feat_dim,
                heads=num_heads,
                concat=concat,
                dropout=dropout,
            ))
            out_dim_layer = hidden_dim if concat else per_head_dim * num_heads // num_heads
            # After concat: num_heads * per_head_dim = hidden_dim
            # After average (last layer): per_head_dim
            actual_out = hidden_dim if concat else per_head_dim
            self.norms.append(nn.LayerNorm(actual_out))

        self.activation = nn.ELU()

        # Final projection if last layer uses average
        last_out = hidden_dim // num_heads  # per_head_dim
        self.project = nn.Linear(last_out, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [N, node_feat_dim].
            edge_index: Edge indices [2, E].
            edge_attr: Edge features [E, edge_feat_dim].
            batch: Batch assignment vector [N] (for batched graphs).

        Returns:
            (node_embeddings, graph_embedding):
                node_embeddings: [N, hidden_dim]
                graph_embedding: [B, 2*hidden_dim] (mean||max pool)
        """
        h = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h = layer(h, edge_index, edge_attr)
            h = norm(h)
            h = self.activation(h)

        # Project last layer output to hidden_dim
        h = self.project(h)  # [N, hidden_dim]
        node_embeddings = h

        # Graph-level readout: mean_pool || max_pool
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        num_graphs = int(batch.max().item()) + 1
        mean_pool = torch.zeros(num_graphs, self.hidden_dim, device=h.device)
        max_pool = torch.full((num_graphs, self.hidden_dim), float('-inf'), device=h.device)

        mean_pool.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
        # Count nodes per graph
        node_counts = torch.zeros(num_graphs, device=h.device)
        node_counts.scatter_add_(0, batch, torch.ones(h.size(0), device=h.device))
        mean_pool = mean_pool / node_counts.unsqueeze(1).clamp(min=1)

        # Max pool
        max_pool.scatter_reduce_(
            0, batch.unsqueeze(1).expand_as(h), h, reduce='amax'
        )

        graph_embedding = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 2*hidden_dim]

        return node_embeddings, graph_embedding

    def get_output_dim(self) -> int:
        """Get graph embedding dimension."""
        return 2 * self.hidden_dim
