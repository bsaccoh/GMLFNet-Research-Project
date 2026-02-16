"""Fast Adaptation Weights (FAW) module — the primary thesis contribution.

FAW generates per-layer FiLM modulation parameters (gamma, beta) that enable
rapid domain adaptation within the MAML inner loop. By concentrating
adaptation into these lightweight parameters (~100K), the inner loop can
adapt to center-specific characteristics (contrast, color, texture) without
modifying the heavy encoder/decoder weights.

Architecture:
    1. Global Average Pool over multi-scale encoder features
    2. Concatenate pooled statistics into a domain descriptor
    3. Lightweight MLP maps descriptor to modulation parameters
    4. Output: (gamma, beta) per decoder layer for FiLM modulation:
       modulated_feature = gamma * feature + beta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastAdaptationWeights(nn.Module):
    """Fast Adaptation Weights (FAW) module.

    Generates channel-wise modulation parameters for each decoder layer
    based on global statistics of the input. These parameters are the
    primary targets for MAML inner-loop adaptation.

    Args:
        encoder_channels: list of channel sizes from encoder stages
        num_modulation_layers: number of decoder layers to modulate
        modulation_channels: channel size of each modulated feature
        hidden_dim: hidden dimension of the adaptation MLP
        num_layers: number of MLP layers
    """

    def __init__(
        self,
        encoder_channels,
        num_modulation_layers=3,
        modulation_channels=32,
        hidden_dim=64,
        num_layers=2,
    ):
        super().__init__()
        self.num_modulation_layers = num_modulation_layers

        # Total statistics dimension after GAP + concat
        total_stats_dim = sum(encoder_channels)

        # Lightweight adaptation MLP
        layers = []
        in_dim = total_stats_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = hidden_dim

        if num_layers == 1:
            layers.extend([
                nn.Linear(total_stats_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])

        self.stats_encoder = nn.Sequential(*layers)

        # Per-layer gamma and beta prediction heads
        self.gamma_heads = nn.ModuleList([
            nn.Linear(hidden_dim, modulation_channels)
            for _ in range(num_modulation_layers)
        ])
        self.beta_heads = nn.ModuleList([
            nn.Linear(hidden_dim, modulation_channels)
            for _ in range(num_modulation_layers)
        ])

        # Initialize to identity modulation (gamma=1, beta=0)
        self._init_identity()

    def _init_identity(self):
        """Initialize so that initial modulation is identity: gamma=1, beta=0."""
        for head in self.gamma_heads:
            nn.init.zeros_(head.weight)
            nn.init.ones_(head.bias)
        for head in self.beta_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, encoder_features):
        """Generate modulation parameters from encoder features.

        Args:
            encoder_features: list of feature maps [f1, f2, f3, f4]
                each with shape (B, C_i, H_i, W_i)

        Returns:
            modulations: list of (gamma, beta) tuples, one per decoder layer.
                gamma and beta have shape (B, modulation_channels, 1, 1)
        """
        # Extract global statistics via Global Average Pooling
        stats = []
        for feat in encoder_features:
            pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, C_i)
            stats.append(pooled)
        stats = torch.cat(stats, dim=1)  # (B, sum(encoder_channels))

        # Map to hidden representation
        h = self.stats_encoder(stats)  # (B, hidden_dim)

        # Generate per-layer modulation parameters
        modulations = []
        for gamma_head, beta_head in zip(self.gamma_heads, self.beta_heads):
            gamma = gamma_head(h).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = beta_head(h).unsqueeze(-1).unsqueeze(-1)    # (B, C, 1, 1)
            modulations.append((gamma, beta))

        return modulations

    def get_param_count(self):
        """Return the number of trainable parameters in FAW."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
