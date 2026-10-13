import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-Modal Attention Fusion with Adaptive Modality Weighting (Eqs. 13-16).
    
    1. Cross-modal mutual attention:
       A_ij = exp(F_i F_j^T / sqrt(d)) / sum_k exp(F_i F_k^T / sqrt(d))  [Eq. 13]
       F_fusion = sum_{i=1}^5 A_i F_i                                      [Eq. 14]
       
    2. Adaptive confidence score estimation:
       s_i = g_i(F_i)                                                     [Modality confidence]
       W_i = exp(s_i) / sum_{j=1}^5 exp(s_j)                              [Eq. 15]
       
    3. Final fused embedding:
       Z = sum_{i=1}^5 W_i * F_i_refined                                  [Eq. 16]
    """
    def __init__(self, embed_dim: int = 256, num_modalities: int = 5, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.scale = embed_dim ** -0.5
        
        # Projection layers for cross-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Modality confidence estimation network (computes s_i for each modality)
        self.confidence_estimator = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, modalities: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            modalities: Tensor of shape (B, num_modalities, embed_dim)
                        containing [F_v, F_t, F_u, F_b, F_m]
        Returns:
            Z: Final fused embedding of shape (B, embed_dim) [Eq. 16]
            attention_weights: Cross-modal attention matrix (B, num_modalities, num_modalities) [Eq. 13]
            modality_weights: Adaptive modality weights W_i of shape (B, num_modalities) [Eq. 15]
        """
        B, M, D = modalities.shape
        assert M == self.num_modalities, f"Expected {self.num_modalities} modalities, got {M}"

        # 1. Compute Cross-Modal Attention (Eq. 13)
        Q = self.q_proj(modalities)  # (B, M, D)
        K = self.k_proj(modalities)  # (B, M, D)
        V = self.v_proj(modalities)  # (B, M, D)

        raw_scores = torch.bmm(Q, K.transpose(1, 2)) * self.scale  # (B, M, M)
        attention_weights = F.softmax(raw_scores, dim=-1)           # (B, M, M) - Eq. 13
        attention_weights = self.dropout(attention_weights)

        # Refined cross-modal modalities
        refined = torch.bmm(attention_weights, V)                   # (B, M, D)
        refined = self.norm(modalities + self.dropout(self.out_proj(refined)))

        # 2. Adaptive Modality Weighting (Eq. 15)
        # s_i: Confidence score for each modality
        s = self.confidence_estimator(refined).squeeze(-1)          # (B, M)
        modality_weights = F.softmax(s, dim=-1)                     # (B, M) - Eq. 15

        # 3. Final Weighted Fusion (Eq. 16: Z = \sum_{i=1}^5 W_i F_i)
        W_expanded = modality_weights.unsqueeze(-1)                 # (B, M, 1)
        Z = torch.sum(W_expanded * refined, dim=1)                  # (B, D) - Eq. 16

        return Z, attention_weights, modality_weights
