import torch
import torch.nn as nn


class TemporalBehavioralEncoder(nn.Module):
    """
    Temporal-attention encoder for mobile behavioral features (Eq. 10: F_b = f_b(B_p)).
    Captures temporal sensor fluctuations, interaction rhythms, scan angle dynamics,
    and atypical touch hold times.
    """
    def __init__(
        self,
        input_dim: int = 8,
        seq_len: int = 10,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        self.pool_weights = nn.Linear(embed_dim, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_dim) or (B, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
            
        h = self.input_proj(x) + self.pos_embed
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        
        # Temporal attention pooling
        scores = torch.softmax(self.pool_weights(h), dim=1)  # (B, seq_len, 1)
        pooled = torch.sum(scores * h, dim=1)                 # (B, embed_dim)
        return pooled
