import torch
import torch.nn as nn


class TransactionMetadataEncoder(nn.Module):
    """
    Transaction & domain metadata encoder (Eq. 11: F_m = f_m(M_p)).
    Encodes payment metadata (amount, timestamp variance, currency MCC, merchant trust,
    certificate verification metrics, domain registration age).
    """
    def __init__(
        self,
        input_dim: int = 16,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.residual = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        res = self.residual(x)
        out = self.net(x)
        return out + res
