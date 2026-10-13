import torch
import torch.nn as nn


class SemanticSequenceEncoder(nn.Module):
    """
    Contextual language & URL encoder (Eq. 8: F_t = f_t(T_p) and Eq. 9: F_u = f_u(U)).
    Combines subword embedding, multi-scale 1D convolutional feature extractors, 
    and bidirectional recurrent pooling to capture character-level URL homoglyphs,
    suspicious URI schemes, and obfuscated domain patterns.
    """
    def __init__(
        self,
        vocab_size: int = 5000,
        embed_dim: int = 128,
        out_dim: int = 256,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Multi-scale character n-gram filters (captures token patterns: 3-grams, 5-grams, 7-grams)
        self.conv3 = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=7, padding=3)
        
        conv_out_dim = (embed_dim // 2) * 3
        self.bilstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=out_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        
        self.proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len)
        emb = self.embedding(x).transpose(1, 2)  # (B, embed_dim, seq_len)
        
        c3 = torch.relu(self.conv3(emb))
        c5 = torch.relu(self.conv5(emb))
        c7 = torch.relu(self.conv7(emb))
        feat = torch.cat([c3, c5, c7], dim=1).transpose(1, 2)  # (B, seq_len, conv_out_dim)
        
        lstm_out, (hn, _) = self.bilstm(feat)
        # Concatenate forward and backward final hidden states: (B, out_dim)
        h_forward = hn[-2, :, :]
        h_backward = hn[-1, :, :]
        pooled = torch.cat([h_forward, h_backward], dim=1)
        
        return self.proj(pooled)
