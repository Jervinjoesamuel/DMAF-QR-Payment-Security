import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from dmaf.config import ModelConfig
from dmaf.models.vision_encoder import LightweightViT
from dmaf.models.text_encoder import SemanticSequenceEncoder
from dmaf.models.behavioral_encoder import TemporalBehavioralEncoder
from dmaf.models.transaction_encoder import TransactionMetadataEncoder
from dmaf.models.attention_fusion import CrossModalAttentionFusion


class ResidualClassificationHead(nn.Module):
    """
    Uncertainty-aware residual classification head (Eq. 17: Y = Softmax(W_c Z + b_c)).
    """
    def __init__(self, in_features: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.drop = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Residual projection
        h = self.drop(self.act1(self.bn1(self.fc1(z))))
        h = h + self.drop(self.act2(self.bn2(self.fc2(h))))
        logits = self.classifier(h)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class DMAFNetwork(nn.Module):
    """
    Deep Multimodal Authentication Framework (DMAF) End-to-End Network.
    
    Integrates 5 Heterogeneous Modalities:
    1. QR Visual Image (I_p) -> Lightweight ViT -> F_v
    2. Decoded Text Payload (T_p) -> Semantic Text Encoder -> F_t
    3. URL/Domain Semantics (U) -> URL Encoder -> F_u
    4. Mobile Behavioral Attributes (B_p) -> Temporal Attention Encoder -> F_b
    5. Transaction Metadata (M_p) -> Metadata Projection -> F_m
    
    Followed by:
    - Cross-Modal Attention Fusion (Eqs. 13-16) -> Z
    - Deep Threat Classification Head (Eqs. 17-18) -> Y
    """
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        d = self.config.embed_dim

        # Modality Encoders (Eqs. 7 - 11)
        self.vision_encoder = LightweightViT(
            img_size=self.config.img_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=d,
            depth=self.config.vit_depth,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        
        self.text_encoder = SemanticSequenceEncoder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.text_hidden_dim,
            out_dim=d,
            max_len=self.config.max_text_len,
            dropout=self.config.dropout,
        )
        
        self.url_encoder = SemanticSequenceEncoder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.text_hidden_dim,
            out_dim=d,
            max_len=self.config.max_url_len,
            dropout=self.config.dropout,
        )
        
        self.behavior_encoder = TemporalBehavioralEncoder(
            input_dim=self.config.behavior_feature_dim,
            seq_len=self.config.behavior_seq_len,
            embed_dim=d,
            num_heads=4,
            dropout=self.config.dropout,
        )
        
        self.meta_encoder = TransactionMetadataEncoder(
            input_dim=self.config.meta_feature_dim,
            embed_dim=d,
            hidden_dim=128,
            dropout=self.config.dropout,
        )

        # Cross-Modal Attention Fusion (Eqs. 13-16)
        self.fusion_module = CrossModalAttentionFusion(
            embed_dim=d,
            num_modalities=5,
            dropout=self.config.dropout,
        )

        # Deep Threat Classifier (Eqs. 17-18)
        self.classifier_head = ResidualClassificationHead(
            in_features=d,
            hidden_dim=self.config.hidden_dims[0],
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )

    def extract_modalities(
        self,
        img: torch.Tensor,
        text: torch.Tensor,
        url: torch.Tensor,
        behavior: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extracts representations for all 5 modalities (Eqs. 7-12).
        Returns tensor F of shape (B, 5, embed_dim)
        """
        F_v = self.vision_encoder(img)            # (B, d) - Eq. 7
        F_t = self.text_encoder(text)             # (B, d) - Eq. 8
        F_u = self.url_encoder(url)               # (B, d) - Eq. 9
        F_b = self.behavior_encoder(behavior)     # (B, d) - Eq. 10
        F_m = self.meta_encoder(meta)             # (B, d) - Eq. 11
        
        # Multimodal feature tensor (B, 5, d) - Eq. 12
        F_modalities = torch.stack([F_v, F_t, F_u, F_b, F_m], dim=1)
        return F_modalities

    def forward(
        self,
        img: torch.Tensor,
        text: torch.Tensor,
        url: torch.Tensor,
        behavior: torch.Tensor,
        meta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-End Forward Pass.
        """
        # 1. Feature Representation Learning (Eqs. 7-12)
        F_modalities = self.extract_modalities(img, text, url, behavior, meta)

        # 2. Cross-Modal Attention Fusion (Eqs. 13-16)
        Z, attention_weights, modality_weights = self.fusion_module(F_modalities)

        # 3. Threat Classification (Eqs. 17-18)
        logits, probs = self.classifier_head(Z)

        return {
            "logits": logits,
            "probabilities": probs,
            "fused_embedding": Z,
            "modalities_features": F_modalities,
            "cross_modal_attention": attention_weights,
            "modality_weights": modality_weights,
        }
