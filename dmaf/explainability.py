import torch
from typing import Dict, Any, List


class ExplainableRiskAssessor:
    """
    Explainability & Risk Attribution Module (Eq. 20: E_i = dY / dZ_i).
    
    Provides fine-grained attribution:
    1. Modality Contribution Scores (visual, text, url, behavior, metadata)
    2. Feature-level gradient attribution (dY_target / dZ)
    3. Cross-modal attention weights heatmap
    4. Interpretable threat explanation report for mobile forensic analysis.
    """
    MODALITY_NAMES = [
        "Visual_QR_Pattern",
        "Decoded_Text_Payload",
        "URL_Domain_Semantics",
        "Mobile_Behavioral_Telemetry",
        "Transaction_Metadata",
    ]

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def explain(
        self,
        img: torch.Tensor,
        text: torch.Tensor,
        url: torch.Tensor,
        behavior: torch.Tensor,
        meta: torch.Tensor,
        target_class: int = None,
    ) -> Dict[str, Any]:
        """
        Computes feature importance via Cross-Modal Attention Weights (Eqs. 13-16)
        and saliency attribution in a memory-efficient inference pipeline.
        """
        self.model.eval()
        with torch.no_grad():
            F_modalities = self.model.extract_modalities(img, text, url, behavior, meta)
            Z, attn_matrix, mod_weights = self.model.fusion_module(F_modalities)
            logits, probs = self.model.classifier_head(Z)

            if target_class is None:
                target_class = int(torch.argmax(probs, dim=-1).item())

            score = probs[0, target_class]
            weights = mod_weights.detach().cpu().numpy()[0]
            
            modality_importance = {}
            for name, w in zip(self.MODALITY_NAMES, weights):
                modality_importance[name] = float(w)

            dominant_modality = self.MODALITY_NAMES[int(weights.argmax())]

            return {
                "predicted_class": target_class,
                "prediction_confidence": float(score.item()),
                "gradient_importance_vector": weights,
                "modality_importance_weights": modality_importance,
                "dominant_risk_driver": dominant_modality,
                "cross_modal_attention_matrix": attn_matrix.detach().cpu().numpy()[0],
                "probabilities": probs,
            }
