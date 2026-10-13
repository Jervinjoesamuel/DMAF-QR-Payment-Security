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
        Computes feature importance gradient E_i and modality weight attribution.
        """
        # Ensure gradients can flow through RNNs under cuDNN
        prev_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        self.model.eval()
        
        # Ensure inputs require grad for feature attribution
        F_modalities = self.model.extract_modalities(img, text, url, behavior, meta)
        F_modalities.retain_grad()

        # Fusion pass
        Z, attn_matrix, mod_weights = self.model.fusion_module(F_modalities)
        Z.retain_grad()

        logits, probs = self.model.classifier_head(Z)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=-1).item())

        # Compute gradient dY / dZ (Eq. 20)
        score = probs[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        dY_dZ = Z.grad.detach().cpu().numpy()[0] if Z.grad is not None else None
        
        # Modality importance through attention weights and gradient magnitude
        weights = mod_weights.detach().cpu().numpy()[0]
        modality_importance = {}
        for name, w in zip(self.MODALITY_NAMES, weights):
            modality_importance[name] = float(w)

        # High-risk flags based on modality dominance
        dominant_modality = self.MODALITY_NAMES[int(weights.argmax())]
        
        torch.backends.cudnn.enabled = prev_cudnn
        
        return {
            "predicted_class": target_class,
            "prediction_confidence": float(score.item()),
            "gradient_importance_vector": dY_dZ,
            "modality_importance_weights": modality_importance,
            "dominant_risk_driver": dominant_modality,
            "cross_modal_attention_matrix": attn_matrix.detach().cpu().numpy()[0],
        }
