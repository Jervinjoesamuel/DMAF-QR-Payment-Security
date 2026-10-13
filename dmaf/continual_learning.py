import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class ContinualLearningEngine:
    """
    Continual Learning & Adaptive Decision Optimization (Eqs. 21 - 24).
    
    1. Online incremental learning without catastrophic forgetting (Eq. 21).
    2. Elastic Weight Consolidation (EWC) Fisher regularization for catastrophic forgetting mitigation.
    3. Adaptive confidence score thresholding (Eqs. 22-23):
       C = max(Y)
       Decision = Legitimate if (C >= tau and y == 0) else Malicious
    4. Weight-decay regularized optimization objective (Eq. 24).
    """
    def __init__(
        self,
        model: nn.Module,
        tau_threshold: float = 0.85,
        continual_lr: float = 5e-5,
        lambda_reg: float = 1e-4,
        ewc_lambda: float = 400.0,
    ):
        self.model = model
        self.tau = tau_threshold
        self.continual_lr = continual_lr
        self.lambda_reg = lambda_reg
        self.ewc_lambda = ewc_lambda
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=continual_lr, weight_decay=lambda_reg
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Stored parameters and Fisher diagonal for EWC
        self.stored_params: Dict[str, torch.Tensor] = {}
        self.fisher_diagonal: Dict[str, torch.Tensor] = {}

    def authenticate(self, probabilities: torch.Tensor, class_idx: int) -> Tuple[str, float]:
        """
        Decision rule (Eqs. 22-23):
        C = max Y
        D = Legitimate if (C >= tau and class_idx == 0) else Malicious
        """
        conf = float(torch.max(probabilities).item())
        is_benign = (class_idx == 0)
        
        if is_benign and conf >= self.tau:
            decision = "Legitimate"
        else:
            decision = "Malicious"
            
        return decision, conf

    def register_ewc_task(self, dataloader: torch.utils.data.DataLoader, device: str = "cpu"):
        """
        Calculates the diagonal of the empirical Fisher information matrix
        to preserve critical synaptic weights during continual updates.
        """
        self.model.eval()
        self.stored_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.fisher_diagonal = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        total_samples = 0
        for batch in dataloader:
            img = batch["img"].to(device)
            text = batch["text"].to(device)
            url = batch["url"].to(device)
            behavior = batch["behavior"].to(device)
            meta = batch["meta"].to(device)
            target = batch["label"].to(device)

            self.model.zero_grad()
            out = self.model(img, text, url, behavior, meta)
            loss = self.criterion(out["logits"], target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None and name in self.fisher_diagonal:
                    self.fisher_diagonal[name] += (param.grad.data ** 2) * img.size(0)

            total_samples += img.size(0)

        for name in self.fisher_diagonal:
            self.fisher_diagonal[name] /= max(1, total_samples)

    def incremental_update_step(
        self, batch: Dict[str, torch.Tensor], device: str = "cpu"
    ) -> float:
        """
        Performs incremental update step (Eq. 21 & 24) on newly observed stream of QR threats.
        """
        self.model.train()
        self.optimizer.zero_grad()

        img = batch["img"].to(device)
        text = batch["text"].to(device)
        url = batch["url"].to(device)
        behavior = batch["behavior"].to(device)
        meta = batch["meta"].to(device)
        target = batch["label"].to(device)

        out = self.model(img, text, url, behavior, meta)
        ce_loss = self.criterion(out["logits"], target)

        # EWC catastrophic forgetting penalty
        ewc_loss = torch.tensor(0.0, device=device)
        if self.stored_params:
            for name, param in self.model.named_parameters():
                if name in self.stored_params:
                    diff = param - self.stored_params[name]
                    ewc_loss += (self.fisher_diagonal[name] * (diff ** 2)).sum()
                    
        total_loss = ce_loss + (self.ewc_lambda / 2.0) * ewc_loss
        total_loss.backward()
        
        # Gradient clipping for stable stream learning
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(total_loss.item())
