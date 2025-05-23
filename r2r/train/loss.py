import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Loss(nn.Module, ABC):
    @abstractmethod
    def forward(self, inputs, targets):
        """
        Compute loss.

        Inputs:
            inputs: (batch_size, num_classes (default: 1))
            targets: (batch_size, num_classes)
        """
        pass

class FocalLoss(nn.Module):
    """Focal Loss for better handling of hard examples."""

    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (
            1 - targets
        )  # alpha for y=1, 1-alpha for y=0
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        return torch.mean(focal_loss)


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    BCEWithLogitsLoss with optional class weighting to improve recall.
    """
    def __init__(self, recall_factor=None, pos_weight=None, reduction='mean'):
        super().__init__()
        self.recall_factor = recall_factor
        self.reduction = reduction
        
        # If pos_weight is directly provided, use it
        if pos_weight is not None:
            if isinstance(pos_weight, float):
                self.pos_weight = torch.tensor([pos_weight])
            else:
                self.pos_weight = pos_weight
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction=self.reduction)
        else:
            # Default BCE loss without weighting
            self.criterion = nn.BCEWithLogitsLoss(reduction=self.reduction)
            self.pos_weight = None
    
    def update_weight_from_dataset(self, dataset_labels):
        """
        Update the positive class weight based on dataset statistics.
        
        Args:
            dataset_labels: A list or tensor of binary labels from the dataset
        """
        if self.recall_factor is not None:
            # Count class frequencies
            if isinstance(dataset_labels, torch.Tensor):
                n_class_0 = (dataset_labels == 0).sum().item()
                n_class_1 = (dataset_labels == 1).sum().item()
            else:
                n_class_0 = sum(1 for x in dataset_labels if x == 0)
                n_class_1 = sum(1 for x in dataset_labels if x == 1)
            
            # Calculate weight with recall factor for the positive class
            pos_weight = (n_class_0 / n_class_1) * self.recall_factor if n_class_1 > 0 else 1.0
            self.pos_weight = torch.tensor([pos_weight], device=dataset_labels.device if isinstance(dataset_labels, torch.Tensor) else None)
            
            # Update the criterion
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
    
    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)


class KLDivergenceLoss(nn.Module):
    """
    KL Divergence loss between original model and fine-tuned model probabilities.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, logits_student, logits_teacher, mask=None):
        """
        Compute KL divergence loss.
        
        Args:
            logits_student: Logits from student model (fine-tuned)
            logits_teacher: Logits from teacher model (original)
            mask: Optional mask to apply (1 for tokens to consider, 0 for tokens to ignore)
            
        Returns:
            KL divergence loss
        """
        # Convert logits to log probabilities and probabilities
        log_probs_student = F.log_softmax(logits_student, dim=-1)
        probs_teacher = F.softmax(logits_teacher, dim=-1)
        
        # Compute KL divergence: KL(teacher || student)
        kl_div = F.kl_div(log_probs_student, probs_teacher, reduction='none').sum(-1)
        
        # Apply mask if provided
        if mask is not None:
            # Reshape mask to match kl_div shape
            mask = mask.view(kl_div.shape)
            # Apply mask and average (avoiding division by zero)
            mask_sum = mask.sum()
            if mask_sum > 0:
                kl_div = (kl_div * mask).sum() / mask_sum
            else:
                kl_div = torch.tensor(0.0, device=kl_div.device)
        else:
            kl_div = kl_div.mean()
            
        return kl_div

class DPOLoss(nn.Module):
    """
    DPO Loss for better handling of hard examples.
    """
    def __init__(self, beta=1, positive_prob=0.5):
        super().__init__()
        self.beta = beta
        self.positive_prob = positive_prob
        self.sigmoid = nn.Sigmoid()
        self.epsilon = 1e-10

    def forward(self, inputs, targets):
        """
        Compute DPO loss.
        """
        positive_probs = self.sigmoid(inputs)
        win_probs = torch.where(targets == 1, positive_probs, 1 - positive_probs)
        lost_probs = 1 - win_probs

        win_action_prob = torch.where(targets == 1, self.positive_prob, 1 - self.positive_prob)
        lost_action_prob = 1 - win_action_prob

        loss = -torch.mean(
            torch.log(
                self.sigmoid(
                    self.beta * torch.log(win_probs + self.epsilon) / win_action_prob
                    - self.beta * torch.log(lost_probs + self.epsilon) / lost_action_prob
                )
                + self.epsilon
            )
        )

        return loss
