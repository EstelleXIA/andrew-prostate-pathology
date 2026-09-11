"""Shared loss functions.

Provides:
  * Survival losses, including ``define_loss``.
  * ``AsymmetricLoss`` — multilabel subtype loss.
  * ``MultiClassFocalLoss`` — probability-input focal loss.

Key numeric constants:
  - AsymmetricLoss: gamma_pos=1, gamma_neg=3, m=0.05, final ``* 10`` scale.
  - NLLSurvLoss is instantiated with ``alpha=0.0`` via ``define_loss('nll_surv')``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# Multilabel subtype loss (PathoSubtype)
# --------------------------------------------------------------------------------------
class AsymmetricLoss(torch.nn.Module):
    """Asymmetric focal loss for multilabel classification.

    Negative-sample loss is zeroed where ``sigmoid(pred) < m`` (low-risk negatives),
    and the whole loss is scaled by 10.
    """

    def __init__(self, gamma_pos=1, gamma_neg=3, m=0.05, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.m = m  # negative-probability shift threshold
        self.eps = eps

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pos_loss = -target * torch.pow(1 - pred_sigmoid, self.gamma_pos) * torch.log(pred_sigmoid + self.eps)
        neg_loss = -(1 - target) * torch.pow(pred_sigmoid, self.gamma_neg) * torch.log(1 - pred_sigmoid + self.eps)
        neg_loss = torch.where(pred_sigmoid >= self.m, neg_loss, 0)
        return (pos_loss + neg_loss).mean() * 10


# --------------------------------------------------------------------------------------
# Focal loss over probabilities (GeneExpression / GeneMolecularSubtype)
# --------------------------------------------------------------------------------------
class MultiClassFocalLoss(torch.nn.Module):
    """Multi-class focal loss. Input ``probs`` must already be softmax probabilities."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, probs, target):
        log_p = torch.log(probs + 1e-10)
        ce_loss = -log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_loss = (self.alpha[target] if self.alpha is not None else 1.0) * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# --------------------------------------------------------------------------------------
# Survival losses (BCR)
# --------------------------------------------------------------------------------------
def define_loss(loss_type):
    if loss_type == "ce_surv":
        loss = CrossEntropySurvLoss(alpha=0.0)
    elif loss_type == "nll_surv":
        loss = NLLSurvLoss(alpha=0.0)
    elif loss_type == "cox_surv":
        loss = CoxSurvLoss()
    elif loss_type == "nll_surv_kl":
        print("Use loss of nll_surv_kl")
        loss = [NLLSurvLoss(alpha=0.0), KLLoss()]
    elif loss_type == "nll_surv_mse":
        print("Use loss of nll_surv_mse")
        loss = [NLLSurvLoss(alpha=0.0), nn.MSELoss()]
    elif loss_type == "nll_surv_l1":
        print("Use loss of nll_surv_l1")
        loss = [NLLSurvLoss(alpha=0.0), nn.L1Loss()]
    elif loss_type == "nll_surv_cos":
        print("Use loss of nll_surv_cos")
        loss = [NLLSurvLoss(alpha=0.0), CosineLoss()]
    elif loss_type == "nll_surv_ol":
        print("Use loss of nll_surv_ol")
        loss = [NLLSurvLoss(alpha=0.0), OrthogonalLoss(gamma=0.5)]
    else:
        raise NotImplementedError
    return loss


def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y) + eps) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    ce_l = -c * torch.log(torch.gather(S, 1, Y).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y).clamp(min=eps))
    loss = (1 - alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)


class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


class CoxSurvLoss(object):
    def __call__(hazards, S, c, **kwargs):
        current_batch_len = len(S)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i, j] = S[j] >= S[i]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazards.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * (1 - c))
        return loss_cox


class KLLoss(object):
    def __call__(self, y, y_hat):
        return F.kl_div(y_hat.softmax(dim=-1).log(), y.softmax(dim=-1), reduction="sum")


class CosineLoss(object):
    def __call__(self, y, y_hat):
        return 1 - F.cosine_similarity(y, y_hat, dim=1)


class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, P, P_hat, G, G_hat):
        pos_pairs = (1 - torch.abs(F.cosine_similarity(P.detach(), P_hat, dim=1))) + (
            1 - torch.abs(F.cosine_similarity(G.detach(), G_hat, dim=1))
        )
        neg_pairs = (
            torch.abs(F.cosine_similarity(P, G, dim=1))
            + torch.abs(F.cosine_similarity(P.detach(), G_hat, dim=1))
            + torch.abs(F.cosine_similarity(G.detach(), P_hat, dim=1))
        )
        loss = pos_pairs + self.gamma * neg_pairs
        return loss
