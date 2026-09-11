"""Shared metrics.

  * quadratic_weighted_kappa (+ confusion_matrix / histogram) — Ben Hamner's
    implementation, with a ``min_rating=0, max_rating=4`` default.
  * calculate_metrics — the multilabel per-class AUC/Acc/F1 + ``Macro_AUC_rare``
    block, parametrized by ``class_names`` / ``rare_class_names``.
  * c_index — thin wrapper over sksurv ``concordance_index_censored`` using the
    ``(1 - censorship).astype(bool)`` convention.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# --------------------------------------------------------------------------------------
# Quadratic weighted kappa (GS_ISUP ordinal grading)
# --------------------------------------------------------------------------------------
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """Returns the confusion matrix between rater's ratings."""
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """Returns the counts of each type of rating that a rater made."""
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=0, max_rating=4):
    """Quadratic weighted kappa (inter-rater agreement).

    ``min_rating=0, max_rating=4`` is the project default (5 ISUP groups mapped 0..4).
    """
    rater_a = np.clip(rater_a, min_rating, max_rating)
    rater_b = np.clip(rater_b, min_rating, max_rating)

    rater_a = np.round(rater_a).astype(int).ravel()
    rater_a[~np.isfinite(rater_a)] = 0
    rater_b = np.round(rater_b).astype(int).ravel()
    rater_b[~np.isfinite(rater_b)] = 0

    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


# --------------------------------------------------------------------------------------
# Multilabel per-class metrics (PathoSubtype / GeneMolecularSubtype infer)
# --------------------------------------------------------------------------------------
def calculate_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray,
                      class_names: Sequence[str],
                      rare_class_names: Optional[Sequence[str]] = None) -> dict:
    """Per-class AUC/Acc/F1 plus a macro-AUC over the "rare" classes.

    Args:
        y_true:      (n_samples, n_classes) binary ground truth.
        y_pred_prob: (n_samples, n_classes) predicted probabilities.
        class_names: column labels, in order.
        rare_class_names: subset of ``class_names`` to average for ``Macro_AUC_rare``.
            Defaults to all classes (matches GeneMolecularSubtype); PathoSubtype passes
            the 3 rare classes ['PDA','IDC-P','NEPC'].
    """
    if rare_class_names is None:
        rare_class_names = list(class_names)

    metrics = {}
    y_pred = (y_pred_prob > 0.5).astype(int)

    for i, name in enumerate(class_names):
        metrics[f'AUC_{name}'] = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
        metrics[f'Acc_{name}'] = accuracy_score(y_true[:, i], y_pred[:, i])
        metrics[f'F1_{name}'] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

    aucs = [metrics[f'AUC_{name}'] for name in rare_class_names]
    metrics['Macro_AUC_rare'] = np.mean(aucs)

    return metrics


# --------------------------------------------------------------------------------------
# Survival concordance index (BCR)
# --------------------------------------------------------------------------------------
def c_index(censorships: np.ndarray, event_times: np.ndarray, risk_scores: np.ndarray,
            tied_tol: float = 1e-08) -> float:
    """Censored concordance index.

    ``censorships`` uses the project convention (1 == censored); the event indicator
    passed to sksurv is therefore ``(1 - censorships).astype(bool)``.
    """
    from sksurv.metrics import concordance_index_censored
    return concordance_index_censored((1 - censorships).astype(bool),
                                      event_times, risk_scores,
                                      tied_tol=tied_tol)[0]
