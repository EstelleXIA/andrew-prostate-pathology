"""Shared plotting / attribution helpers.

Config-driven ROC and attention-heatmap utilities used by the downstream tasks.
Class names, titles and file paths are passed as arguments, so one implementation
serves every task. KM / log-rank plotting is BCR-specific (see ``tasks/bcr``).
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

_DEFAULT_TITLES = {
    "val": "Validation Cohort (n={n})",
    "test": "NMEV Cohort (n={n})",
    "PLCO": "PLCO Cohort (n={n})",
}


def plot_multiclass_auc(y_true: np.ndarray, y_pred_prob: np.ndarray,
                        save_fig_base: str, split: str,
                        class_names: Sequence[str],
                        colors: Optional[Sequence[str]] = None,
                        plot_micro: bool = False,
                        title_map: Optional[dict] = None) -> dict:
    """One-vs-rest per-class ROC + macro (+ optional micro) average.

    Returns the ``roc_auc`` dict (keys: class index, 'macro', and 'micro' if drawn).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    title_map = title_map or _DEFAULT_TITLES
    if colors is None:
        colors = ['#DF7795', '#FEA809', '#A486C4', '#0AAAAE', '#2EB8DE']

    plt.rcParams['axes.unicode_minus'] = False
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = y_true.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(6, 6))
    for i, color, name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f'{name} (AUC = {roc_auc[i]:.3f})')
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro (AUC = {roc_auc["macro"]:.3f})',
             color='#3367BB', linestyle=':', linewidth=1.5)
    if plot_micro:
        plt.plot(fpr["micro"], tpr["micro"], label=f'Micro (AUC = {roc_auc["micro"]:.3f})',
                 color='#2EB8DE', linestyle=':', linewidth=1.5)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1, label='')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    if split not in title_map:
        raise ValueError(f"No title mapping for split={split!r}")
    plt.title(title_map[split].format(n=y_true.shape[0]), fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_fig_base, exist_ok=True)
    plt.savefig(os.path.join(save_fig_base, f"{split}_auc.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    return roc_auc


def plot_binary_auc(y_true: np.ndarray, y_pred_prob: np.ndarray,
                    save_fig_base: str, split: str,
                    pos_label: str = "Mutant",
                    title_map: Optional[dict] = None) -> float:
    """Single positive-class ROC (GeneExpression binary mutation task).

    Expects one-hot ``y_true``/``y_pred_prob`` with the positive class in column 1.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    title_map = title_map or _DEFAULT_TITLES
    plt.rcParams['axes.unicode_minus'] = False
    fpr, tpr, _ = roc_curve(y_true[:, 1], y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='#DF7795', lw=1.5, label=f'{pos_label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', lw=1, label='')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    if split not in title_map:
        raise ValueError(f"No title mapping for split={split!r}")
    plt.title(title_map[split].format(n=y_true.shape[0]), fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_fig_base, exist_ok=True)
    plt.savefig(os.path.join(save_fig_base, f"{split}_auc.png"), dpi=300, bbox_inches='tight')
    plt.close('all')
    return roc_auc


def export_attention_bins(attention_hat, loc_256, loc_512, wsi_ids,
                          save_base_heatmap: str, patient_name: str, n_bins: int = 15):
    """Persist per-tile attention quantile bins for the winning class.

    ``attention_hat`` is ``(A_256_hat, A_512_hat)``. Reads only the tile-location
    JSONs; the WSI-image overlay rendering (which needs the raw slides) is left to
    task scripts.
    """
    import pandas as pd

    A_256_hat, A_512_hat = attention_hat
    os.makedirs(save_base_heatmap, exist_ok=True)
    for wsi_id in wsi_ids:
        loc_selected_256 = list(filter(lambda x: x.split("_")[1] == str(wsi_id), loc_256))
        loc_selected_512 = list(filter(lambda x: x.split("_")[1] == str(wsi_id), loc_512))

        ig_plot_256 = A_256_hat[np.array([x.split("_")[1] for x in loc_256]) == str(wsi_id)]
        ig_plot_512 = A_512_hat[np.array([x.split("_")[1] for x in loc_512]) == str(wsi_id)]

        ig_plot_bins_256, _ = pd.qcut(ig_plot_256.detach().float().cpu().numpy(), n_bins, retbins=True)
        ig_plot_bins_512, _ = pd.qcut(ig_plot_512.detach().float().cpu().numpy(), n_bins, retbins=True)

        np.savez(os.path.join(save_base_heatmap, f"{patient_name}_256.npz"),
                 index=loc_selected_256, data=ig_plot_bins_256)
        np.savez(os.path.join(save_base_heatmap, f"{patient_name}_512.npz"),
                 index=loc_selected_512, data=ig_plot_bins_512)


def load_tile_locations(json_path_256: str, json_path_512: str, patient_name: str):
    """Load the 256/512 tile-location JSONs and the WSIs common to both scales."""
    with open(os.path.join(json_path_256, f"{patient_name}.json")) as f:
        loc_256 = json.load(f)
    with open(os.path.join(json_path_512, f"{patient_name}.json")) as f:
        loc_512 = json.load(f)
    wsi_256 = list(set([x.split("_")[1] for x in loc_256]))
    wsi_512 = list(set([x.split("_")[1] for x in loc_512]))
    common_wsis = sorted(list(set(wsi_256).intersection(wsi_512)))
    return loc_256, loc_512, common_wsis
