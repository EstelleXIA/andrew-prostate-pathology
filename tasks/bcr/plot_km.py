"""Kaplan-Meier + log-rank / Cox HR plots from stage-2 risk CSVs.

Run as::

    python -m tasks.bcr.plot_km --model_type andrew --arch_type final

Consumes ``save_risk_pd_<model>_<arch>.csv`` (written by ``infer_stage_2``),
splits each cohort at its median risk into High/Low groups, and writes one KM PDF
per cohort with the Cox hazard ratio (95% CI) and p-value annotated.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

from common import get_paths

from . import config

NAME_CODEBOOK = {"val": "Validation", "test": "NMEV", "PLCO": "PLCO"}


def main():
    parser = argparse.ArgumentParser(description="Arguments for BCR Kaplan-Meier plots.")
    parser.add_argument("--model_type", type=str, default="andrew", choices=config.MODEL_CHOICES)
    parser.add_argument("--arch_type", type=str, default="final", choices=config.ARCH_CHOICES)
    parser.add_argument("--risk_csv", type=str, default=None,
                        help="input risk CSV; defaults to "
                             "patients_cache/<TASK_DIR>/risk_pd/save_risk_pd_<model>_<arch>.csv.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="output dir for KM PDFs; defaults to "
                             "patients_cache/<TASK_DIR>/figures/<model>_<arch>/.")
    args = parser.parse_args()

    paths = get_paths()
    risk_csv = args.risk_csv or paths.patients_cache(
        config.TASK_DIR, "risk_pd", f"save_risk_pd_{args.model_type}_{args.arch_type}.csv")
    save_path = args.save_dir or paths.patients_cache(
        config.TASK_DIR, "figures", f"{args.model_type}_{args.arch_type}")
    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(risk_csv, index_col=0)

    for split in ["val", "test", "PLCO"]:
        selected_data = data[data["split"] == split]
        median_risk = selected_data["risk"].median()
        selected_data.loc[selected_data["risk"] > median_risk, "group"] = "High Risk"
        selected_data.loc[selected_data["risk"] <= median_risk, "group"] = "Low Risk"
        selected_data_plot = selected_data[["time", "event", "group"]]

        kmf_1 = KaplanMeierFitter()
        kmf_2 = KaplanMeierFitter()
        plt.rcParams['font.size'] = 20
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        fig, ax = plt.subplots(figsize=(12, 8))

        censor_styles = {'ms': 6, 'marker': '|', 'mew': 1.5}

        group_data_1 = selected_data_plot[selected_data_plot['group'] == "High Risk"]
        kmf_1.fit(group_data_1['time'], event_observed=group_data_1['event'], label="High Risk")
        kmf_1.plot_survival_function(ax=ax, ci_show=False, color="#DF7795", linewidth=2,
                                     show_censors=True, censor_styles=censor_styles)

        group_data_2 = selected_data_plot[selected_data_plot['group'] == "Low Risk"]
        kmf_2.fit(group_data_2['time'], event_observed=group_data_2['event'], label="Low Risk")
        kmf_2.plot_survival_function(ax=ax, ci_show=False, color="#2EB8DE", linewidth=2,
                                     show_censors=True, censor_styles=censor_styles)

        cph = CoxPHFitter()
        selected_data_plot["group_id"] = selected_data_plot["group"].map({"High Risk": 1, "Low Risk": 0})
        cph.fit(selected_data_plot, duration_col='time', event_col='event', formula='group_id')

        print(f"HR:{cph.summary['exp(coef)'][0]:.3f}")
        print(f"p-value:{cph.summary['p'][0]:.3f}")
        print(f"HR 95%CI:({cph.summary['exp(coef) lower 95%'][0]:.3f},"
              f"{cph.summary['exp(coef) upper 95%'][0]:.3f})")

        ax.text(0.1, 0.2, f"HR: {cph.summary['exp(coef)'][0]:.3f} "
                          f"({cph.summary['exp(coef) lower 95%'][0]:.3f}, {cph.summary['exp(coef) upper 95%'][0]:.3f}) \n"
                          f"p = {cph.summary['p'][0]:.3e}", transform=ax.transAxes, size=20)

        add_at_risk_counts(kmf_1, kmf_2, rows_to_show=["At risk"])

        ax.set_title(f"{NAME_CODEBOOK[split]} Cohort", fontsize=32)
        ax.set_xlabel('Time (Months)')
        ax.set_ylabel('Survival Probability')
        ax.set_xlim(0)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right', fontsize=18)

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{split}.pdf"), dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    main()
