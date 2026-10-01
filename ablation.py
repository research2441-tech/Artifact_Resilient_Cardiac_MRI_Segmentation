from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from evaluate import run_evaluation
from train import load_config, run_training


def _set_flags(cfg: Dict[str, Any], aug: bool, eff: bool, pre: bool, att: bool, msf: bool) -> Dict[str, Any]:
    x = copy.deepcopy(cfg)
    x["augmentation"]["enabled"] = bool(aug)
    x["model"]["use_efficientformerv2"] = bool(eff)
    x["model"]["pretrained"] = bool(pre and eff)
    x["model"]["use_channel_attention"] = bool(att)
    x["model"]["use_multiscale_fusion"] = bool(msf)
    return x


def ablation_grid(cfg: Dict[str, Any]) -> List[Tuple[str, bool, bool, bool, bool, bool]]:
    rows: List[Tuple[str, bool, bool, bool, bool, bool]] = []
    ac = cfg.get("ablation", {})
    if ac.get("run_cumulative", True):
        rows += [
            ("baseline_transformer_unet", False, False, False, False, False),
            ("plus_motion_augmentation", True, False, False, False, False),
            ("plus_efficientformerv2", True, True, False, False, False),
            ("plus_pretrained_initialization", True, True, True, False, False),
            ("plus_channel_attention", True, True, True, True, False),
            ("full_model", True, True, True, True, True),
        ]
    if ac.get("run_leave_one_out", True):
        rows += [
            ("full_minus_augmentation", False, True, True, True, True),
            ("full_minus_efficientformerv2", True, False, False, True, True),
            ("full_minus_pretrained", True, True, False, True, True),
            ("full_minus_channel_attention", True, True, True, False, True),
            ("full_minus_multiscale_fusion", True, True, True, True, False),
        ]
    seen = set()
    unique = []
    for row in rows:
        if row[0] not in seen:
            unique.append(row)
            seen.add(row[0])
    return unique


def run_ablation(cfg: Dict[str, Any]) -> pd.DataFrame:
    results = []
    split = cfg.get("ablation", {}).get("test_split", "test")
    for name, aug, eff, pre, att, msf in ablation_grid(cfg):
        local = _set_flags(cfg, aug, eff, pre, att, msf)
        ckpt = run_training(local, name)
        summary = run_evaluation(local, ckpt, split=split, prefix=name)
        summary_df = pd.read_csv(f"{name}_summary.csv").set_index("metric")
        def pick(metric: str, field: str):
            return float(summary_df.loc[metric, field])
        results.append({
            "configuration": name,
            "augmentation": aug,
            "efficientformerv2": eff,
            "pretrained": pre and eff,
            "channel_attention": att,
            "multiscale_fusion": msf,
            "dsc_mean": pick("dsc_macro", "mean"),
            "dsc_sd": pick("dsc_macro", "sd"),
            "dsc_ci_low": pick("dsc_macro", "ci_low"),
            "dsc_ci_high": pick("dsc_macro", "ci_high"),
            "iou_mean": pick("iou_macro", "mean"),
            "iou_sd": pick("iou_macro", "sd"),
            "hd95_mean": pick("hd95_macro", "mean"),
            "hd95_sd": pick("hd95_macro", "sd"),
        })
        pd.DataFrame(results).to_csv("results_ablation.csv", index=False)

    df = pd.DataFrame(results)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.bar(range(len(df)), df["dsc_mean"], yerr=df["dsc_sd"], capsize=3)
        ax.set_ylabel("Macro Dice")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["configuration"], rotation=45, ha="right")
        ax.set_ylim(max(0.0, float(df["dsc_mean"].min()) - 0.1), 1.0)
        fig.tight_layout()
        fig.savefig("figure_ablation.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    df = run_ablation(load_config(args.config))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
