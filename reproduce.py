from __future__ import annotations

import argparse

from ablation import run_ablation
from benchmark import benchmark
from evaluate import run_evaluation
from sanity_check import run_checks
from train import load_config, run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_checks(cfg)
    ckpt = run_training(cfg, "full_model")
    run_evaluation(cfg, ckpt, split=cfg.get("evaluation", {}).get("split", "test"), prefix="full_model")
    if not args.skip_benchmark:
        benchmark(cfg, ckpt)
    if not args.skip_ablation:
        run_ablation(cfg)
    print("Reproduction workflow completed.")


if __name__ == "__main__":
    main()
