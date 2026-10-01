from __future__ import annotations

import argparse
import time
from typing import Any, Dict

import pandas as pd
import torch
from thop import profile

from models import build_model
from train import load_config, seed_everything


def benchmark(cfg: Dict[str, Any], checkpoint: str | None = None) -> pd.DataFrame:
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device).eval()
    if checkpoint:
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model_state"] if "model_state" in payload else payload)

    size = int(cfg["data"]["image_size"])
    in_ch = int(cfg["model"].get("in_channels", 1))
    x = torch.randn(1, in_ch, size, size, device=device)
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops = 2.0 * macs

    bc = cfg.get("benchmark", {})
    warmup = int(bc.get("warmup_iterations", 30))
    timed = int(bc.get("timed_iterations", 100))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(timed):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

    latency_ms = elapsed * 1000.0 / timed
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else float("nan")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    row = {
        "model": "ArtifactResilientUNet",
        "device": str(device),
        "input_size": f"1x{in_ch}x{size}x{size}",
        "parameters_million": trainable / 1e6,
        "flops_g": flops / 1e9,
        "latency_ms_per_slice": latency_ms,
        "throughput_slices_per_sec": 1000.0 / latency_ms,
        "peak_gpu_memory_gb": peak_gb,
    }
    df = pd.DataFrame([row])
    df.to_csv("results_complexity.csv", index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    print(benchmark(load_config(args.config), args.checkpoint).to_string(index=False))


if __name__ == "__main__":
    main()
