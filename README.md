# Artifact-Resilient Cardiac MRI Segmentation

This repository provides a flat, reproducible implementation of the cardiac MR segmentation pipeline described in the manuscript **Artifact-Resilient Deep Learning Segmentation for Enhanced Structural and Functional Cardiac Analysis**. The implementation is designed specifically to support controlled reviewer-requested experiments for respiratory-motion-aware augmentation, EfficientFormerV2 encoding, pretrained initialization, channel-attention skip connections, and multi-scale fusion.

## Scope

The pipeline performs multi-class short-axis cardiac MR segmentation for:

- background
- left ventricular cavity (LV)
- right ventricular cavity (RV)
- myocardium (MYO)

The code uses segmentation-specific evaluation only: Dice similarity coefficient (DSC), intersection-over-union (IoU), and 95th-percentile Hausdorff distance (HD95). Precision and recall are available as secondary diagnostics. OA, AA, Cohen's Kappa, and generic image-level test accuracy are intentionally not used as primary manuscript metrics.

## Repository files

| File | Purpose |
|---|---|
| `config.yaml` | Single source of truth for data, model, optimization, ablation, and evaluation settings. |
| `data.py` | NIfTI/NumPy loading, canonical orientation, optional spacing resampling, z-score normalization, label mapping, slice generation, and subject-safe data loading. |
| `augmentations.py` | Respiratory-motion-oriented training augmentation. |
| `models.py` | Baseline Transformer U-Net and the proposed EfficientFormerV2 + channel-attention + multi-scale-fusion model. |
| `losses.py` | Dice + cross-entropy segmentation loss. |
| `metrics.py` | Per-class and macro DSC, IoU, HD95, precision, recall, and bootstrap confidence intervals. |
| `train.py` | Deterministic mixed-precision training, early stopping, checkpointing, and CSV logging. |
| `evaluate.py` | Subject-level held-out evaluation and qualitative figure generation. |
| `ablation.py` | Quantitative component-wise and leave-one-component-out ablation experiments. |
| `benchmark.py` | Parameters, FLOPs, inference latency, and peak GPU-memory benchmarking. |
| `reproduce.py` | One-command orchestration of sanity checks, full-model training/evaluation, ablation, and benchmark. |
| `sanity_check.py` | Dataset, split, mask-label, configuration, and leakage validation. |
| `requirements.txt` | Reproducible Python dependencies. |

No dataset, proprietary weight file, DOI, or license text is bundled here.

## Data manifest

The implementation does not assume an unpublished directory layout. Instead, create a CSV manifest named `manifest.csv` with one row per 3-D image volume:

```csv
image,mask,subject_id,split,condition,phase
/path/to/sub001_ED_image.nii.gz,/path/to/sub001_ED_mask.nii.gz,sub001,train,full_breath_hold,ED
/path/to/sub001_ES_image.nii.gz,/path/to/sub001_ES_mask.nii.gz,sub001,train,full_breath_hold,ES
/path/to/sub021_ED_image.nii.gz,/path/to/sub021_ED_mask.nii.gz,sub021,val,free_breathing,ED
/path/to/sub026_ED_image.nii.gz,/path/to/sub026_ED_mask.nii.gz,sub026,test,intensive_breathing,ED
```

Required columns are `image`, `mask`, `subject_id`, and `split`. `condition` and `phase` are retained in result files when supplied. Splits must be subject-disjoint.

The manuscript describes 20 volunteers/160 volumes for training, 5 volunteers/40 volumes for validation, and 15 volunteers/120 volumes for held-out testing. If the local challenge release uses a different officially supplied partition, update only `manifest.csv`; do not mix slices from the same subject across splits.

## Mask labels

The default configuration expects:

```text
0 = background
1 = LV
2 = RV
3 = myocardium
```

If the challenge release uses different integer values, update `data.label_map` in `config.yaml`. `sanity_check.py` reports all observed labels before training.

## Installation

Python 3.10 is recommended.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

EfficientFormerV2 pretrained initialization is provided through `timm`. On the first pretrained run, the corresponding public ImageNet weight file may need to be downloaded by `timm`; in offline environments, cache the weight beforehand or set `model.pretrained: false` for experiments that intentionally remove pretrained initialization.

## Configuration

Edit only `config.yaml` for routine reproduction. Important defaults are:

- input size: 224 x 224
- batch size: 16
- optimizer: SGD
- learning rate: 0.1
- momentum: 0.9
- weight decay: 1e-4
- mixed precision: enabled
- deterministic seed: 42
- primary metrics: DSC, IoU, HD95

The configuration intentionally resolves the manuscript inconsistency in which batch size was reported as both 32 and 16 by exposing one explicit reproducibility setting. The value should be changed only to match the experiment actually used for the manuscript.

## Pre-flight validation

Run this before any expensive experiment:

```bash
python sanity_check.py --config config.yaml
```

The checker verifies:

- missing files
- image/mask shape mismatch
- subject leakage between train/validation/test
- unknown split names
- unexpected mask labels
- NaN/Inf intensities
- class-count mismatch
- missing target structures

## Train the complete proposed model

```bash
python train.py --config config.yaml --run-name full_model
```

The best checkpoint is written as:

```text
full_model_best.pt
```

and training history as:

```text
full_model_history.csv
```

## Held-out evaluation

```bash
python evaluate.py --config config.yaml --checkpoint full_model_best.pt --split test --prefix full_model
```

Key outputs are:

```text
full_model_subject_metrics.csv
full_model_summary.csv
full_model_qualitative.png
```

Metrics are aggregated at subject level, not by treating every 2-D slice as an independent patient. This avoids artificially increasing the effective sample size.

## Quantitative ablation required by the reviewers

Run:

```bash
python ablation.py --config config.yaml
```

The script executes controlled configurations using identical data partitions, optimizer settings, stopping criteria, and evaluation code.

It includes:

1. baseline Transformer U-Net
2. baseline + motion-aware augmentation
3. + EfficientFormerV2
4. + pretrained initialization
5. + channel-attention skip connections
6. full model with multi-scale fusion
7. full model minus augmentation
8. full model minus EfficientFormerV2
9. full model minus pretrained initialization
10. full model minus channel attention
11. full model minus multi-scale fusion

The primary output is:

```text
results_ablation.csv
```

with fields such as:

```text
configuration,augmentation,efficientformerv2,pretrained,channel_attention,multiscale_fusion,dsc_mean,dsc_sd,dsc_ci_low,dsc_ci_high,iou_mean,iou_sd,hd95_mean,hd95_sd
```

The script never fabricates missing performance values. Each row is produced from an actual trained checkpoint and held-out evaluation.

## Computational comparison

Run:

```bash
python benchmark.py --config config.yaml --checkpoint full_model_best.pt
```

This produces:

```text
results_complexity.csv
```

including trainable parameters, FLOPs, mean inference latency, throughput, and peak GPU memory measured under a common input shape.

## Complete reproducibility command

After the manifest and configuration have been verified:

```bash
python reproduce.py --config config.yaml
```

Use `--skip-ablation` when only validating the complete model, because the full ablation grid requires multiple complete training runs.

## Reproducibility and fair comparison rules

For manuscript tables, all compared in-repository configurations should use:

- the same subject-level partition
- the same image preprocessing
- the same target resolution
- the same loss
- the same optimizer and learning-rate schedule
- the same early-stopping rule
- the same evaluation code
- the same held-out test subjects

Do not report numbers from unrelated datasets such as MoNuSeg or Synapse as if they were directly comparable CMRxMotion cardiac-MR results. Literature values from external 2024-2026 cardiac MR studies should be clearly identified as published values unless those methods are re-trained under the same local protocol.

## Figure quality

`evaluate.py` and `ablation.py` save figures at 300 DPI with readable labels. For final production, vector export can additionally be enabled by changing the output extension to PDF/SVG where permitted by the journal.

## Expected reviewer-facing evidence

A complete run of this repository supplies the evidence required to support:

- the independent effect of respiratory-motion-aware augmentation
- the effect of replacing the baseline encoder with EfficientFormerV2
- the effect of pretrained initialization
- the effect of channel-attention skip refinement
- the effect of multi-scale dilated fusion
- the combined performance of the complete model
- segmentation-specific evaluation using DSC, IoU, and HD95
- computational cost under one common benchmark protocol

Only experimentally generated values should be transferred into the revised manuscript and response letter.
