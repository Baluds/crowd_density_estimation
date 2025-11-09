## Crowd Density Estimation (Guided Diffusion + Super-resolution)

This repository contains code and data for crowd density estimation using guided diffusion and super-resolution approaches. The project provides data preprocessing utilities, training/test scripts, a Colab notebook, and helper modules for evaluation and visualization.

## Contents

- `scripts/` — training and testing scripts (e.g. `super_res_train.py`, `super_res_test.py`, `test_single_image.py`, and others).
- `guided_diffusion/` and `guided_diffusion_new/` — diffusion model code (model, diffusion scheduler, losses, training utilities).
- `cc_utils/` — evaluation, preprocessing, and utility scripts for crowd counting and visualization.
- `data_full/` — expected dataset layout (contains `New_Test`, `part_A_final`, `part_B_final`, etc.).
- `vidToImg/` — video-to-frames and event-based sampling utilities.
- `Crowd_Density_v1.ipynb` — Colab notebook with example preprocessing and quick-start commands.
- `requirements.txt`, `req_new.txt` — Python dependencies (use one of these to install required packages).

## Quick overview

The repo is designed to: prepare image / density-map datasets, train a diffusion-based super-resolution model that predicts density maps, and run inference to generate and visualize maps.

Key scripts:
- `scripts/super_res_train.py` — train a super-resolution diffusion model.
- `scripts/super_res_test.py` — generate density maps with a trained model.
- `scripts/test_single_image.py` — run inference on a single image.
- `cc_utils/evaluate.py` — evaluation utilities for comparing predicted maps against ground truth.

## Installation

1. Clone the repo:

```bash
git clone https://github.com/Baluds/crowd_density_estimation.git
cd crowd_density_estimation
```

2. Create and activate a Python virtual environment (recommended) and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# or, if you use the notebook's requirements: pip install -r req_new.txt
```

Notes:
- GPU + CUDA and a compatible PyTorch build are recommended for training and fast inference.
- If you run in Google Colab, the included `Crowd_Density_v1.ipynb` demonstrates installing extra packages and running quick experiments.

## Dataset layout

The repository expects a dataset organized under `data_full/` (example structure present in the workspace):

- `data_full/New_Test/` — test images and corresponding density maps in `test_den` or `test_den/*.csv`.
- `data_full/part_A_final/` and `data_full/part_B_final/` — training and test splits for established crowd datasets. Each split contains `train_data` and `test_data`, with `images/` and `ground_truth/` directories.

Each ground-truth density map is stored in .mat or .csv files (see `data_full/*/test_data/ground_truth/` for examples: `GT_IMG_1.mat`, etc.). When using `super_res` scripts in this repo, some utilities expect `.csv` density maps (see `scripts` and `cc_utils` for conversion helpers).

## Preprocessing

- Use `vidToImg/` utilities to convert video to frames if your input comes from video.
- The Colab notebook `Crowd_Density_v1.ipynb` includes example preprocessing (edge-detection, resizing to 256×256, preparing `test_den` CSVs) used in experiments.

## Training (example)

Training uses the super-resolution training script and many configurable flags. A minimal example:

```bash
python scripts/super_res_train.py \
  --data_dir /path/to/data_dir \
  --log_dir /path/to/logs \
  --batch_size 4 \
  --large_size 256 --small_size 256 \
  --pred_channels 1 \
  --normalizer 0.8 \
  --use_fp16 True \
  --resume_checkpoint "" \
  --lr 1e-4
```

The model and diffusion defaults are defined in `guided_diffusion.script_util`. The project includes many model-related flags (attention resolutions, channels, steps). See `scripts/super_res_train.py` and `guided_diffusion/script_util.py` for all options.

## Testing / Inference (example)

Use `scripts/super_res_test.py` to run generation/inference with a checkpoint. Example taken from Colab usage in the notebook:

```bash
python scripts/super_res_test.py \
  --data_dir /path/to/input_images \
  --log_dir /path/to/output_dir --model_path /path/to/demo.pt \
  --normalizer 0.8 --pred_channels 1 --batch_size 1 \
  --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 \
  --large_size 256 --small_size 256 --learn_sigma True --noise_schedule linear \
  --num_channels 192 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True \
  --use_fp16 True --use_scale_shift_norm True
```

Adjust flags depending on the checkpoint and expected input channels (e.g., `pred_channels=1` when density maps are single-channel).

## Running a single image

```bash
python scripts/test_single_image.py --input /path/to/image.jpg --checkpoint /path/to/model.pt --out /path/to/out.png
```

Check the specific script's `--help` for all available flags.

## Evaluation & Visualization

- `cc_utils/evaluate.py` — helpers to compute errors/metrics vs ground truth.
- `cc_utils/vis_test.py` — visualization utilities used to inspect model outputs.

## Notebooks and Colab

- `Crowd_Density_v1.ipynb` contains Colab-ready examples for preprocessing, installing dependencies, and running test/validation snippets. Use that notebook for fast experimentation on Colab GPUs.

## Notes & tips

- Many scripts assume density maps normalized to [0,1] or [-1,1] depending on code paths — the `normalizer` flag controls scaling used when preparing batches. Inspect `scripts/super_res_train.py` to confirm expected normalizer values.
- If you see imports like `from guided_diffusion import ...`, ensure your Python path includes the repo root (running scripts from repo root will usually work).

