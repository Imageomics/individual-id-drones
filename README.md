# individual-id-drones

Pose-aware drone navigation for zebra individual identification using [MMLA data](https://huggingface.co/collections/imageomics/mmla) and [WildWing](https://imageomics.github.io/wildwing/).

## Overview

This repository contains pose-guided drone navigation utilities and experiment scripts developed for autonomous zebra individual identification.

The system combines:
- YOLO-based zebra detection
- DINOv2/MMLA pose estimation
- pose-aware navigation logic
- automated batch experiment evaluation

The goal is to autonomously capture high-quality side-view zebra images suitable for individual ID tasks.

This repository accompanies our CV4Animals workshop paper at CVPR.

---

## Repository Contents

### Core Scripts

- `navigation_pose.py`:
  Main pose-aware navigation pipeline.

- `make_summary.py`:
  Generates experiment summaries and navigation statistics.

- `run_all_raw_videos.py`:
  Runs navigation experiments on batches of raw UAV videos.

- `run_all_tests.sh`:
  Helper script for automated experiment execution.

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Models and Dataset

Hugging Face model:
https://huggingface.co/imageomics/mmla-dino-pose

Hugging Face dataset:
https://huggingface.co/datasets/imageomics/mmla-pose

---

## Example Usage

Run pose-aware navigation:

```bash
python navigation_pose.py
```

Run batch experiments:

```bash
python run_all_raw_videos.py
```

Generate experiment summaries:

```bash
python make_summary.py
```

---

## Citation

If you use this repository, please cite both the software and the associated CV4Animals workshop paper.