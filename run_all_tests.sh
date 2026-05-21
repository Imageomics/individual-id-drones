#!/bin/bash

BASE_DIR="/fs/ess/PAS2136/individual_id_zebras/Claire"
VIDEO_DIR="$BASE_DIR/test_clips"
OUT_DIR="$BASE_DIR/batch_outputs"

YOLO_MODEL="$BASE_DIR/detection_model/mmlaYOLOV11.pt"
POSE_MODEL="$BASE_DIR/pose_model/checkpoints/best_pose_model.pth"

mkdir -p "$OUT_DIR"

for video in "$VIDEO_DIR"/*.mp4; do
    name=$(basename "$video" .mp4)

    echo "Processing $name ..."

    mkdir -p "$OUT_DIR/$name"
    mkdir -p "$OUT_DIR/$name/nav"

    python "$BASE_DIR/data/test_detections.py" \
        --video "$video" \
        --save-video \
        --yolo-model "$YOLO_MODEL" \
        --pose-model "$POSE_MODEL" \
        --out "$OUT_DIR/$name"

    python "$BASE_DIR/wildwing/navigation_pose.py" \
        --video "$video" \
        --out "$OUT_DIR/$name/nav" \
        --every-n 2 \
        --max-frames 2000 \
        --device cpu

done

echo "ALL DONE"
