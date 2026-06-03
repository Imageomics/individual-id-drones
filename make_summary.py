#!/usr/bin/env python3
"""
Summarize navigation_pose.py outputs.

Scans nav_outputs/*/capture/meta_*.json and writes:
- summary.csv
- summary.json

Also prints quick stats (capture_reason, pose distribution).

Usage:
  python make_summary.py \
    --nav-outputs /fs/ess/PAS2136/individual_id_zebras/Claire/nav_outputs \
    --out /fs/ess/PAS2136/individual_id_zebras/Claire/nav_outputs/summary.csv

Optional:
  --out-json /path/to/summary.json
"""

import argparse
import csv
import json
from pathlib import Path
from collections import Counter


def safe_get(d, key, default=None):
    return d.get(key, default) if isinstance(d, dict) else default


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nav-outputs", type=str, required=True,
                   help="Path to Claire/nav_outputs (contains run folders).")
    p.add_argument("--out", type=str, required=True,
                   help="Output CSV path, e.g. .../nav_outputs/summary.csv")
    p.add_argument("--out-json", type=str, default=None,
                   help="Optional output JSON path.")
    args = p.parse_args()

    nav_outputs = Path(args.nav_outputs)
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.out_json) if args.out_json else None
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)

    meta_paths = sorted(nav_outputs.glob("*/capture/meta_*.json"))

    rows = []
    reason_ctr = Counter()
    pose_ctr = Counter()

    for mp in meta_paths:
        try:
            data = json.loads(mp.read_text())
        except Exception as e:
            print(f"[WARN] Failed to read {mp}: {e}")
            continue

        run_dir = mp.parents[1]  # .../<run>/capture/meta_*.json -> <run>
        run_name = run_dir.name

        bbox_w = safe_get(data, "bbox_w", "")
        bbox_h = safe_get(data, "bbox_h", "")
        pose = safe_get(data, "pose_smooth", safe_get(data, "pose", ""))
        pose_raw = safe_get(data, "pose_raw", "")
        frame_idx = safe_get(data, "frame_idx", "")
        det_conf = safe_get(data, "det_conf", "")
        det_cls = safe_get(data, "det_cls", "")
        capture_reason = safe_get(data, "capture_reason", "")
        state = safe_get(data, "state", "")
        bbox_xyxy = safe_get(data, "bbox_xyxy", safe_get(data, "bbox_xyxy", ""))

        # pull policy params if present
        policy_params = safe_get(data, "policy_params", {})
        bbox_ratio = safe_get(policy_params, "BBOX_RATIO", "")
        bbox_thresh = safe_get(policy_params, "BBOX_THRESH", "")
        approach_max = safe_get(policy_params, "APPROACH_MAX_STEPS", "")
        smooth_k = safe_get(policy_params, "SMOOTH_K", "")
        side_window = safe_get(policy_params, "SIDE_WINDOW", "")
        side_need = safe_get(policy_params, "SIDE_NEED", "")
        conf_thres = safe_get(policy_params, "CONF_THRES", "")

        # derive "video stem" from run folder name: DJI_0018_30_YYYYMMDD_HHMMSS -> DJI_0018_30
        # split from the right by "_" twice to strip timestamp
        video_stem = run_name
        parts = run_name.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            video_stem = parts[0]

        row = {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "video_stem": video_stem,
            "meta_path": str(mp),

            "frame_idx": frame_idx,
            "state": state,
            "pose_smooth": pose,
            "pose_raw": pose_raw,
            "capture_reason": capture_reason,

            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "bbox_xyxy": bbox_xyxy,
            "det_conf": det_conf,
            "det_cls": det_cls,

            "BBOX_RATIO": bbox_ratio,
            "BBOX_THRESH": bbox_thresh,
            "APPROACH_MAX_STEPS": approach_max,
            "SMOOTH_K": smooth_k,
            "SIDE_WINDOW": side_window,
            "SIDE_NEED": side_need,
            "CONF_THRES": conf_thres,
        }

        rows.append(row)

        if capture_reason:
            reason_ctr[capture_reason] += 1
        if pose:
            pose_ctr[pose] += 1

    # write CSV
    fieldnames = [
        "run_dir", "run_name", "video_stem", "meta_path",
        "frame_idx", "state", "pose_smooth", "pose_raw", "capture_reason",
        "bbox_w", "bbox_h", "bbox_xyxy", "det_conf", "det_cls",
        "BBOX_RATIO", "BBOX_THRESH", "APPROACH_MAX_STEPS",
        "SMOOTH_K", "SIDE_WINDOW", "SIDE_NEED", "CONF_THRES",
    ]

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # write JSON
    if out_json:
        out_json.write_text(json.dumps(rows, indent=2))

    # print stats
    print(f"Found {len(rows)} capture meta files under: {nav_outputs}")
    print(f"Wrote CSV: {out_csv}")
    if out_json:
        print(f"Wrote JSON: {out_json}")

    if len(rows) == 0:
        return

    print("\nCapture reason distribution:")
    for k, v in reason_ctr.most_common():
        print(f"  {k:20s} {v}")

    print("\nPose distribution (pose_smooth):")
    for k, v in pose_ctr.most_common():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
