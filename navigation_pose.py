"""
Pose-based navigation logic (offline simulation on pre-recorded video or single image)

Goal:
- Detect zebra (YOLO)
- Classify pose from bbox crop (pose classifier)
- ORIENT: rotate yaw until side-view (left/right) stable
- APPROACH: move forward until bbox is "sufficiently large"
- CAPTURE: save frame + crop + meta + stop

This file does NOT fly a real drone. It outputs recommended actions.

Key update vs previous:
- Capture is now robust:
  (1) primary: ratio-based threshold vs frame size (scale-invariant)
  (2) fallback: timeout best-effort capture (offline videos can't actually move closer)
"""

import sys
import csv
import json
import datetime
from pathlib import Path
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ----------------------------
# Paths (use absolute paths to avoid OSC relative-path pain)
# ----------------------------
YOLO_WTS = "/fs/ess/PAS2136/individual_id_zebras/Claire/detection_model/mmlaYOLOV11.pt"
POSE_WTS = "/fs/ess/PAS2136/individual_id_zebras/Claire/pose_model/checkpoints/best_pose_model.pth"


def _find_repo_root_containing_pose_model(start: Path, max_up: int = 8) -> Path:
    """
    Robustly find the Claire/ repo root by walking upward until we see pose_model/.
    This avoids fragile parents[k] assumptions.
    """
    cur = start
    for _ in range(max_up):
        if (cur / "pose_model").exists():
            return cur
        cur = cur.parent
    raise RuntimeError(f"Cannot find repo root containing 'pose_model/' starting from {start}")


# We are in Claire/wildwing/navigation_pose.py
HERE = Path(__file__).resolve()
REPO_ROOT = _find_repo_root_containing_pose_model(HERE.parent)
sys.path.insert(0, str(REPO_ROOT))

from pose_model.pose_classifier import ViewPointClassifier  # noqa: E402


# ----------------------------
# Tunables
# ----------------------------
# Original spec (may be unreachable in some offline clips)
BBOX_THRESH = 500          # px, w>=500 and h>=500

# Robust capture criteria for offline videos
BBOX_RATIO = 0.1          # capture if min(bbox_w,bbox_h) >= BBOX_RATIO * min(frame_w, frame_h)
APPROACH_MAX_STEPS = 300   # fallback: after this many APPROACH steps, capture best-effort

SMOOTH_K = 5               # pose smoothing window (majority vote over last K frames)
SIDE_WINDOW = 7            # side stability window
SIDE_NEED = 5              # in SIDE_WINDOW frames, need >= SIDE_NEED frames to be side-view

CONF_THRES = 0.25          # YOLO confidence filter
TARGET_CLASSES = None      # if you know zebra class id, set like {17}; else None = accept all


# Movement outputs (arbitrary units; for offline logging)
YAW_RATE = 0.15            # positive = rotate right, negative = rotate left
FORWARD_SPEED = 0.10       # forward step


def majority_vote(q):
    if not q:
        return None
    return Counter(q).most_common(1)[0][0]


def action_from_pose(pose: str, scan_dir: int):
    """
    Map pose -> yaw direction

    Table (align with your assignment):
      - front/back -> rotate in yaw (either direction OK; we alternate via scan_dir)
      - front-left/back-left -> rotate right
      - front-right/back-right -> rotate left
      - left/right -> good (side view)

    Returns (action_name, yaw_rate, next_scan_dir)
    """
    if pose in ["front", "back"]:
        # alternate direction to avoid getting stuck spinning wrong way forever
        next_scan_dir = -scan_dir
        return ("ROTATE_YAW", next_scan_dir * YAW_RATE, next_scan_dir)
    if pose in ["front-left", "back-left"]:
        return ("ROTATE_RIGHT", +YAW_RATE, scan_dir)
    if pose in ["front-right", "back-right"]:
        return ("ROTATE_LEFT", -YAW_RATE, scan_dir)
    if pose in ["left", "right"]:
        return ("SIDE_OK", 0.0, scan_dir)
    # unknown pose -> keep scanning
    next_scan_dir = -scan_dir
    return ("ROTATE_YAW", next_scan_dir * YAW_RATE, next_scan_dir)


def pick_target_box(results, prev_xyxy=None, iou_lock_thresh=0.3):
    """
    Pick a target bbox.

    Strategy:
      - If prev_xyxy exists: pick candidate with max IoU to prev (if IoU >= thresh).
      - Else: pick largest bbox area.

    Returns (x1,y1,x2,y2,conf,cls) or None.
    """
    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None

    xyxy = r0.boxes.xyxy.cpu().numpy()  # (N,4)
    conf = r0.boxes.conf.cpu().numpy()  # (N,)
    cls  = r0.boxes.cls.cpu().numpy()   # (N,)

    candidates = []
    for i in range(len(xyxy)):
        if conf[i] < CONF_THRES:
            continue
        if TARGET_CLASSES is not None and int(cls[i]) not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = xyxy[i]
        area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        candidates.append((area, float(x1), float(y1), float(x2), float(y2), float(conf[i]), int(cls[i])))

    if not candidates:
        return None

    # IoU lock if prev exists
    if prev_xyxy is not None:
        px1, py1, px2, py2 = prev_xyxy

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            return 0.0 if union <= 0 else inter / union

        best = None
        best_iou = -1.0
        for area, x1, y1, x2, y2, c, k in candidates:
            v = iou((x1, y1, x2, y2), (px1, py1, px2, py2))
            if v > best_iou:
                best_iou = v
                best = (x1, y1, x2, y2, c, k)

        if best is not None and best_iou >= iou_lock_thresh:
            return best

    # fallback: largest area
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x1, y1, x2, y2, c, k = candidates[0]
    return (x1, y1, x2, y2, c, k)


def crop_for_pose(frame_bgr, box_xyxy):
    x1, y1, x2, y2, _, _ = box_xyxy
    h, w = frame_bgr.shape[:2]
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w,     x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h,     y2)))
    if x2 <= x1 or y2 <= y1:
        return None, None
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    crop_rgb = crop_bgr[:, :, ::-1]  # BGR->RGB
    return Image.fromarray(crop_rgb), (x1, y1, x2, y2)


class PoseNavPolicy:
    def __init__(self, device="cpu"):
        self.pose_clf = ViewPointClassifier(weight_path=POSE_WTS, device=device)

        self.state = "ORIENT"
        self.pose_hist = deque(maxlen=SMOOTH_K)

        # side stability via windowed count (more robust than strict consecutive frames)
        self.side_hist = deque(maxlen=SIDE_WINDOW)

        # scan direction for front/back (alternate so we don't spin forever)
        self.scan_dir = +1

        # for IoU target locking
        self.prev_xyxy = None

        # store last target for logging
        self.last_target = None

        # APPROACH tracking for best-effort capture
        self.approach_steps = 0
        self.best_min_side = 0
        self.best_frame_idx = None
        self.best_bbox_xyxy = None
        self.capture_reason = None

    def reset(self):
        self.state = "ORIENT"
        self.pose_hist.clear()
        self.side_hist.clear()
        self.prev_xyxy = None
        self.last_target = None

        self.approach_steps = 0
        self.best_min_side = 0
        self.best_frame_idx = None
        self.best_bbox_xyxy = None
        self.capture_reason = None

    def step(self, frame_bgr, results, frame_idx=None):
        """
        Returns dict with:
          - state, pose_raw, pose (smoothed)
          - bbox_w, bbox_h, bbox_xyxy, det_conf, det_cls
          - min_side, frame_min
          - action: forward, yaw, capture_bool
          - capture_reason (if capture=True)
        """
        target = pick_target_box(results, prev_xyxy=self.prev_xyxy)
        self.last_target = target

        if target is None:
            self.reset()
            # no detections -> keep scanning
            return {
                "state": self.state,
                "pose_raw": None,
                "pose": None,
                "bbox_w": 0,
                "bbox_h": 0,
                "bbox_xyxy": None,
                "det_conf": None,
                "det_cls": None,
                "min_side": 0,
                "frame_min": min(frame_bgr.shape[1], frame_bgr.shape[0]),
                "forward": 0.0,
                "yaw": +YAW_RATE,
                "capture": False,
                "capture_reason": None,
            }

        x1, y1, x2, y2, det_conf, det_cls = target
        bbox_w = int(x2 - x1)
        bbox_h = int(y2 - y1)

        pil_crop, clipped_xyxy = crop_for_pose(frame_bgr, target)
        if pil_crop is None:
            self.reset()
            return {
                "state": self.state,
                "pose_raw": None,
                "pose": None,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "bbox_xyxy": None,
                "det_conf": det_conf,
                "det_cls": det_cls,
                "min_side": min(bbox_w, bbox_h),
                "frame_min": min(frame_bgr.shape[1], frame_bgr.shape[0]),
                "forward": 0.0,
                "yaw": +YAW_RATE,
                "capture": False,
                "capture_reason": None,
            }

        # remember clipped bbox for next-frame IoU locking
        cx1, cy1, cx2, cy2 = clipped_xyxy
        self.prev_xyxy = (float(cx1), float(cy1), float(cx2), float(cy2))

        # pose inference
        pose_raw = self.pose_clf([pil_crop])[0]
        self.pose_hist.append(pose_raw)
        pose = majority_vote(list(self.pose_hist))

        # update side stability window
        self.side_hist.append(pose_raw in ["left", "right"])

        # computed scale stats
        H, W = frame_bgr.shape[:2]
        min_side = min(bbox_w, bbox_h)
        frame_min = min(W, H)

        # --- state machine ---
        forward = 0.0
        yaw = 0.0
        capture = False
        capture_reason = None

        if self.state == "ORIENT":
            # reset approach bookkeeping anytime we're not approaching
            self.approach_steps = 0
            self.best_min_side = 0
            self.best_frame_idx = None
            self.best_bbox_xyxy = None

            if sum(self.side_hist) >= SIDE_NEED and pose in ["left", "right"]:
                self.state = "APPROACH"
                forward = 0.0
                yaw = 0.0
            else:
                _, yaw, self.scan_dir = action_from_pose(pose, self.scan_dir)
                forward = 0.0

        elif self.state == "APPROACH":
            if pose not in ["left", "right"]:
                # lost side view, go back to orient
                self.state = "ORIENT"
                self.side_hist.clear()
                self.approach_steps = 0
                _, yaw, self.scan_dir = action_from_pose(pose, self.scan_dir)
                forward = 0.0
            else:
                yaw = 0.0
                forward = FORWARD_SPEED

                # track best so far (for best-effort capture)
                if min_side > self.best_min_side:
                    self.best_min_side = min_side
                    self.best_frame_idx = frame_idx
                    self.best_bbox_xyxy = (cx1, cy1, cx2, cy2)

                self.approach_steps += 1

                # (1) primary: ratio-based capture
                if min_side >= BBOX_RATIO * frame_min:
                    self.state = "CAPTURE"
                    capture_reason = "ratio_threshold"
                # (2) optional: strict absolute threshold still supported
                elif bbox_w >= BBOX_THRESH and bbox_h >= BBOX_THRESH:
                    self.state = "CAPTURE"
                    capture_reason = "abs_500x500"
                # (3) fallback: timeout best-effort
                elif self.approach_steps >= APPROACH_MAX_STEPS:
                    self.state = "CAPTURE"
                    capture_reason = "timeout_best_effort"

        if self.state == "CAPTURE":
            forward = 0.0
            yaw = 0.0
            capture = True
            # if we arrived here without setting a reason (edge case), label it
            if capture_reason is None:
                capture_reason = self.capture_reason or "capture"

            # persist reason so caller can log it even if capture happens for multiple calls
            self.capture_reason = capture_reason

        return {
            "state": self.state,
            "pose_raw": pose_raw,
            "pose": pose,
            "bbox_w": bbox_w,
            "bbox_h": bbox_h,
            "bbox_xyxy": (cx1, cy1, cx2, cy2),
            "det_conf": det_conf,
            "det_cls": det_cls,
            "min_side": min_side,
            "frame_min": frame_min,
            "forward": forward,
            "yaw": yaw,
            "capture": capture,
            "capture_reason": self.capture_reason if capture else None,
        }


def _make_run_dir(out_dir: Path, video_or_image: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = Path(video_or_image).stem
    run_dir = out_dir / f"{name}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "debug_frames").mkdir(exist_ok=True)
    (run_dir / "capture").mkdir(exist_ok=True)
    return run_dir


def _write_capture_assets(run_dir: Path, frame_bgr, crop_pil: Image.Image, info: dict):
    cap_dir = run_dir / "capture"
    frame_path = cap_dir / f"frame_{info['frame_idx']:06d}.jpg"
    crop_path = cap_dir / f"crop_{info['frame_idx']:06d}.jpg"
    meta_path = cap_dir / f"meta_{info['frame_idx']:06d}.json"

    cv2.imwrite(str(frame_path), frame_bgr)

    crop_rgb = np.array(crop_pil)
    crop_bgr = crop_rgb[:, :, ::-1]
    cv2.imwrite(str(crop_path), crop_bgr)

    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✅ CAPTURE saved: {frame_path}")
    print(f"✅ CAPTURE crop : {crop_path}")
    print(f"✅ CAPTURE meta : {meta_path}")


def run_on_video(video_path, out_dir, max_frames=None, every_n=1, device="cpu"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_dir, video_path)

    model = YOLO(YOLO_WTS)
    policy = PoseNavPolicy(device=device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    txt_path = run_dir / "nav_log.txt"
    csv_path = run_dir / "events.csv"

    frame_idx = 0
    max_w, max_h, max_min_side, max_frame = 0, 0, 0, 0
    saved_capture = False

    with open(txt_path, "w") as ftxt, open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "frame_idx", "state", "pose_raw", "pose_smooth",
                "bbox_w", "bbox_h", "min_side", "frame_min",
                "x1", "y1", "x2", "y2",
                "det_conf", "det_cls",
                "forward", "yaw", "capture", "capture_reason"
            ],
        )
        writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break
            if frame_idx % every_n != 0:
                continue

            results = model(frame, verbose=False)
            info = policy.step(frame, results, frame_idx=frame_idx)

            # safe max bbox stats
            m = 0
            if info["bbox_w"] > 0 and info["bbox_h"] > 0:
                m = min(info["bbox_w"], info["bbox_h"])
                if m > max_min_side:
                    max_min_side = m
                    max_w = info["bbox_w"]
                    max_h = info["bbox_h"]
                    max_frame = frame_idx

            line = (
                f"frame={frame_idx} state={info['state']} pose={info['pose']} "
                f"bbox=({info['bbox_w']}x{info['bbox_h']}) min_side={info['min_side']} "
                f"action=(forward={info['forward']:.2f}, yaw={info['yaw']:.2f}, capture={info['capture']})"
            )
            if info["capture"]:
                line += f" reason={info['capture_reason']}"
            print(line)
            ftxt.write(line + "\n")

            xyxy = info["bbox_xyxy"]
            row = {
                "frame_idx": frame_idx,
                "state": info["state"],
                "pose_raw": info["pose_raw"],
                "pose_smooth": info["pose"],
                "bbox_w": info["bbox_w"],
                "bbox_h": info["bbox_h"],
                "min_side": info["min_side"],
                "frame_min": info["frame_min"],
                "x1": xyxy[0] if xyxy else "",
                "y1": xyxy[1] if xyxy else "",
                "x2": xyxy[2] if xyxy else "",
                "y2": xyxy[3] if xyxy else "",
                "det_conf": info["det_conf"] if info["det_conf"] is not None else "",
                "det_cls": info["det_cls"] if info["det_cls"] is not None else "",
                "forward": info["forward"],
                "yaw": info["yaw"],
                "capture": int(bool(info["capture"])),
                "capture_reason": info["capture_reason"] if info["capture"] else "",
            }
            writer.writerow(row)

            # save debug annotated frame occasionally (every ~1 second of video if 30fps)
            if frame_idx % (30 * every_n) == 0:
                dbg = frame.copy()
                target = policy.last_target
                if target is not None:
                    x1, y1, x2, y2, _, _ = target
                    cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(
                        dbg, f"{info['state']} {info['pose']}",
                        (int(x1), max(30, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                    )
                cv2.imwrite(str(run_dir / "debug_frames" / f"dbg_{frame_idx:06d}.jpg"), dbg)

            # capture condition: save once and stop
            if info["capture"] and not saved_capture:
                saved_capture = True

                # save crop + meta from current frame's bbox
                if policy.last_target is not None:
                    pil_crop, _ = crop_for_pose(frame, policy.last_target)
                else:
                    pil_crop = None

                meta = {
                    "frame_idx": frame_idx,
                    "state": info["state"],
                    "pose_raw": info["pose_raw"],
                    "pose_smooth": info["pose"],
                    "bbox_w": info["bbox_w"],
                    "bbox_h": info["bbox_h"],
                    "min_side": info["min_side"],
                    "frame_min": info["frame_min"],
                    "bbox_xyxy": info["bbox_xyxy"],
                    "det_conf": info["det_conf"],
                    "det_cls": info["det_cls"],
                    "forward": info["forward"],
                    "yaw": info["yaw"],
                    "capture": True,
                    "capture_reason": info["capture_reason"],
                    "policy_params": {
                        "BBOX_THRESH": BBOX_THRESH,
                        "BBOX_RATIO": BBOX_RATIO,
                        "APPROACH_MAX_STEPS": APPROACH_MAX_STEPS,
                        "SMOOTH_K": SMOOTH_K,
                        "SIDE_WINDOW": SIDE_WINDOW,
                        "SIDE_NEED": SIDE_NEED,
                        "CONF_THRES": CONF_THRES,
                    },
                }

                if pil_crop is not None:
                    _write_capture_assets(run_dir, frame, pil_crop, meta)
                else:
                    cap_dir = run_dir / "capture"
                    frame_path = cap_dir / f"frame_{frame_idx:06d}.jpg"
                    meta_path = cap_dir / f"meta_{frame_idx:06d}.json"
                    cv2.imwrite(str(frame_path), frame)
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                    print(f"✅ CAPTURE saved: {frame_path}")
                    print(f"✅ CAPTURE meta : {meta_path}")

                break

    cap.release()
    print(f"Max bbox in this clip: {max_w}x{max_h} (min_side={max_min_side}) at frame {max_frame}")
    print(f"Logs saved:\n- {txt_path}\n- {csv_path}")
    print(f"Run dir: {run_dir}")


def run_on_image(image_path, out_dir, device="cpu"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(out_dir, image_path)

    model = YOLO(YOLO_WTS)
    policy = PoseNavPolicy(device=device)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    results = model(frame, verbose=False)
    info = policy.step(frame, results, frame_idx=0)

    dbg = frame.copy()
    target = policy.last_target
    if target is not None:
        x1, y1, x2, y2, _, _ = target
        cv2.rectangle(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(
            dbg, f"{info['state']} {info['pose']}",
            (int(x1), max(30, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

    out_img = run_dir / "image_debug.jpg"
    cv2.imwrite(str(out_img), dbg)

    if target is not None:
        pil_crop, _ = crop_for_pose(frame, target)
    else:
        pil_crop = None

    meta = {
        "frame_idx": 0,
        "state": info["state"],
        "pose_raw": info["pose_raw"],
        "pose_smooth": info["pose"],
        "bbox_w": info["bbox_w"],
        "bbox_h": info["bbox_h"],
        "min_side": info["min_side"],
        "frame_min": info["frame_min"],
        "bbox_xyxy": info["bbox_xyxy"],
        "det_conf": info["det_conf"],
        "det_cls": info["det_cls"],
        "forward": info["forward"],
        "yaw": info["yaw"],
        "capture": bool(info["capture"]),
        "capture_reason": info["capture_reason"],
        "policy_params": {
            "BBOX_THRESH": BBOX_THRESH,
            "BBOX_RATIO": BBOX_RATIO,
            "APPROACH_MAX_STEPS": APPROACH_MAX_STEPS,
            "SMOOTH_K": SMOOTH_K,
            "SIDE_WINDOW": SIDE_WINDOW,
            "SIDE_NEED": SIDE_NEED,
            "CONF_THRES": CONF_THRES,
        },
    }
    with open(run_dir / "image_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if info["capture"] and pil_crop is not None:
        _write_capture_assets(run_dir, frame, pil_crop, meta)

    print("Result:", info)
    print(f"Saved:\n- {out_img}\n- {run_dir/'image_meta.json'}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    """
    Usage examples:

    # video (clip is fastest)
    python navigation_pose.py \
      --video /fs/ess/PAS2136/individual_id_zebras/Claire/test_clips/DJI_0018_30.mp4 \
      --out /fs/ess/PAS2136/individual_id_zebras/Claire/nav_outputs \
      --every-n 2 --max-frames 3000 --device cpu

    # image
    python navigation_pose.py \
      --image /path/to/frame.jpg \
      --out /fs/ess/PAS2136/individual_id_zebras/Claire/nav_outputs
    """
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--every-n", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()

    if args.video is None and args.image is None:
        raise SystemExit("Provide --video or --image")

    if args.video is not None:
        run_on_video(
            args.video,
            args.out,
            max_frames=args.max_frames,
            every_n=max(1, args.every_n),
            device=args.device,
        )
    else:
        run_on_image(
            args.image,
            args.out,
            device=args.device,
        )
