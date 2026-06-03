import subprocess
from pathlib import Path

videos = [
    "DJI_0006.MP4",
    "DJI_0007.MP4",
    "DJI_0018.MP4",
    "DJI_0019.MP4",
    "DJI_0070.MP4",
    "DJI_0142.MP4",
    "DJI_0207.MP4",
]

raw_dir = Path("data/raw_videos")
out_root = Path("all_raw_runs")
out_root.mkdir(exist_ok=True)

for video in videos:
    video_path = raw_dir / video

    print(f"\n=== Running {video_path.name} ===")

    cmd = [
        "python",
        "wildwing/navigation_pose.py",
        "--video", str(video_path),
        "--out", str(out_root / video_path.stem),
    ]

    subprocess.run(cmd, check=True)

print("\nAll raw videos completed.")
