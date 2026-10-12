"""Show what a LeRobot dataset contains and whether it has what GAM needs.

    python benchmarks/lerobot/inspect_dataset.py user/dataset            # Hub id
    python benchmarks/lerobot/inspect_dataset.py /path/to/local/dataset  # local folder

Prints the cameras (with resolution), state / action sizes, episode count, task strings and the
config lines to put under `data:` in configs/lerobot_bimanual.yaml.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from GAM.data.lerobot_episodes import _Meta
from GAM.hub import fetch


def main():
    src = sys.argv[1]
    root = src if os.path.isdir(src) else fetch(f"hf://datasets/{src}/")
    m = _Meta(root)
    cams = {k: v["shape"] for k, v in m.features.items() if v["dtype"] in ("video", "image")}
    arrays = {k: v["shape"] for k, v in m.features.items() if v["dtype"] in ("float32", "float64", "int64", "uint16") and k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
    tasks = sorted({ep["task"] for ep in m.episodes})
    print(f"dataset {src}  (format {m.version}, {m.fps:g} fps, {len(m.episodes)} episodes, {sum(e['length'] for e in m.episodes)} frames)")
    print("cameras:", {k: f"{s[0]}x{s[1]}" for k, s in cams.items()} or "NONE - GAM needs one RGB camera")
    print("arrays: ", arrays)
    print(f"tasks ({len(tasks)}):", tasks[:10], "..." if len(tasks) > 10 else "")
    cam = next(iter(cams), None)
    H, W = (cams[cam][:2] if cam else (0, 0))
    if cam and (H < 400 or W < 400):
        print(f"note: {W}x{H} is small for the detector (it works at 1008x1008); small objects may be missed")
    print("\nconfig lines (fill in the calibration; intrinsics are for the stored resolution):")
    print(f"""data:
  source: lerobot
  repo_id: {src if not os.path.isdir(src) else os.path.basename(os.path.abspath(src))}
  root: {os.path.abspath(src) if os.path.isdir(src) else 'null'}
  camera: {cam}
  state_key: {'observation.state' if 'observation.state' in arrays else '<state feature>'}
  action_key: {'action' if 'action' in arrays else '<action feature>'}
  camera_intrinsics: [[fx, 0, {W / 2:g}], [0, fy, {H / 2:g}], [0, 0, 1]]
  camera_extrinsics: [[...], [...], [...]]        # 3x4 robot-base -> camera (OpenCV)
  slots: [<object named in the task string>]""")


if __name__ == "__main__":
    main()
