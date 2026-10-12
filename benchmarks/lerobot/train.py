"""Train GAM on a LeRobot dataset with one command.

    python benchmarks/lerobot/train.py --config configs/lerobot_bimanual.yaml [--mode precompute|e2e] [--batch-size 16 --grad-accum 3] [--max-episodes 3]

  First GAM.tools.track (object phrases from the task string, frame-0 detection, offline CoTracker) when
  data.clicks.source is tracks. Then:
  precompute (default): run GAM.precompute over every episode once (frozen WildDet3D -> feature cache on disk,
                        episodes already in the cache are skipped), then GAM.train on the cache. Training is fast
                        (no perception in the loop) but the cache must be stored: roughly 0.3-1 GB per 1000 frames
                        with the 36x36 patch grid, and it must be rebuilt when the perception settings change.
  e2e:                  GAM.train --e2e, the perception runs inside the training loop, nothing is written to disk;
                        ~0.6-1 s per sample, so much slower per step.

The dataset is read directly from its files (Hub id downloaded on first use, or a local folder).
The resulting checkpoint (train.out_dir/latest.pt) is evaluated with GAM/eval/inprocess_policy.py or
served with benchmarks/robotwin/server.py.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from GAM.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--mode", choices=["precompute", "e2e"], default="precompute")
    ap.add_argument("--batch-size", type=int, default=0); ap.add_argument("--grad-accum", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--gpus", type=int, default=1, help="GPUs for training (torchrun)")
    ap.add_argument("--max-episodes", type=int, default=0, help="only track / precompute the first N episodes (a quick smoke run)")
    a = ap.parse_args()
    cfg = load_config(a.config); assert cfg["data"].get("source") == "lerobot", "data.source must be lerobot (see configs/lerobot_bimanual.yaml)"
    py = [sys.executable]
    extra = (["--batch-size", str(a.batch_size)] if a.batch_size else []) + (["--grad-accum", str(a.grad_accum)] if a.grad_accum else []) \
        + (["--max-steps", str(a.max_steps)] if a.max_steps else [])
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    lim = ["--ep-end", str(a.max_episodes)] if a.max_episodes else []
    if cfg["data"].get("clicks", {}).get("source") == "tracks":
        print(f"[lerobot] tracks (span head + offline CoTracker) -> {cfg['data']['clicks'].get('tracks_dir')}", flush=True)
        subprocess.check_call(py + ["-m", "GAM.tools.track", "--config", a.config] + lim, cwd=ROOT, env=env)
    if a.mode == "precompute":
        print(f"[lerobot] precompute -> {cfg['train']['cache_dir']}", flush=True)
        subprocess.check_call(py + ["-m", "GAM.precompute", "--config", a.config] + lim, cwd=ROOT, env=env)
        train = py + ["-m", "GAM.train", "--config", a.config] + extra
    else:
        train = py + ["-m", "GAM.train", "--config", a.config, "--e2e"] + extra
    if a.gpus > 1:
        train = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(a.gpus)] + train[1:]
    print(f"[lerobot] train ({a.mode}) -> {cfg['train']['out_dir']}", flush=True)
    subprocess.check_call(train, cwd=ROOT, env=env)
    print(f"[lerobot] done: {os.path.join(cfg['train']['out_dir'], 'latest.pt')}")


if __name__ == "__main__":
    main()
