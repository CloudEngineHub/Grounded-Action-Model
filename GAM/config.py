"""YAML config loading.

String values may reference environment variables as ${VAR} or ${VAR:-default}; relative
paths marked with a leading "repo://" are resolved against the repository root.
"""
from __future__ import annotations

import os
import re

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(v):
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    if isinstance(v, str):
        def sub(m):
            val = os.environ.get(m.group(1), m.group(2))
            return m.group(0) if val is None else val          # unset and no default: left as written (checked where used)
        v = _VAR.sub(sub, v)
        if v.startswith("repo://"):
            v = os.path.join(REPO_ROOT, v[len("repo://"):])
    return v


def load_yaml(path: str) -> dict:
    return _expand(yaml.safe_load(open(path)))


def load_config(path: str) -> dict:
    cfg = load_yaml(path)
    for sec in ("data", "perception", "train"):
        assert sec in cfg, f"config missing section '{sec}'"
    d = cfg["data"]
    if d.get("slots") == "auto":                      # object names come from the span head (GAM.tools.track)
        n = int(d.get("n_slots", 4)); d["slots"] = [f"slot{k}" for k in range(n)]; d["slots_auto"] = True
    assert d.get("slots"), "data.slots must list at least one object slot (or 'auto')"
    return cfg


def perception_settings(cfg: dict) -> dict:
    """The perception block with the checkpoint path and defaults filled in: grounding (text | text+track | click),
    detector_depth (sensor | zero | sim), geometry_depth (sensor | pred | sim), geometry_source (box3d | box2d_depth),
    precision (fp32 | bf16), deterministic, wilddet3d (checkpoint)."""
    pc = dict(cfg.get("perception", {})); ck = cfg.get("checkpoints") or {}
    pc.setdefault("wilddet3d", ck.get("wilddet3d"))
    pc.setdefault("grounding", "text"); pc.setdefault("geometry_depth", "pred")
    pc.setdefault("detector_depth", "zero" if pc["geometry_depth"] == "pred" else "sensor")
    pc.setdefault("geometry_source", "box3d"); pc.setdefault("precision", "fp32"); pc.setdefault("deterministic", True)
    return pc


def set_precision(pc: dict):
    """fp32 + deterministic: TF32 off, cudnn deterministic (what the released caches were built with)."""
    import torch
    if pc.get("precision", "fp32") == "fp32" and pc.get("deterministic", False):
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")


def build_extractor(pc: dict, grid=None, device="cuda"):
    """PatchTokenExtractor (WildDet3D) configured from the perception block."""
    from GAM.hub import fetch
    from GAM.perception.patch_tokens import PatchTokenExtractor
    set_precision(pc)
    ext = PatchTokenExtractor(fetch(pc["wilddet3d"]), channel_swap=bool(pc.get("image_channel_swap", False)),
                              grid=int(grid or pc.get("patch_grid", 36)), bf16=(pc.get("precision", "fp32") == "bf16"),
                              cell_min_overlap=float(pc.get("cell_min_overlap", 0.30)), device=device,
                              box_whl_perm=pc.get("box_whl_perm"), letterbox=pc.get("letterbox"))
    set_precision(pc)                                 # building SAM3 turns TF32 back on
    return ext
