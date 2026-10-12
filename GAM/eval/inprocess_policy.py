"""Closed-loop policy adapter with the get_model / encode_obs / eval / reset_model contract that
RoboTwin-style harnesses use, GAMRunner (perception + policy) and ActionChunker (replan / tempens / rtc),
everything driven by the yaml (`eval.chunking` and its sub-sections).

    model = get_model({"config_path": "configs/robotwin/click_bell.yaml", "ckpt_path": None})   # or env GAM_CONFIG / GAM_CKPT
    reset_model(model)                                   # at the start of every episode
    eval(TASK_ENV, model, observation)                   # executes actions until the next policy query is due

The harness supplies an observation dict with
    obs["observation"]["head_camera"]{rgb (H,W,3) uint8, depth (H,W) mm or m, intrinsic_cv (3,3), extrinsic_cv (3,4) base->cam}
    obs["joint_action"]["vector"]   the joint state (layout of the checkpoint)
and a TASK_ENV with take_action(action, action_type="qpos") and, optionally, get_instruction() (used for lang_mode film).
For your own robot call the pieces directly: GAMRunner.act(rgb, depth, K, E, state, prefix) and ActionChunker.act(predict).
"""
from __future__ import annotations

import os

import numpy as np

from GAM.config import load_config
from GAM.eval.chunker import ActionChunker
from GAM.eval.runner import GAMRunner

GRIPPER_DIMS = (6, 13)                                 # default (bimanual 7+7 layout); eval.gripper_dims overrides, e.g. [7] for a single-arm Franka


class GAMModel:
    def __init__(self, config_path, ckpt=None):
        self.cfg = load_config(config_path)
        self.runner = GAMRunner(self.cfg, ckpt or self.cfg["checkpoints"]["policy"])
        self.chunker = ActionChunker.from_config(self.cfg.get("eval", {}), gripper_dims=GRIPPER_DIMS)
        self.n_ep = -1; self.ready = False

    def reset(self, instruction=""):
        self.n_ep += 1; self.runner.reset(instruction, seed=self.n_ep); self.chunker.reset(); self.ready = True

    def predict(self, observation, prefix=None):
        hc = observation["observation"]["head_camera"]
        chunk, _ = self.runner.act(hc["rgb"], hc["depth"], hc["intrinsic_cv"], hc["extrinsic_cv"], observation["joint_action"]["vector"], prefix)
        return chunk


def encode_obs(observation):
    return observation


def get_model(usr_args):
    cfg = (usr_args or {}).get("config_path") or os.environ["GAM_CONFIG"]
    return GAMModel(cfg, (usr_args or {}).get("ckpt_path") or os.environ.get("GAM_CKPT"))


def reset_model(model):
    model.ready = False


def eval(TASK_ENV, model, observation):  # noqa: A001
    if not model.ready:
        model.reset(TASK_ENV.get_instruction() if hasattr(TASK_ENV, "get_instruction") else "")
    obs = encode_obs(observation)
    while True:                                          # one query, then the actions it covers
        a = model.chunker.act(lambda prefix: model.predict(obs, prefix))
        TASK_ENV.take_action(a, action_type="qpos")
        if model.chunker.needs_query():
            break
        if model.runner.grounding == "text+track" and hasattr(TASK_ENV, "get_obs"):
            model.runner.advance(TASK_ENV.get_obs()["observation"]["head_camera"]["rgb"])   # the tracker sees every frame
