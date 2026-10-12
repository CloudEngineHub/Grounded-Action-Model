"""Instruction -> T5 span head (GAM.text.segmenter): the whole-sentence embedding that train.lang_mode: film
feeds to the policy. The span head checkpoint comes from checkpoints.span_head (or text.span_head).
"""
import numpy as np

_seg = None


def get_segmenter(cfg):
    """Load (once) the T5 segmenter from config: top-level `text:` section, falling back to
    checkpoints.span_head: object spans for grounding and the whole-sentence embedding for lang_mode: film."""
    global _seg
    if _seg is None:
        from GAM.text import T5ObjectSegmenter
        t = dict(cfg.get("text") or {})
        c = cfg["data"].get("clicks", {})
        ckpt = t.get("span_head") or (cfg.get("checkpoints") or {}).get("span_head")
        from GAM.hub import fetch
        ckpt = fetch(ckpt) if ckpt else ckpt
        assert ckpt, "checkpoints.span_head (or text.span_head) is required for the language pathway"
        _seg = T5ObjectSegmenter.from_checkpoint(
            ckpt, cache_dir=t.get("hf_cache_dir") or c.get("hf_cache_dir"))
    return _seg


def sentence_embedding(cfg, instruction):
    """Whole-sentence T5 embedding (768,), the CondFuse language token's input."""
    return get_segmenter(cfg).sentence_embedding(instruction or "")


