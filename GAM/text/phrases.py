"""Object phrase -> detection prompt.

The span head returns the full referring phrase ("black bowl between the plate and ramekin",
"top of the cabinet"). The open-vocabulary detector works best with the object name itself, so the
spatial clause is dropped and "part of the X" phrases resolve to X.
"""
import re

PART_HEADS = ("layer", "top", "bottom", "side", "front", "back", "compartment", "corner", "surface", "shelf", "middle", "center", "centre")
REL = re.compile(r"\s+(between|next to|beside|near|from the|on the|in the|on top of|at the|behind the|under the|inside the)\s+.*$")


def detection_prompt(phrase: str) -> str:
    t = " ".join(str(phrase).lower().split())
    t = REL.sub("", t).strip() or " ".join(str(phrase).lower().split())
    if " of the " in t:
        head, tail = t.split(" of the ", 1)
        if re.fullmatch(r"(the |a |an )?(\w+\s+)?(%s)" % "|".join(PART_HEADS), head.strip()):
            t = tail.split(" of the ")[0]                 # top of the cabinet -> cabinet
        else:
            t = head                                      # bottom drawer of the cabinet -> bottom drawer
    return REL.sub("", t).strip() or str(phrase)


LANDMARK = re.compile(r"\b(next to|beside|between|near|from|behind|in front of|to the (left|right) of|on the (left|right) of|"
                      r"in the (top|bottom|middle) (layer|part|half) of|at the (top|bottom) of|under|underneath|above|around)\s+(the |a |an |both )?$")


def manipulation_objects(instruction, spans):
    """Keep the phrases that name objects the policy acts on (things picked, opened, pressed, or placed onto)
    and drop pure landmarks ('the bowl *next to the ramekin*', 'the bowl *from the top drawer*'): a phrase
    whose left context ends with a landmark preposition is a reference, not a target.
    spans: [{"text", "span": (c0, c1)}] from T5ObjectSegmenter.extract."""
    s = " ".join(str(instruction).lower().split())
    out, prev_landmark, prev_end = [], False, -1
    for g in sorted(spans, key=lambda g: g["span"][0]):
        c0, c1 = int(g["span"][0]), int(g["span"][1]); left = s[max(0, c0 - 40):c0]
        name = re.sub(r"\s+(next to|next|beside|from|near|between|on|in|at|of|to)$", "", detection_prompt(g["text"])).strip()
        between = s[prev_end:c0] if prev_end >= 0 else ""
        landmark = bool(LANDMARK.search(left)) or (prev_landmark and re.fullmatch(r"\s*(and|,|or)\s+(the |a |an )?", between) is not None)
        # "pick the bowl ON the box and place it on the plate": an on/in phrase inside the pick clause (before any
        # place verb) locates the picked object -> landmark; the same preposition after a place verb is the target
        if not landmark and out and re.search(r"\b(on|in|inside|at)\s+(the |a |an )?$", left):
            before = s[:c0]
            if re.search(r"\b(pick|pick up|take|grab|lift|get|grasp|hold)\b", before) and not re.search(r"\b(put|place|set|drop|move|stack|insert|leave|lay|rest)\b", before):
                landmark = True
        prev_landmark, prev_end = landmark, c1
        if landmark or not name or name in out:
            continue
        out.append(name)
    return out
