"""Span-head training data from RoboTwin's instruction templates.

    python benchmarks/robotwin/build_span_labels.py --robotwin $ROBOTWIN_ROOT --out benchmarks/robotwin/assets/span_labels_robotwin.json

RoboTwin ships ~2900 instructions over 50 tasks (description/task_instruction/*.json, 'seen' + 'unseen')
with placeholders: {A}, {B}, ... stand for objects (filled here with phrases from its object-description lists,
description/objects_description/*/*.json) and {a}, {b} for arms. Every object phrase in the filled sentence
(longest first, non-overlapping) is an object span.
Output rows: {"task", "split": seen|unseen, "instruction", "span": [phrases]}, the format of GAM.text.train_span_head.
"""
import argparse, glob, json, os, re


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--robotwin", required=True); ap.add_argument("--out", required=True); ap.add_argument("--fills", type=int, default=3, help="object phrases drawn per template")
    a = ap.parse_args(); D = os.path.join(a.robotwin, "description")
    vocab = set()
    for f in glob.glob(os.path.join(D, "objects_description", "*", "*.json")):
        o = json.load(open(f))
        for k in ("raw_description", "seen", "unseen"):
            v = o.get(k) or []
            for p in ([v] if isinstance(v, str) else v):
                p = " ".join(str(p).lower().split())
                if p and len(p) < 60:
                    vocab.add(p)
    vocab = sorted(vocab, key=len, reverse=True)
    pat = re.compile(r"\b(" + "|".join(re.escape(p) for p in vocab) + r")\b")
    import random
    rng = random.Random(0); ARMS = ["left arm", "right arm"]
    rows, n_sent = [], 0
    for f in sorted(glob.glob(os.path.join(D, "task_instruction", "*.json"))):
        task = os.path.splitext(os.path.basename(f))[0]; d = json.load(open(f))
        for split in ("seen", "unseen"):
            for tpl in d.get(split) or []:
                n_sent += 1
                for _ in range(a.fills):                              # {A}, {B}, ... = object phrases; {a}, {b} = arms
                    fill = {}
                    for ph in sorted(set(re.findall(r"\{([A-Za-z])\}", tpl))):
                        fill[ph] = rng.choice(vocab) if ph.isupper() else rng.choice(ARMS)
                    s_ = tpl
                    for ph, v in fill.items():
                        s_ = s_.replace("{" + ph + "}", v)
                    low = s_.lower(); spans, taken = [], [False] * len(low)
                    for m in pat.finditer(low):                       # longest phrases first thanks to the alternation order
                        if not any(taken[m.start():m.end()]):
                            spans.append(m.group(1)); taken[m.start():m.end()] = [True] * (m.end() - m.start())
                    if spans:
                        rows.append({"task": task, "split": split, "instruction": s_, "span": spans})
                    if not fill:
                        break
    json.dump(rows, open(a.out, "w"), indent=0)
    print(f"{len(vocab)} object phrases; {n_sent} templates -> {len(rows)} labelled sentences -> {a.out}")
    for r in rows[:3] + rows[-2:]:
        print("  ", r["task"], r["split"], repr(r["instruction"]), "->", r["span"])


if __name__ == "__main__":
    main()
