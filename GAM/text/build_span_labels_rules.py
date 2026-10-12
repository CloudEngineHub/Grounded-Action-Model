"""Rule-based object-span labels for plain imperative instructions (DROID-style English), used to
pre-train the universal span head on large instruction corpora that have no hand labels.

    python -m GAM.text.build_span_labels_rules --jsonl annotations.jsonl --field instruction --out labels.json [--tasks tasks.txt]

Rule: an object phrase starts after a determiner (the / a / an / this / that / both / two) or a possessive and
runs until a preposition / conjunction / punctuation / verb particle. Arm and robot mentions are not objects.
Noisy by design (no parser): the span head trained on it, together with the exact RoboTwin / LIBERO labels,
smooths the noise; check the printed samples.
"""
import argparse, json, re

DET = r"(?:the|a|an|this|that|these|those|both|two|three|some|my|your|its)"
STOP = {"in", "into", "on", "onto", "to", "from", "with", "and", "then", "at", "under", "over", "next", "near", "inside", "out",
        "off", "up", "down", "left", "right", "towards", "toward", "by", "of", "for", "behind", "above", "below", "beside", "between",
        "until", "so", "after", "before", "where", "it", "them", "there", "back", "away", "together", "around", "through", "across",
        "one", "little", "bit", "time", "except", "further", "slightly", "closer", "again", "while", "using", "that", "which", "as"}
NON_OBJECT = {"arm", "arms", "hand", "hands", "gripper", "robot", "table", "left arm", "right arm", "left hand", "right hand",
              "left gripper", "right gripper", "side", "top", "bottom", "middle", "front", "way", "position", "place", "spot", "area",
              "same", "other", "end", "edge", "center", "centre", "counter", "floor", "ground", "desk", "tabletop", "workspace", "scene"}
POSITIONAL = {"top", "bottom", "back", "front", "side", "middle", "centre", "center", "edge", "end", "inside", "rest", "half", "part"}
WORD = re.compile(r"[a-z][a-z\-']*")


def spans_of(instruction):
    s = " ".join(instruction.lower().replace(",", " , ").replace(".", " . ").split())
    toks = s.split(); out = []; i = 0
    while i < len(toks):
        if re.fullmatch(DET, toks[i]) and i + 1 < len(toks):
            j = i + 1; words = []
            while j < len(toks) and WORD.fullmatch(toks[j]) and toks[j] not in STOP and not re.fullmatch(DET, toks[j]):
                words.append(toks[j]); j += 1
            phrase = " ".join(words)
            if words and words[-1] in POSITIONAL and j + 2 < len(toks) and toks[j] == "of" and re.fullmatch(DET, toks[j + 1]):
                i = j + 1; continue                           # "top of the cabinet": the object is the cabinet
            if words and phrase not in NON_OBJECT and words[-1] not in NON_OBJECT and phrase not in out:
                out.append(phrase)
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jsonl", nargs="*", default=[]); ap.add_argument("--field", default="instruction")
    ap.add_argument("--tasks", nargs="*", default=[], help="plain text files, one instruction per line"); ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0); ap.add_argument("--source", default="rules")
    a = ap.parse_args()
    sents = []
    for f in a.jsonl:
        for ln in open(f):
            try:
                sents.append(json.loads(ln)[a.field])
            except Exception:
                pass
    for f in a.tasks:
        sents += [ln.strip() for ln in open(f) if ln.strip()]
    seen, rows = set(), []
    for s in sents:
        k = " ".join(str(s).lower().split())
        if not k or k in seen:
            continue
        seen.add(k); sp = spans_of(s)
        if sp:
            rows.append({"source": a.source, "instruction": " ".join(str(s).split()), "span": sp})
        if a.max and len(rows) >= a.max:
            break
    json.dump(rows, open(a.out, "w"))
    print(f"{len(sents)} sentences, {len(seen)} unique, {len(rows)} labelled -> {a.out}")
    import random; random.seed(1)
    for r in random.sample(rows, min(25, len(rows))):
        print("  ", repr(r["instruction"]), "->", r["span"])


if __name__ == "__main__":
    main()
