"""Fine-tune the T5 object-span head on labelled instructions (this is how the released LIBERO-PRO
span head was trained).

    # 1. (optional) pre-train the head on synthetic templates x an object vocabulary
    python -m GAM.text.train_segmenter --out span_head_synth.pt --steps 1500
    # 2. fine-tune on your instructions
    python -m GAM.text.train_span_head --labels benchmarks/libero_pro/assets/span_labels.json \\
        --init span_head_synth.pt --out span_head.pt [--steps 4000] [--mix 0.3]

Labels: a JSON list of {"instruction": str, "span": [phrase, ...]}, every object phrase exactly as it
appears in the instruction (longer phrases are located first, so 'plate' inside 'bowl between the
plate and ...' is not matched twice). The Flan-T5 encoder stays frozen; only the B/I/O head is
trained. `--mix` keeps a fraction of synthetic sentences in every batch so the head does not merely
memorise the labelled set. Reports exact-match accuracy on the labelled sentences.
"""
import argparse
import json
import random

import torch
import torch.nn.functional as F

from GAM.text.segmenter import T5ObjectSegmenter
from GAM.text.train_segmenter import OBJECTS, sample as synth_sample, tags_from_spans


def locate(instruction, phrases):
    """phrases -> character spans in the instruction, longest first, non-overlapping."""
    work, out = instruction.lower(), []
    for t in sorted({" ".join(x.lower().split()) for x in phrases}, key=len, reverse=True):
        k = work.find(t)
        while k >= 0:
            out.append((k, k + len(t))); work = work[:k] + "#" * len(t) + work[k + len(t):]
            k = work.find(t, k + len(t))
    return sorted(out)


def load_labels(path):
    data, seen = [], set()
    for r in json.load(open(path)):
        ins = r["instruction"]
        if ins in seen or not r.get("span"):
            continue
        seen.add(ins)
        sp = locate(ins, r["span"])
        if len(sp) != len({" ".join(x.lower().split()) for x in r["span"]}):
            print(f"  [skipped] phrase not found in the sentence: {ins!r} {r['span']}")
            continue
        data.append((ins, sp))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, nargs="+", help="JSON lists of {instruction, span: [phrases]} (several files allowed)")
    ap.add_argument("--eval-labels", nargs="*", default=[], help="held-out JSON lists: exact-match accuracy is reported on them")
    ap.add_argument("--out", default="span_head.pt")
    ap.add_argument("--init", default="", help="head to warm-start from (train_segmenter.py output); empty = fresh head")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--mix", type=float, default=0.3, help="fraction of synthetic sentences per batch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--t5", default="google/flan-t5-base", help="frozen encoder (google/flan-t5-large for the universal head)")
    ap.add_argument("--head", choices=["mlp", "transformer"], default="mlp"); ap.add_argument("--head-hidden", type=int, default=256)
    ap.add_argument("--head-layers", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    data = [r for f in a.labels for r in load_labels(f)]
    held = [r for f in a.eval_labels for r in load_labels(f)]
    print(f"[span head] {len(data)} labelled sentences (+{len(held)} held out), synthetic mix {a.mix}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.init:
        from GAM.hub import fetch
        model = T5ObjectSegmenter.from_checkpoint(fetch(a.init), device=dev)
    else:
        model = T5ObjectSegmenter(a.t5, device=dev, head_type=a.head, head_hidden=a.head_hidden, head_layers=a.head_layers)
    print(f"[span head] encoder {model.t5_name}, head {model.head_type} ({sum(p.numel() for p in model.head.parameters()) / 1e6:.1f}M params)", flush=True)
    opt = torch.optim.AdamW(model.head.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    rng = random.Random(0); n_syn = int(a.batch * a.mix)
    for step in range(1, a.steps + 1):
        batch = [rng.choice(data) for _ in range(a.batch - n_syn)] + [synth_sample(rng, OBJECTS) for _ in range(n_syn)]
        texts, spans = zip(*batch)
        logits, m, off, _ = model(list(texts))
        L = logits.shape[1]
        y = torch.stack([torch.from_numpy(tags_from_spans(off[i].tolist(), spans[i], L)) for i in range(len(batch))]).to(dev)
        w = m.bool() & torch.tensor([[o[1] > 0 for o in off[i].tolist()] for i in range(len(batch))], device=dev)
        loss = F.cross_entropy(logits[w], y[w])
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 200 == 0 or step == 1 or step == a.steps:
            with torch.no_grad():
                em = lambda rows: sum(int([g["text"].lower() for g in model.extract(t)] == [t[c0:c1].lower() for c0, c1 in s]) for t, s in rows)
                hit = em(data[:400]); hh = em(held) if held else None
            print(f"step {step}/{a.steps}  loss {loss.item():.4f}  exact match train {hit}/{min(len(data), 400)}"
                  + (f"  held-out {hh}/{len(held)} = {100 * hh / len(held):.1f}%" if held else ""), flush=True)
    model.save_head(a.out)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
