"""Collect instruction sentences from LeRobot datasets on the Hugging Face Hub (metadata only).

    python -m GAM.text.collect_hub_tasks --out hub_tasks.txt [--limit 600] [--author lerobot]

Every LeRobot dataset stores its task strings in meta/ (tasks.jsonl for v2.x, tasks.parquet for v3.0);
only those few KB are downloaded per dataset. Output: one unique instruction per line, plus a
<out>.sources.json with the dataset each sentence came from.
"""
import argparse, json, os, re

from huggingface_hub import HfApi, snapshot_download


def tasks_of(root):
    out = set()
    p = os.path.join(root, "meta", "tasks.jsonl")
    if os.path.exists(p):
        for ln in open(p):
            try:
                out.add(str(json.loads(ln)["task"]))
            except Exception:
                pass
    p = os.path.join(root, "meta", "tasks.parquet")
    if os.path.exists(p):
        import pyarrow.parquet as pq
        t = pq.read_table(p).to_pandas()
        out |= {str(i) for i in (t["task"] if "task" in t.columns else t.index)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--author", default=None); ap.add_argument("--min-words", type=int, default=2)
    a = ap.parse_args(); api = HfApi()
    ds = list(api.list_datasets(tags="LeRobot", author=a.author, limit=a.limit, sort="downloads", direction=-1))
    print(f"{len(ds)} LeRobot datasets listed", flush=True)
    sents, src = {}, {}
    for i, d in enumerate(ds):
        try:
            root = snapshot_download(d.id, repo_type="dataset", allow_patterns=["meta/tasks.*", "meta/info.json"])
            for t in tasks_of(root):
                t = " ".join(t.split())
                if len(t.split()) >= a.min_words and re.search(r"[a-zA-Z]", t) and not re.search(r"[一-鿿]", t):
                    k = t.lower()
                    if k not in sents:
                        sents[k] = t; src[k] = d.id
        except Exception as e:
            print(f"  skip {d.id}: {type(e).__name__}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)} datasets, {len(sents)} sentences", flush=True)
    open(a.out, "w").write("\n".join(sents.values()))
    json.dump(src, open(a.out + ".sources.json", "w"))
    print(f"{len(sents)} unique instructions from {len(set(src.values()))} datasets -> {a.out}")


if __name__ == "__main__":
    main()
