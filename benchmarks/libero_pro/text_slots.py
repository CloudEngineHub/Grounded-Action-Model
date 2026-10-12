"""Instruction -> the four object slots [pick0, place0, pick1, place1] as text prompts.

The object phrases of each instruction come from a span table (`perception.span_cache`); an
optional second table (`perception.lang_slots`) gives the final slot assignment directly.
A place target on the table / floor has nothing to detect and stays None.
"""
import json
import re

_TABLE_HEADS = ("table", "floor", "ground", "desk", "counter", "tabletop", "center")


PLACE_VERB = r"(place|put|set|drop|leave|lay|rest|move)"
PLACE_IT = re.compile(r"^(then )?(place|put|set|drop|leave|lay|rest|slide)\s+(it|them)\b")
PICK_ONLY = re.compile(r"^(then )?(open|close|shut|turn|switch|push|pull|slide open|press)\b")


def normalise_instruction(ins, spans):
    """Small rewrites so the clause rules below see a uniform form: 'grab X and put into Y' -> '... and put it into Y';
    'put A and B in Y' (two phrases in the second clause, no 'both') -> 'put both A and B in Y'."""
    s = " ".join(str(ins).lower().split())
    s = re.sub(r" and (then )?%s (in|into|on|onto|inside|to)\b" % PLACE_VERB, r" and \1\2 it \3", s)
    sp = sorted({x.lower() for x in spans}, key=len, reverse=True)
    if "both" not in s and s.count(" and ") == 1:
        l, r = s.split(" and ")
        nl = [t for t in sp if t in l]; nr = [t for t in sp if t in r]
        if len(nl) == 1 and len(nr) == 2 and any(r.startswith(t) for t in nr) and not re.search(r"\b(it|them)\b", r):
            s = l.replace(nl[0], "both " + nl[0], 1) + " and " + r
    return s


def pairs_from_spans(ins, spans):
    """Instruction + its object phrases -> (slots, [(pick_idx, place_idx), ...]): one slot per distinct phrase in reading
    order; clauses split at ' and '; 'put it on Y' joins the previous clause; 'both A and B in Y' -> two pairs sharing Y;
    open / close / turn clauses have no place; a place on the table / floor is dropped."""
    s = " ".join(str(ins).lower().split())
    sp = [" ".join(str(x).lower().split()) for x in spans if str(x).strip()]
    work, loc = s, []
    for t in sorted(set(sp), key=len, reverse=True):
        k = work.find(t)
        while k >= 0:
            loc.append((k, t)); work = work[:k] + "#" * len(t) + work[k + len(t):]
            k = work.find(t, k + len(t))
    loc.sort()
    slots = []
    for _, t in loc:
        if t not in slots:
            slots.append(t)
    idx = {t: i for i, t in enumerate(slots)}
    table = lambda t: t is not None and t.split()[-1] in _TABLE_HEADS
    if "both" in s and len(loc) == 3:
        a, b, y = [t for _, t in loc]
        return slots, ([(idx[a], None), (idx[b], None)] if table(y) else [(idx[a], idx[y]), (idx[b], idx[y])])
    if "both" in s and len(loc) == 2 and loc[0][1].startswith("both "):
        x, y = loc[0][1][5:], loc[1][1]
        if x.endswith("s") and not x.endswith("ss"):
            x = x[:-1]                                       # "both moka pots" -> the prompt "moka pot"
        return [x] + ([] if table(y) else [y]), ([(0, None)] if table(y) else [(0, 1), (0, 1)])
    cuts = [m.start() for m in re.finditer(r" and ", work)]
    raw, pe = [], 0
    for b in cuts + [len(s)]:
        raw.append((pe, b, s[pe:b].strip())); pe = b + 5
    clauses = []
    for a, b, c in raw:
        if clauses and PLACE_IT.match(c):
            pa, _, pc = clauses[-1]; clauses[-1] = (pa, b, pc + " and " + c)
        else:
            clauses.append((a, b, c))
    pairs = []
    for a, b, c in clauses:
        here = []
        for k, t in loc:
            if a <= k < b and t not in here:
                here.append(t)
        if len(here) >= 2:
            pick, place = here[0], here[1]
        elif len(here) == 1:
            if PICK_ONLY.match(c):
                pick, place = here[0], None
            elif re.search(r"\b(it|them|inside|there|on top)\b", c) and pairs:
                pick, place = here[0], pairs[-1][0]
            else:
                pick, place = here[0], None
        else:
            if re.search(r"\b(it|them)\b", c) and pairs and pairs[-1][1]:
                pick, place = pairs[-1][1], None
            else:
                continue
        pairs.append((pick, None if table(place) else place))
    return slots, [(idx[pk], idx[pl] if pl else None) for pk, pl in pairs[:2]]


def four_slots(slots, pairs, ins):
    """(slots, pairs) -> [pick0, place0, pick1, place1]: (pick, place) pairs first, pick-only clauses after;
    'open X and put Y in X' drops the open (a precondition, not a goal); '... and close the door' re-uses the last place
    as a pick; a place target that is a side of an object ('to the left of the plate') is a table region -> None."""
    s = " ".join(str(ins).lower().split())
    tp = [(slots[i], slots[j]) for i, j in pairs if i is not None and j is not None]
    op = [(slots[i], None) for i, j in pairs if i is not None and j is None]
    places = {pl for _, pl in tp}
    op = [(pk, None) for pk, _ in op if not (pk in places and re.search(r"\bopen\b[^,]*?\b" + re.escape(pk), s))]
    if re.search(r"\band (then )?(close|shut) (the )?(door|it)\b", s) and tp and not any(pk == tp[-1][1] for pk, _ in op):
        op.append((tp[-1][1], None))
    seen = {}

    def side_of_object(pl):
        if pl is None:
            return False
        pos = [m.start() for m in re.finditer(re.escape(pl), s)]
        n = seen.get(pl, 0); seen[pl] = n + 1
        k = pos[min(n, len(pos) - 1)] if pos else -1
        return k >= 0 and re.search(r"\b(to|at|towards|in) (the )?(front|right|left|back|side)( of)? (the )?$", s[max(0, k - 40):k]) is not None
    tp = [(pk, None if side_of_object(pl) else pl) for pk, pl in tp]
    out = []
    for pk, pl in (tp + op)[:2]:
        out += [pk, pl]
    return (out + [None] * 4)[:4]


class TextSlots:
    """instruction -> ([pick0, place0, pick1, place1], phrases).
    mode "table":     the released lookup tables (exactly what the reported numbers used)
    mode "span_head": the T5 span head finds the object phrases, the rules above assign the slots"""

    def __init__(self, mode="table", span_cache=None, lang_slots=None, span_head=None, device="cuda"):
        assert mode in ("table", "span_head"), mode
        self.mode = mode
        self.spans = json.load(open(span_cache)) if span_cache else {}
        self.table = json.load(open(lang_slots)) if lang_slots else {}
        self.head = None; self.span_head = span_head; self.device = device
        if mode == "span_head":
            self._load_head()

    def _load_head(self):
        assert self.span_head, "checkpoints.span_head is required (text_slots: span_head, or an instruction outside the table)"
        from GAM.hub import fetch
        from GAM.text.segmenter import T5ObjectSegmenter
        self.head = T5ObjectSegmenter.from_checkpoint(fetch(self.span_head), device=self.device)

    def __call__(self, instruction: str):
        key = " ".join(str(instruction).split())
        if self.mode == "table" and key in self.table:
            return self.table[key], self.spans.get(key)
        if self.head is None:                                     # table mode, instruction not in the table: run the span head
            print(f"[text_slots] not in the lookup table, using the span head: {instruction!r}", flush=True); self._load_head()
        spans = [g["text"] for g in self.head.extract(instruction)]
        norm = normalise_instruction(instruction, spans)
        slots, pairs = pairs_from_spans(norm, spans)
        return four_slots(slots, pairs, norm), spans
