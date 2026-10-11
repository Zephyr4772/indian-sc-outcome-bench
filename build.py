"""Build the Indian Supreme Court outcome benchmark (recent vs old, matched) from sinhal/Indian_Supreme_Court_Judgments.

Gold label: allowed / dismissed, kept only when the dataset's `outcome` field AND the judgment's own closing paragraphs
agree ("appeal is allowed" / "appeal is dismissed"). Partly-allowed, disposed-of and ambiguous cases are excluded.
Input: the opening of the judgment (facts and arguments), cut at a paragraph boundary before the final 35% of the
text and capped at 14,000 characters. Items whose kept text still states the outcome are dropped.
Writes data/test.jsonl.
"""
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "raw", "laww_dataset.jsonl")
OUT = os.path.join(HERE, "data", "test.jsonl")
SEED = 26
MAX_CHARS, KEEP_FRAC = 14000, 0.65

SUBJ = r"\b(?:appeals?|petitions?|writ petitions?|special leave petitions?|civil appeals?|criminal appeals?)\s+"
VERB = r"(?:is|are|stands?|stand)\s+(?:accordingly\s+|hereby\s+)?"
ALLOW = re.compile(SUBJ + VERB + r"allowed", re.I)
DISMISS = re.compile(SUBJ + VERB + r"dismissed", re.I)
PART = re.compile(r"partly allowed|partially allowed|allowed in part", re.I)
LEAK = re.compile(SUBJ + VERB + r"(?:partly\s+|partially\s+)?(?:allowed|dismissed)|we\s+(?:allow|dismiss)\s+(?:the|this|these)\s+appeal", re.I)
NOISE = re.compile(r"^\s*(?:http://JUDIS\.NIC\.IN|Page \d+ of \d+|Signature Not Verified.*|Digitally signed.*|Reason:.*|Date: \d.*)\s*$", re.I | re.M)


def year_of(r):
    m = re.search(r"(\d{4})$", str(r.get("judgment_dates", "")).split(",")[0].strip())
    return int(m.group(1)) if m else 0


def label_of(r):
    t = r.get("full_text") or ""
    tail = t[-2500:]
    a, d = bool(ALLOW.search(tail)), bool(DISMISS.search(tail))
    if PART.search(tail) or a == d:
        return None
    text_label = "allowed" if a else "dismissed"
    o = r.get("outcome")
    if o not in ("allowed", "dismissed", "set aside", "upheld"):
        return None
    meta = "allowed" if o in ("allowed", "set aside") else "dismissed"
    return text_label if meta == text_label else None


def make_input(t):
    t = NOISE.sub("", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    cut = min(int(len(t) * KEEP_FRAC), MAX_CHARS)
    head = t[:cut]
    para = head.rfind("\n\n")
    head = head[:para] if para > cut * 0.6 else head[: head.rfind(". ") + 1 or cut]
    return head.strip()


def main():
    pools = {"recent": {"allowed": [], "dismissed": []}, "old": {"allowed": [], "dismissed": []}}
    seen = set()
    for line in open(SRC, encoding="utf8", errors="ignore"):
        r = json.loads(line)
        y = year_of(r)
        era = "recent" if y >= 2023 else ("old" if 1995 <= y <= 2005 else None)
        if not era or str(r.get("language")) not in ("nan", "ENG", "None", "") or r.get("is_short_order"):
            continue
        t = r.get("full_text") or ""
        if len(t) < 6000 or r["temp_link"] in seen:  # same PDF can appear under several years
            continue
        lab = label_of(r)
        if not lab:
            continue
        x = make_input(t)
        if len(x) < 3000 or LEAK.search(x):
            continue
        seen.add(r["temp_link"])
        pools[era][lab].append({"source_id": str(r["csv_id"]), "era": era, "year": y,
                                "petitioner": r.get("pet"), "respondent": r.get("res"),
                                "judgment_date": r.get("judgment_dates"), "pdf": r.get("temp_link"),
                                "text": x, "label": lab, "full_text_chars": len(t)})
    rng = random.Random(SEED)
    items = list(pools["recent"]["allowed"]) + list(pools["recent"]["dismissed"])
    na, nd = len(pools["recent"]["allowed"]), len(pools["recent"]["dismissed"])
    items += rng.sample(pools["old"]["allowed"], na) + rng.sample(pools["old"]["dismissed"], nd)
    rng.shuffle(items)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf8") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"id": "ISC-%04d" % i, **it}, ensure_ascii=False) + "\n")
    print({e: {k: len(v) for k, v in p.items()} for e, p in pools.items()}, "->", len(items), "items")


if __name__ == "__main__":
    main()
