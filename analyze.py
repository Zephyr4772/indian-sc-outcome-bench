"""Score preds_*.jsonl from the Kaggle run. Usage: python analyze.py <results_dir>

Per model x condition x era: accuracy, balanced accuracy (unparsed counts as wrong), share answered "allowed",
unparsed rate, AUROC of p_allowed. Contamination gap = balanced acc on OLD minus RECENT, with a 95% bootstrap CI
(resampling items within each era). Writes <results_dir>/summary.json and prints Markdown tables.
"""
import glob
import json
import os
import random
import sys
from collections import defaultdict

ERAS = ("old", "recent")


def bal_acc(rows):
    per = []
    for lab in ("allowed", "dismissed"):
        r = [x for x in rows if x["label"] == lab]
        if r:
            per.append(sum(x["pred"] == lab for x in r) / len(r))
    return sum(per) / len(per) if per else float("nan")


def auroc(rows):
    # Probabilities are only meaningful when the first token actually carried the options (e.g. Mistral's first
    # generated token is a bare space, so its captured option mass is ~0): report N/A then.
    if sum(x["option_mass"] for x in rows) / max(1, len(rows)) < 0.05:
        return float("nan")
    pos = [x["p_allowed"] for x in rows if x["label"] == "allowed" and x["p_allowed"] is not None]
    neg = [x["p_allowed"] for x in rows if x["label"] == "dismissed" and x["p_allowed"] is not None]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def gap_ci(old, recent, n=2000, seed=0):
    rng = random.Random(seed)
    g = []
    for _ in range(n):
        o = [rng.choice(old) for _ in old]
        r = [rng.choice(recent) for _ in recent]
        g.append(bal_acc(o) - bal_acc(r))
    g.sort()
    return g[int(0.025 * n)], g[int(0.975 * n)]


def main(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "preds_*.jsonl"))):
        rows += [json.loads(l) for l in open(f, encoding="utf8")]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["condition"], r["era"])].append(r)
    models = sorted({r["model"] for r in rows}, key=lambda m: [x["model"] for x in rows].index(m))
    summary = []
    for m in models:
        for c in ("text", "name"):
            o, rc = groups[(m, c, "old")], groups[(m, c, "recent")]
            if not o or not rc:
                continue
            row = {"model": m, "condition": c}
            for era, g in (("old", o), ("recent", rc), ("all", o + rc)):
                row[era] = {"n": len(g), "acc": sum(x["pred"] == x["label"] for x in g) / len(g),
                            "bal_acc": bal_acc(g), "said_allowed": sum(x["pred"] == "allowed" for x in g) / len(g),
                            "unparsed": sum(x["pred"] == "unparsed" for x in g) / len(g), "auroc": auroc(g)}
            row["gap"] = row["old"]["bal_acc"] - row["recent"]["bal_acc"]
            row["gap_ci95"] = gap_ci(o, rc)
            summary.append(row)
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)

    pct = lambda x: "%.1f" % (100 * x)
    for c, title in (("text", "WITH judgment text"), ("name", "NAME + DATE only (recall probe)")):
        print("\n### %s\n" % title)
        print("| model | bal. acc all | bal. acc old | bal. acc recent | old - recent (95% CI) | said 'allowed' | unparsed | AUROC |")
        print("|---|---|---|---|---|---|---|---|")
        for s in summary:
            if s["condition"] != c:
                continue
            lo, hi = s["gap_ci95"]
            print("| %s | %s | %s | %s | %+.1f [%+.1f, %+.1f] | %s | %s | %s |" % (
                s["model"].split("/")[-1], pct(s["all"]["bal_acc"]), pct(s["old"]["bal_acc"]), pct(s["recent"]["bal_acc"]),
                100 * s["gap"], 100 * lo, 100 * hi, pct(s["all"]["said_allowed"]), pct(s["all"]["unparsed"]),
                "N/A" if s["all"]["auroc"] != s["all"]["auroc"] else "%.2f" % s["all"]["auroc"]))
    print("\nReference: always 'allowed' = 50.0 balanced (71.1 raw); random = 50.0 balanced.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
