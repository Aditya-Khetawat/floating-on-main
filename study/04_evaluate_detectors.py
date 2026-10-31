"""Step 4 (RQ1): compare PinCheck with the pattern-matching baseline.

For every file of a sample, runs
  P  PinCheck as frozen for the evaluation (pipeline_rule="legacy"),
  N  the same without constant propagation (ablation),
  G  the line-based baseline (baseline.py),
matches call sites by (file, line, api), classifies every disagreement by
cause, applies the adjudicated verdict for that cause, and computes the
measures reported in Tables II and III.

Ground truth. Disagreements were grouped by cause; for each cause, randomly
drawn cases (all cases for small causes) were inspected manually and every
inspected case confirmed the cause-level verdict below. 30 random agreements
were also inspected and found correct. The inspected sites are flagged in
data/adjudication_test.csv. Sites missed by both tools cannot be counted, so
recall is relative to the union of both tools' detections.

Usage: python study/04_evaluate_detectors.py [--split test] [--files-dir output/evalfiles/test]
"""
import argparse
import collections
import io
import json
import os
import re
import tokenize

from common import data_path, out_path
from baseline import detect as baseline_detect
from pincheck.core import analyze_source

# Verdict for each disagreement cause, established by manual adjudication.
VERDICTS = {
    "ID: constant propagation": "pincheck_correct",
    "ID: multi-line literal": "pincheck_correct",
    "REV: revision on another line": "pincheck_correct",
    "RX-FP: function definition": "baseline_false_site",
    "RX-only: inside string literal": "baseline_false_site",
    "RX-FP: pipeline object invocation": "baseline_false_site",
    "RX-only: other": "baseline_false_site",
    "ID: none vs indirect": "both_false_site",  # project-defined pipeline() function
}


def pincheck_sites(src, cp):
    found = analyze_source(src, constant_propagation=cp, pipeline_rule="legacy")
    if found is None:
        return None
    return [dict(line=f.line, api=f.api, id_class=f.id_class, model_id=f.model_id,
                 revision="absent" if f.revision == "absent" else "pinned") for f in found]


def string_spans(src):
    spans = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.STRING:
                spans.append((t.start, t.end))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return spans


def cause(row, src_lines, spans):
    P, G, N = row["P"], row["G"], row["N"]
    line = src_lines[row["line"] - 1]
    causes = []
    if G and not P:
        if re.search(r"\bdef\s+" + row["api"] + r"\s*\(", line):
            causes.append("RX-FP: function definition")
        else:
            m = re.search(r"\b" + row["api"] + r"\s*\(", line)
            pos = (row["line"], m.start() if m else 0)
            if any(s <= pos < e for s, e in spans):
                causes.append("RX-only: inside string literal")
            elif row["api"] == "pipeline" and re.search(r"\.\s*pipeline\s*\(", line):
                causes.append("RX-FP: pipeline object invocation")
            else:
                causes.append("RX-only: other")
    elif P and G:
        if P["id_class"] != G["id_class"]:
            if N and N["id_class"] != P["id_class"]:
                causes.append("ID: constant propagation")
            elif P["id_class"] == "hub" and G["id_class"] == "indirect":
                causes.append("ID: multi-line literal")
            else:
                causes.append(f"ID: {P['id_class']} vs {G['id_class']}")
        if P["revision"] != G["revision"]:
            causes.append("REV: revision on another line")
    return causes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--files-dir", default=None)
    args = ap.parse_args()
    files_dir = args.files_dir or out_path("evalfiles", args.split)
    spec = json.load(open(data_path("eval_samples.json")))[args.split]

    rows, n_files = [], 0
    for f in spec["files"]:
        local = os.path.join(files_dir, f"{f['repo'].replace('/', '__')}__{f['path'].replace('/', '__')}")
        if not os.path.exists(local):
            continue
        n_files += 1
        src = open(local, encoding="utf-8").read()
        P = pincheck_sites(src, True) or []
        N = pincheck_sites(src, False) or []
        G = baseline_detect(src)
        kP = {(c["line"], c["api"]): c for c in P}
        kN = {(c["line"], c["api"]): c for c in N}
        kG = {}
        for c in G:
            kG.setdefault((c["line"], c["api"]), c)
        lines, spans = src.splitlines(), string_spans(src)
        for k in sorted(set(kP) | set(kG)):
            row = {"repo": f["repo"], "path": f["path"], "line": k[0], "api": k[1],
                   "P": kP.get(k), "N": kN.get(k), "G": kG.get(k)}
            row["causes"] = cause(row, lines, spans)
            rows.append(row)
    json.dump(rows, open(out_path(f"eval_rows_{args.split}.json"), "w"))

    by_cause = collections.Counter(c for r in rows for c in r["causes"])
    print(f"{args.split}: {n_files} files, {len(rows)} candidate sites")
    print("Disagreement causes (Table III):")
    for c, n in sorted(by_cause.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {c}  -> {VERDICTS.get(c, 'UNADJUDICATED')}")
    unknown = set(by_cause) - set(VERDICTS)
    if unknown:
        print("WARNING: causes without an adjudicated verdict:", unknown)

    def false_site(r):
        return any(VERDICTS.get(c, "").endswith("false_site") for c in r["causes"])
    false_both = [r for r in rows if "ID: none vs indirect" in r["causes"]]
    true_sites = [r for r in rows if r["P"] and r["G"] and not false_site(r)]
    T = len(true_sites)
    n_P = sum(1 for r in rows if r["P"])
    n_G = sum(1 for r in rows if r["G"])
    hub = [r for r in true_sites if r["P"]["id_class"] == "hub"]
    pinned = [r for r in true_sites if r["P"]["revision"] == "pinned"]
    g_hub_all = [r for r in rows if r["G"] and r["G"]["id_class"] == "hub"]
    nonlocal_P = [r for r in true_sites if r["P"]["id_class"] != "local"]
    nonlocal_G = [r for r in rows if r["G"] and r["G"]["id_class"] != "local"]
    pct = lambda a, b: round(100 * a / b, 1) if b else float("nan")
    table = {
        "detection_precision": {"baseline": pct(T, n_G), "pincheck": pct(T, n_P)},
        "detection_recall_vs_union": {"baseline": pct(T, T), "pincheck": pct(T, T)},
        "hub_literal_recall": {"baseline": pct(sum(r["G"]["id_class"] == "hub" for r in hub), len(hub)),
                               "pincheck_no_cp": pct(sum(r["N"] and r["N"]["id_class"] == "hub" for r in hub), len(hub)),
                               "pincheck": 100.0, "n": len(hub)},
        "hub_literal_precision_baseline": pct(sum(1 for r in g_hub_all if r in hub), len(g_hub_all)),
        "revision_recall": {"baseline": pct(sum(r["G"]["revision"] == "pinned" for r in pinned), len(pinned)),
                            "pincheck": 100.0, "n": len(pinned)},
        "estimated_pinning_rate": {"baseline": pct(sum(r["G"]["revision"] == "pinned" for r in nonlocal_G), len(nonlocal_G)),
                                   "pincheck": pct(sum(r["P"]["revision"] == "pinned" for r in nonlocal_P), len(nonlocal_P))},
        "true_sites": T, "baseline_false_sites": n_G - T, "pincheck_false_sites": n_P - T,
        "sites_false_for_both": len(false_both),
    }
    json.dump(table, open(out_path(f"rq1_{args.split}.json"), "w"), indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
