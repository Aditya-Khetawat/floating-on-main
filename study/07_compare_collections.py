"""Compare the two data collections (Section VI-A, "Reliability").

data/results_2026-10-05.jsonl  first collection, 5 October 2026
data/results.jsonl             re-collection used in the paper, 20 October 2026

Both cover the same 939 candidate repositories. The first collection was produced
by an earlier PinCheck version that counted any bare pipeline(...) call in a file
mentioning transformers (the "legacy" rule, see README). That rule only affects
the default-pipeline measure, which is therefore reported separately and excluded
from the maximum difference.
"""
from common import data_path, load_results


def lib_pinned(r):
    return any(v == "exact" for v in r["deps"].values()) or r["lockfile"]


def rates(path):
    repos = [r for r in load_results(path) if r["calls"]]
    n = len(repos)
    pin = [r for r in repos if any(c["revision"] != "absent" for c in r["calls"])]
    lib = [r for r in repos if lib_pinned(r)]
    pct = lambda k: round(100 * k / n, 1)
    return n, {
        "never pin a model": pct(n - len(pin)),
        "any revision (pin)": pct(len(pin)),
        "strict pin (hash or named ref)": pct(sum(any(c["revision"] in ("hash", "named_ref") for c in r["calls"]) for r in repos)),
        "pin a commit hash": pct(sum(any(c["revision"] == "hash" for c in r["calls"]) for r in repos)),
        "library-pinned": pct(len(lib)),
        "library-pinned but no model pinned (of library-pinned)": round(100 * sum(r not in pin for r in lib) / len(lib), 1),
        "use trust_remote_code": pct(sum(any(c["trust_remote_code"] for c in r["calls"]) for r in repos)),
        "CI workflows": pct(sum(r["ci"] for r in repos)),
        "automated tests": pct(sum(r["tests"] for r in repos)),
        "Docker": pct(sum(r["docker"] for r in repos)),
        "lockfile": pct(sum(r["lockfile"] for r in repos)),
    }, pct(sum(any(c["pipeline_default"] for c in r["calls"]) for r in repos))


def main():
    n1, a, dp1 = rates(data_path("results_2026-10-05.jsonl"))
    n2, b, dp2 = rates(data_path("results.jsonl"))
    print(f"{'project-level rate (%)':58s} {'5 Oct':>7s} {'20 Oct':>7s} {'diff':>6s}")
    print(f"{'model-consuming repositories (n)':58s} {n1:7d} {n2:7d}")
    practice = {"CI workflows", "automated tests", "Docker", "lockfile"}
    pin_diffs, practice_diffs = [], []
    for k in a:
        d = round(b[k] - a[k], 1)
        (practice_diffs if k in practice else pin_diffs).append(abs(d))
        print(f"{k:58s} {a[k]:7.1f} {b[k]:7.1f} {d:+6.1f}")
    print(f"\nmaximum absolute difference, pinning rates:   {max(pin_diffs):.1f} percentage points")
    print(f"maximum absolute difference, practice signals: {max(practice_diffs):.1f} percentage points")
    print(f"default pipeline (analyzer rule changed between runs): {dp1:.1f}% -> {dp2:.1f}%")


if __name__ == "__main__":
    main()
