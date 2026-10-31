"""Step 5 (RQ2-RQ5): compute every statistic reported in Section V.

Reads data/results.jsonl (or --results) and writes output/summary.json.
Definitions follow Section IV:
  pi(C)   share of call sites with any revision                      Eq. (1)
  pin(P)  1 if any call site of repository P specifies a revision     Eq. (2)
  kappa   pinning coverage pi(C_P) of a pinning repository            Eq. (2)
  M(P)    CI + Test + Docker + LibPin indicators                      Eq. (3)
  OR      odds ratio (a/b)/(c/d), Fisher exact test                   Eq. (4)
  logit   logistic regression of pin(P) on six predictors             Eq. (5)
  R2_McF  McFadden pseudo-R^2                                         Eq. (6)
"""
import argparse
import collections
import json
import statistics as st

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact, mannwhitneyu, norm

from common import COLLECTION_DATE, data_path, load_results, out_path


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def lib_pinned(r):
    return any(v == "exact" for v in r["deps"].values()) or r["lockfile"]


def fisher(repos, pred, key="pin"):
    X = [r for r in repos if pred(r)]
    Y = [r for r in repos if not pred(r)]
    a, c = sum(r[key] for r in X), sum(r[key] for r in Y)
    b, d = len(X) - a, len(Y) - c
    odds, p = fisher_exact([[a, b], [c, d]])
    return {"with": f"{a}/{len(X)}", "with_pct": pct(a, len(X)), "without": f"{c}/{len(Y)}",
            "without_pct": pct(c, len(Y)), "OR": round(odds, 2), "p": round(p, 4)}


def cochran_armitage(n, k):
    x = np.arange(len(n))
    p = k.sum() / n.sum()
    T = (x * (k - n * p)).sum()
    V = p * (1 - p) * ((n * x ** 2).sum() - (n * x).sum() ** 2 / n.sum())
    z = T / np.sqrt(V)
    return round(float(z), 2), float(2 * (1 - norm.cdf(abs(z))))


def headline(repos):
    calls = [c for r in repos for c in r["calls"]]
    lib = [r for r in repos if r["lib"]]
    trc_repos = [r for r in repos if any(c["trust_remote_code"] for c in r["calls"])]
    trc = [c for c in calls if c["trust_remote_code"]]
    return {"n": len(repos), "never_pin_pct": pct(sum(not r["pin"] for r in repos), len(repos)),
            "hash_pct": pct(sum(r["cat"] == "hash" for r in repos), len(repos)),
            "call_sites_unpinned_pct": pct(sum(c["revision"] == "absent" for c in calls), len(calls)),
            "libpinned_no_model_pct": pct(sum(not r["pin"] for r in lib), len(lib)),
            "remote_code_repos_pct": pct(len(trc_repos), len(repos)),
            "remote_code_sites_unpinned_pct": pct(sum(c["revision"] == "absent" for c in trc), len(trc))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=data_path("results.jsonl"))
    args = ap.parse_args()
    raw = load_results(args.results)
    S = {"repositories_analyzed": len(raw), "errors": sum(1 for r in raw if r["error"]),
         "python_files": sum(r["py_files"] for r in raw), "notebooks": sum(r["nb_files"] for r in raw)}
    repos = [r for r in raw if r["calls"]]
    for r in repos:
        revs = {c["revision"] for c in r["calls"]}
        r["pin"] = revs != {"absent"}
        r["strict"] = bool(revs & {"hash", "named_ref"})
        r["cat"] = ("hash" if "hash" in revs else "named_ref" if "named_ref" in revs
                    else "configurable" if "indirect" in revs else "none")
        r["lib"] = lib_pinned(r)
        r["age"] = (pd.Timestamp(COLLECTION_DATE, tz="UTC") - pd.Timestamp(r["created_at"])).days / 365.25
        r["M"] = int(r["ci"]) + int(r["tests"]) + int(r["docker"]) + int(r["lib"])
    calls = [dict(c, repo=r["repo"]) for r in repos for c in r["calls"]]
    hub = [c for c in calls if c["id_class"] == "hub"]

    # RQ2: prevalence
    S["model_consuming_repositories"] = len(repos)
    S["call_sites"] = len(calls)
    S["distinct_hub_ids"] = len({c["model_id"] for c in hub})
    S["stars"] = {"median": st.median(r["stars"] for r in repos), "min": min(r["stars"] for r in repos),
                  "max": max(r["stars"] for r in repos)}
    S["identifier_classes"] = dict(collections.Counter(c["id_class"] for c in calls))
    S["call_revision"] = dict(collections.Counter(c["revision"] for c in calls))
    S["repo_revision_category"] = dict(collections.Counter(r["cat"] for r in repos))
    S["repos_all_calls_pinned"] = sum(all(c["revision"] != "absent" for c in r["calls"]) for r in repos)
    hub_repos = [r for r in repos if any(c["id_class"] == "hub" for c in r["calls"])]
    S["hub_literal_sites"] = {"n": len(hub), "unpinned_pct": pct(sum(c["revision"] == "absent" for c in hub), len(hub)),
                              "revision": dict(collections.Counter(c["revision"] for c in hub)),
                              "repos": len(hub_repos),
                              "repos_pinning_a_literal": sum(any(c["id_class"] == "hub" and c["revision"] != "absent"
                                                                 for c in r["calls"]) for r in hub_repos)}
    api = {}
    for a in sorted({c["api"] for c in calls}):
        cs = [c for c in calls if c["api"] == a]
        reps = {c["repo"] for c in cs}
        api[a] = {"call_sites": len(cs), "call_pin_pct": pct(sum(c["revision"] != "absent" for c in cs), len(cs)),
                  "repos": len(reps), "repo_pin_pct": pct(len({c["repo"] for c in cs if c["revision"] != "absent"}), len(reps))}
    S["by_interface"] = api
    cov = [sum(c["revision"] != "absent" for c in r["calls"]) / len(r["calls"]) for r in repos if r["pin"]]
    S["pinning_coverage_median_pct"] = round(100 * st.median(cov), 1)
    unpinned = collections.Counter()
    for r in repos:
        for m in {c["model_id"] for c in r["calls"] if c["id_class"] == "hub" and c["revision"] == "absent"}:
            unpinned[m] += 1
    S["top_unpinned_ids"] = unpinned.most_common(10)
    ids = {c["model_id"] for c in hub}
    dm = [len({c["model_id"] for c in r["calls"] if c["id_class"] == "hub"}) for r in hub_repos]
    is_test = lambda f: "/tests/" in "/" + f or "/test/" in "/" + f or f.split("/")[-1].startswith("test_")
    S["characteristics"] = {
        "repos_loading_multiple_models": f"{sum(d > 1 for d in dm)}/{len(dm)}", "max_models_per_repo": max(dm),
        "unprefixed_ids": f"{sum('/' not in i for i in ids)}/{len(ids)}",
        "unprefixed_sites": f"{sum('/' not in c['model_id'] for c in hub)}/{len(hub)}",
        "indirect_only_repos": sum(all(c["id_class"] != "hub" for c in r["calls"]) for r in repos),
        "pin_pct_test_code": pct(sum(c["revision"] != "absent" for c in calls if is_test(c["file"])), sum(is_test(c["file"]) for c in calls)),
        "pin_pct_non_test_code": pct(sum(c["revision"] != "absent" for c in calls if not is_test(c["file"])), sum(not is_test(c["file"]) for c in calls)),
        "pin_pct_notebooks": pct(sum(c["revision"] != "absent" for c in calls if c["file"].endswith(".ipynb")), sum(c["file"].endswith(".ipynb") for c in calls)),
        "pin_pct_modules": pct(sum(c["revision"] != "absent" for c in calls if c["file"].endswith(".py")), sum(c["file"].endswith(".py") for c in calls)),
    }

    # RQ3: asymmetry
    S["dependency_declarations"] = {p: dict(collections.Counter(str(r["deps"][p]) for r in repos))
                                    for p in ["transformers", "torch", "sentence-transformers", "huggingface-hub"]}
    lib = [r for r in repos if r["lib"]]
    tx = [r for r in repos if r["deps"]["transformers"] == "exact"]
    lk = [r for r in repos if r["lockfile"]]
    S["asymmetry"] = {"lockfile_repos": len(lk), "library_pinned": len(lib),
                      "library_pinned_no_model": f"{sum(not r['pin'] for r in lib)}/{len(lib)}",
                      "transformers_exact_no_model": f"{sum(not r['pin'] for r in tx)}/{len(tx)}",
                      "lockfile_no_model": f"{sum(not r['pin'] for r in lk)}/{len(lk)}"}

    # RQ4: high-risk patterns
    trc = [c for c in calls if c["trust_remote_code"]]
    trc_repos = [r for r in repos if any(c["trust_remote_code"] for c in r["calls"])]
    S["remote_code"] = {"call_sites": len(trc), "repos": len(trc_repos),
                        "revision": dict(collections.Counter(c["revision"] for c in trc)),
                        "repos_with_hash_on_remote_code": sum(any(c["trust_remote_code"] and c["revision"] == "hash"
                                                                  for c in r["calls"]) for r in trc_repos),
                        "indirect_identifier_sites": sum(c["id_class"] == "indirect" for c in trc),
                        "by_interface": dict(collections.Counter(c["api"] for c in trc))}
    dp = [c for c in calls if c["pipeline_default"]]
    S["default_pipelines"] = {"call_sites": len(dp), "repos": len({c["repo"] for c in dp})}

    # RQ5: maturity
    signals = {"ci": lambda r: r["ci"], "tests": lambda r: r["tests"],
               "docker": lambda r: r["docker"], "library_pinned": lambda r: r["lib"]}
    S["fisher"] = {k: fisher(repos, f) for k, f in signals.items()}
    S["fisher_strict"] = {k: fisher(repos, f, "strict") for k, f in signals.items()}
    S["strict_pinning_repos"] = sum(r["strict"] for r in repos)
    n = np.array([sum(r["M"] == m for r in repos) for m in range(5)])
    k = np.array([sum(r["M"] == m and r["pin"] for r in repos) for m in range(5)])
    S["maturity"] = {"n": n.tolist(), "pinning": k.tolist(), "pct": [pct(a, b) for a, b in zip(k, n)],
                     "cochran_armitage_z_p": cochran_armitage(n, k)}
    s1 = [r["stars"] for r in repos if r["pin"]]
    s0 = [r["stars"] for r in repos if not r["pin"]]
    a1 = [r["age"] for r in repos if r["pin"]]
    a0 = [r["age"] for r in repos if not r["pin"]]
    S["popularity"] = {"median_stars_pin": st.median(s1), "median_stars_nopin": st.median(s0),
                       "p": float(mannwhitneyu(s1, s0).pvalue),
                       "median_age_pin": round(st.median(a1), 1), "median_age_nopin": round(st.median(a0), 1),
                       "age_p": round(float(mannwhitneyu(a1, a0).pvalue), 2)}
    df = pd.DataFrame([{"pin": int(r["pin"]), "ci": int(r["ci"]), "tests": int(r["tests"]),
                        "docker": int(r["docker"]), "lib": int(r["lib"]),
                        "log10_stars": np.log10(r["stars"]), "age": r["age"]} for r in repos])
    X = sm.add_constant(df[["ci", "tests", "docker", "lib", "log10_stars", "age"]])
    model = sm.Logit(df["pin"], X).fit(disp=0)
    ci = model.conf_int()
    S["logistic_regression"] = {
        "adjusted_OR": {v: [round(float(np.exp(model.params[v])), 2), round(float(np.exp(ci.loc[v, 0])), 2),
                            round(float(np.exp(ci.loc[v, 1])), 2), round(float(model.pvalues[v]), 3)]
                        for v in X.columns if v != "const"},
        "mcfadden_r2": round(float(model.prsquared), 3), "lr_test_p": float(model.llr_pvalue),
        "corr_ci_tests": round(float(df[["ci", "tests"]].corr().iloc[0, 1]), 2)}

    # Robustness to popularity (Section V-F)
    S["robustness_by_min_stars"] = {t: headline([r for r in repos if r["stars"] >= t]) for t in (20, 100, 500, 1000)}

    json.dump(S, open(out_path("summary.json"), "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
