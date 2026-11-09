"""Step 7: metrics, statistics, subsets, cost tables, and manifest.
Usage: python s7_report.py   -> results/report/ (CSV tables, summary.txt, manifest.json, rankings/*.csv.gz)"""
import csv, glob, gzip, json, os, sys, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

check_inputs()
D = load_db(DB22); gold = load_gold(D["uidx"]); NQ = len(gold); NU = D["NU"]; GM = gold_matrix(gold, NU); units = D["units"]
RD = os.path.join(OUT, "report"); os.makedirs(os.path.join(RD, "rankings"), exist_ok=True)
B = 10000
NOT_RUN = "not run"
MAIN = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]; GRAPH = ["G1", "G2"]
NAMES = dict(C1="BM25 (Kiwi)", C2="BM25 (jamo 3-gram)", C3="BGE-M3 dense", C4="Dense + Rocchio", C5="RRF(C3, C1)", C6="SAR", C7="RRF + rerank (depth 30)",
             G1="Scope-weighted (grouped CV)", G2="In-degree prior (grouped CV)")

# ------------------------------------------------------------------ reranking depth sensitivity (reuses the depth-30 scores)
for what, first in (("C7", "C5"),):
    p = os.path.join(WORK, f"rerank_{what}.jsonl")
    if not (have_run(what) and have_run(first) and os.path.exists(p)): continue
    rec = {}
    for l in open(p, encoding="utf-8"): r = json.loads(l); rec[r["cid"]] = r
    t1, _ = load_run(first)
    for d in (10, 20):
        top = t1.copy(); sc = np.full(top.shape, np.nan)
        for q in range(NQ):
            s = np.array(rec[gold[q]["cid"]]["scores"][:d]); o = np.argsort(-s, kind="stable")
            top[q, :d] = t1[q, :d][o]; sc[q, :d] = s[o]
        save_run(f"S_{what}_d{d}", top, sc, f"sensitivity: rerank only the top {d} of {first} (depth-30 scores reused)")

runs = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(RUNS, "*.npz")))
PQ = {n: evaluate(load_run(n)[0], GM) for n in runs}
log("runs found", runs)

def wcsv(name, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(RD, name), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ table: mean metrics per condition
tab = [dict(condition=n, name=NAMES.get(n, ""), n=NQ, **{k: round(float(PQ[n][:, J[k]].mean()), 4) for k in KEYS}) for n in runs]
wcsv("table_all_conditions.csv", tab)

# ------------------------------------------------------------------ six primary comparisons (query bootstrap + cluster bootstrap, Holm)
freq = collections.Counter(u for g in gold for u in g["G"])
rep = np.array([sorted(g["G"], key=lambda u: (-freq[u], u))[0] for g in gold])
GL = [np.where(rep == r)[0] for r in np.unique(rep)]
BI = np.random.default_rng(SEED).integers(0, NQ, (B, NQ)); CB = np.random.default_rng(SEED).integers(0, len(GL), (B, len(GL)))
gsz = np.array([len(g) for g in GL])
def boot(d, kind):
    if kind == "query": m = d[BI].mean(1)
    else: s = np.array([d[g].sum() for g in GL]); m = s[CB].sum(1) / gsz[CB].sum(1)
    p = min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())); p = max(p, 1 / B)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), p
def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0.0
    for i, j in enumerate(o): run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj
COMP = [("C1", "C3"), ("C5", "C3"), ("C6", "C3"), ("C4", "C3"), ("C7", "C5"), ("C7", "C3")]
comp = []
for kind in ("query", "cluster"):
    rows = []
    for a, b in COMP:
        for met in ("R@10", "nDCG@10"):
            if a in PQ and b in PQ:
                est, lo, hi, p = boot(PQ[a][:, J[met]] - PQ[b][:, J[met]], kind)
                rows.append(dict(bootstrap=kind, comparison=f"{a} − {b}", metric=met, diff=round(est, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4), p=round(p, 5)))
            else: rows.append(dict(bootstrap=kind, comparison=f"{a} − {b}", metric=met, diff=NOT_RUN))
    ok = [r for r in rows if r["diff"] != NOT_RUN]
    if len(ok) == len(rows):
        adj = holm(np.array([r["p"] for r in rows]))
        for r, x in zip(rows, adj): r["p_holm"] = round(float(x), 5); r["holm_reject_0.05"] = bool(x < 0.05)
    else:
        for r in rows: r["p_holm"] = "requires all 12 primary comparisons"
    comp += rows
wcsv("main_comparisons.csv", comp)

# other comparisons, intervals only
extra = []
for a, b in [("G1", "C6"), ("G2", "C3"), ("G1", "C3"), ("C2", "C1"),
             ("T_C1", "C1"), ("T_C2", "C2"), ("T_C4", "C4"), ("T_C6", "C6")]:
    if a in PQ and b in PQ:
        for met in ("R@10", "nDCG@10"):
            est, lo, hi, _ = boot(PQ[a][:, J[met]] - PQ[b][:, J[met]], "query")
            extra.append(dict(comparison=f"{a} − {b}", metric=met, diff=round(est, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
wcsv("descriptive_comparisons.csv", extra)

# ------------------------------------------------------------------ subsets (RQ2)
trunc = {}
p = os.path.join(WORK, "rerank_C7.jsonl")
if os.path.exists(p):
    for l in open(p, encoding="utf-8"): r = json.loads(l); trunc[r["cid"]] = r["truncated"]
SUB = {"number cue": [g["question_has_cue"] == "1" for g in gold], "cross-system": [g["cross_system"] == "1" for g in gold],
       "attached table": ["별표" in g["gold_kinds"] for g in gold], "fire safety code": ["기준" in g["gold_kinds"] for g in gold],
       "one gold": [len(g["G"]) == 1 for g in gold], "two or more gold": [len(g["G"]) > 1 for g in gold],
       "2024 edition": [g["edition"] == "2024" for g in gold], "2025 edition": [g["edition"] == "2025" for g in gold], "2026 edition": [g["edition"] == "2026" for g in gold]}
if "C3" in PQ: SUB["dense miss (no gold in C3 top 10)"] = list(PQ["C3"][:, J["R@10"]] == 0)
if len(trunc) == NQ:
    SUB["query truncated (>256 tokens)"] = [trunc[g["cid"]] for g in gold]; SUB["query not truncated"] = [not trunc[g["cid"]] for g in gold]
subrows = []
for sn, mask in SUB.items():
    mk = np.array(mask); idx = np.where(mk)[0]; n = len(idx)
    if n == 0: continue
    bi = np.random.default_rng(SEED).integers(0, n, (B, n))
    for c in [x for x in MAIN + GRAPH if x in PQ]:
        row = dict(subset=sn, n=n, condition=c, **{m: round(float(PQ[c][idx, J[m]].mean()), 4) for m in ("R@10", "nDCG@10")},
                   n_with_gold_in_top10=int((PQ[c][idx, J["R@10"]] > 0).sum()))
        for base in ("C3", "C5"):
            if base in PQ and c != base:
                d = PQ[c][idx, J["R@10"]] - PQ[base][idx, J["R@10"]]; m = d[bi].mean(1)
                row[f"R@10 − {base}"] = round(float(d.mean()), 4); row[f"95%CI vs {base}"] = f"[{np.percentile(m, 2.5):+.4f}, {np.percentile(m, 97.5):+.4f}]"
        subrows.append(row)
wcsv("subsets.csv", subrows)

# ------------------------------------------------------------------ reranking records and cost
def jl(name):
    p = os.path.join(OUT, name); return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
rer = {}
for what in ("C7",):
    p = os.path.join(WORK, f"rerank_{what}.jsonl")
    if os.path.exists(p):
        R = [json.loads(l) for l in open(p, encoding="utf-8")]
        rer[what] = dict(n=len(R), truncated=int(sum(r["truncated"] for r in R)), truncated_ratio=round(sum(r["truncated"] for r in R) / max(1, len(R)), 4),
                         pairs_mean=round(float(np.mean([r["pairs"] for r in R])), 1), seconds_mean=round(float(np.mean([r["seconds"] for r in R])), 2),
                         seconds_median=round(float(np.median([r["seconds"] for r in R])), 2), reused_pairs_mean=round(float(np.mean([r["reused"] for r in R])), 1),
                         total_hours=round(sum(r["seconds"] for r in R) / 3600, 2))
lat = jl("s6_latency.json")
cost = []
if lat:
    LABEL = {"(BGE-M3 질의 인코딩만)": "(BGE-M3 query encoding only)"}
    for k, v in lat["latency"].items(): cost.append(dict(condition=LABEL.get(k, k), name=NAMES.get(k, ""), **v))
wcsv("latency.csv", cost)

# ------------------------------------------------------------------ rankings (csv.gz)
for n in runs:
    top, sc = load_run(n)
    with gzip.open(os.path.join(RD, "rankings", f"{n}.csv.gz"), "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["cid", "rank", "unit_id", "score", "is_gold"])
        for q in range(NQ):
            for r in range(top.shape[1]):
                u = int(top[q, r]); w.writerow([gold[q]["cid"], r + 1, units[u], "" if np.isnan(sc[q, r]) else f"{sc[q, r]:.6f}", int(GM[q, u])])
# per-query metrics
with gzip.open(os.path.join(RD, "per_query_metrics.csv.gz"), "wt", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["cid", "condition"] + KEYS)
    for n in runs:
        for q in range(NQ): w.writerow([gold[q]["cid"], n] + [f"{x:.4f}" for x in PQ[n][q]])

# ------------------------------------------------------------------ manifest
scripts = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(HERE, "*.py")))}
IMPL = os.path.join(HERE, "PREREG_v1.0_implementation_notes.md")
man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), prereg_sha256=sha(PREREG), impl_spec_sha256=(sha(IMPL) if os.path.exists(IMPL) else "missing"), db_v22_sha256=sha(DB22), db_v21_sha256=sha(DB21), gold_sha256=sha(GOLD),
           query_text_sha256=hashlib.sha256("\n".join(g["question"] for g in gold).encode()).hexdigest(),
           cache={os.path.basename(p): sha(p) for p in glob.glob(os.path.join(CACHE, "*.npy"))}, scripts=scripts, env=env_info(),
           runs={n: json.load(open(os.path.join(RUNS, n + ".json"), encoding="utf-8")).get("note", "") for n in runs},
           rerank=rer, latency_offline=(lat["offline"] if lat else NOT_RUN), selection=jl("s1_selection.json"))
json.dump(man, open(os.path.join(RD, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

# ------------------------------------------------------------------ summary
L = [f"report {man['created']}  | queries {NQ}  | PREREG sha256 {man['prereg_sha256'][:12]}...", ""]
L.append("Conditions (primary metrics R@10, nDCG@10; secondary metrics)")
L.append(f"{'cond':5s} {'name':30s} {'R@10':>6s} {'nDCG@10':>8s} {'R@5':>6s} {'R@20':>6s} {'R@30':>6s} {'R@100':>6s} {'Comp@10':>8s} {'R-Prec':>7s} {'P@10':>6s} {'MRR':>6s}")
for c in MAIN + GRAPH:
    if c in PQ:
        M = PQ[c].mean(0)
        L.append(f"{c:5s} {NAMES[c]:30s} {M[J['R@10']]:6.3f} {M[J['nDCG@10']]:8.3f} {M[J['R@5']]:6.3f} {M[J['R@20']]:6.3f} {M[J['R@30']]:6.3f} {M[J['R@100']]:6.3f} {M[J['Complete@10']]:8.3f} {M[J['R-Prec']]:7.3f} {M[J['P@10']]:6.3f} {M[J['MRR']]:6.3f}")
    else: L.append(f"{c:5s} {NAMES[c]:30s} {NOT_RUN}")
L += ["", "Primary comparisons (query bootstrap, Holm)"]
for r in comp:
    if r["bootstrap"] == "query":
        L.append(f"{r['comparison']:8s} {r['metric']:8s} " + (f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  p={r['p']:.4f}  Holm p={r.get('p_holm')}" if r["diff"] != NOT_RUN else NOT_RUN))
if rer: L += ["", "Reranking"] + [f"{k}: {v}" for k, v in rer.items()]
if cost: L += ["", "Latency (batch 1, median / p95, ms)"] + [f"{r['condition']:30s} {r['median_ms']:10.2f} / {r['p95_ms']:10.2f}" for r in cost]
open(os.path.join(RD, "summary.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L)); log("step 7 done ->", RD)
