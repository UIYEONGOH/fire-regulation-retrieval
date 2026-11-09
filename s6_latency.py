"""Step 6: query latency (RQ3). Batch size 1, end to end including query encoding. The first 5 queries are warm-up and excluded.
Latency of a condition = sum of the components it uses, each timed once per query:
  C1 = Kiwi tokenization + BM25 + sort        C2 = jamo 3-gram + BM25 + sort
  C3 = BGE-M3 query encoding + dense MaxP + sort   C4 = C3 + Rocchio second pass
  C5 = C3 + C1 + RRF                          C6/G1/G2 = C3 + graph scoring
  C7 = C5 + reranking time recorded in step 5
Usage: python s6_latency.py"""
import json, os, sys, time, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from nn import M3

check_inputs()
D = load_db(DB22); gold = load_gold(D["uidx"]); NQ = len(gold); NU = D["NU"]; owner = D["owner"]
E_ch, _, _ = dense_scores(D, gold); uch = unit_chunks(owner)
off = {}
t = time.perf_counter(); bk = BM25([tok_kiwi(D["utext"][u]) for u in D["units"]]); bk.W(0.9, 0.4); off["bm25_kiwi_build_s"] = time.perf_counter() - t
t = time.perf_counter(); bj = BM25([tok_jamo(D["utext"][u]) for u in D["units"]]); bj.W(0.9, 0.4); off["bm25_jamo_build_s"] = time.perf_counter() - t
t = time.perf_counter(); GS = graph_struct(D); off["graph_build_s"] = time.perf_counter() - t
mb = lambda M: (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 2 ** 20
off.update(bm25_kiwi_index_mb=mb(bk.W(0.9, 0.4)), bm25_jamo_index_mb=mb(bj.W(0.9, 0.4)), dense_index_mb=E_ch.nbytes / 2 ** 20,
           graph_edges=int(len(GS["PI"])), vocab_kiwi=len(bk.vocab), vocab_jamo=len(bj.vocab))
GRID, SARD, SARG, PRIOR = graph_grids()
sel = json.load(open(os.path.join(OUT, "s1_selection.json"), encoding="utf-8"))
def most_common(name):
    c = collections.Counter(tuple(f["cfg"]) for f in sel[name]["folds"]); return c.most_common(1)[0][0]
CFG = {"C6": SARD, "G1": most_common("G1"), "G2": most_common("G2")}
log("graph configurations for latency", CFG)
m3 = M3(); pt = time.perf_counter
def sort100(S): return np.argsort(-S, kind="stable")[:TOPN]
rows = []
for q in range(NQ):
    text = gold[q]["question"]; r = dict(cid=gold[q]["cid"])
    t = pt(); v = m3.encode([text])["dense"]; r["enc"] = pt() - t
    t = pt(); S = maxp(v @ E_ch.T, owner, NU); top_d = sort100(S[0]); r["dense"] = pt() - t
    t = pt(); ORD = np.argsort(-S, axis=1, kind="stable"); SIM = v @ E_ch.T
    Sr = rocchio_scores(v, E_ch, SIM, ORD, uch, owner, NU, 15, 0.3); sort100(Sr[0]); r["rocchio"] = pt() - t
    t = pt(); Sk = bk.scores(bk.qmat([tok_kiwi(text)]), 0.9, 0.4); sort100(Sk[0]); r["kiwi"] = pt() - t
    t = pt(); Sj = bj.scores(bj.qmat([tok_jamo(text)]), 0.9, 0.4); sort100(Sj[0]); r["jamo"] = pt() - t
    t = pt(); Rk = np.argsort(-Sk, axis=1, kind="stable"); sort100(rrf([ORD, Rk], 60)[0]); r["rrf"] = pt() - t
    for nm, cfg in CFG.items():
        t = pt()
        F = graph_scores(GS, cfg, S, np.argsort(-S, axis=1, kind="stable")); sort100(F[0]); r[nm] = pt() - t
    rows.append(r)
    if q % 50 == 0: log("latency", q, "/", NQ)
rr = {}
p = os.path.join(WORK, "rerank_C7.jsonl")
if os.path.exists(p):
    for l in open(p, encoding="utf-8"): x = json.loads(l); rr[x["cid"]] = x["seconds"]
lat = collections.defaultdict(list)
for r in rows[5:]:
    lat["C1"].append(r["kiwi"]); lat["C2"].append(r["jamo"]); lat["C3"].append(r["enc"] + r["dense"])
    lat["C4"].append(r["enc"] + r["dense"] + r["rocchio"]); c5 = r["enc"] + r["dense"] + r["kiwi"] + r["rrf"]; lat["C5"].append(c5)
    for nm in ("C6", "G1", "G2"): lat[nm].append(r["enc"] + r["dense"] + r[nm])
    if r["cid"] in rr: lat["C7"].append(c5 + rr[r["cid"]])
    lat["(BGE-M3 query encoding only)"].append(r["enc"])
summ = {k: dict(n=len(v), median_ms=round(float(np.median(v)) * 1000, 2), p95_ms=round(float(np.percentile(v, 95)) * 1000, 2)) for k, v in lat.items()}
json.dump(dict(latency=summ, offline={k: (round(v, 3) if isinstance(v, float) else v) for k, v in off.items()}, graph_cfg={k: list(v) for k, v in CFG.items()},
               m3_revision=m3.revision, m3_params=int(m3.params), warmup_excluded=5, env=env_info(), per_query=rows),
          open(os.path.join(OUT, "s6_latency.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in summ.items(): print(f"{k:30s} n={v['n']:3d}  median {v['median_ms']:10.2f} ms  p95 {v['p95_ms']:10.2f} ms")
if "C7" not in summ: print("C7 latency is added when this step is rerun after s5_rerank.py has finished.")
log("step 6 done")
