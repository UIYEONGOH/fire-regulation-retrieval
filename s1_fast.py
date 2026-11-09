"""Step 1: C1-C6, G1, G2, equal-budget tuning, and sensitivity runs that need no model download.
Dense scores come from the BGE-M3 embedding cache in data/cache.
Usage:   python s1_fast.py
Output:  results/runs/*.npz (top-100 ranking and scores per query), results/s1_selection.json"""
import itertools, json, math, os, sys, time, collections
import numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

check_inputs()
D = load_db(DB22); D21 = load_db(DB21)
assert D["units"] == D21["units"] and D["chunks"] == D21["chunks"], "retrieval units differ between v2.1 and v2.2"
NU = D["NU"]; gold = load_gold(D["uidx"]); NQ = len(gold); GM = gold_matrix(gold, NU)
log("units", NU, "chunks", len(D["chunks"]), "queries", NQ, "edges v2.2", len(D["cite"]), "v2.1", len(D21["cite"]))
rep, FOLDS = folds(gold)
log("groups (representative gold unit)", len(np.unique(rep)), "| test/train queries per fold", [(int(te.sum()), int(tr.sum())) for te, tr in FOLDS])
SEL = {}

def cv_assemble(name, cfgs, pq, scorer, note):
    """5-fold grouped CV: select a configuration on the training queries, apply it to the test queries. scorer(cfg) -> NQ x NU scores"""
    top = np.zeros((NQ, TOPN), np.int64); sc = np.zeros((NQ, TOPN)); chosen = []
    cache = {}
    for f, (te, tr) in enumerate(FOLDS):
        x = select(pq, cfgs, tr); chosen.append(dict(fold=f, n_test=int(te.sum()), n_train=int(tr.sum()), cfg=list(x) if isinstance(x, tuple) else x))
        if x not in cache: cache[x] = topn(scorer(x))
        top[te], sc[te] = cache[x][0][te], cache[x][1][te]
    SEL[name] = dict(n_candidates=len(cfgs), folds=chosen)
    save_run(name, top, sc, note, dict(n_candidates=len(cfgs), folds=chosen))
    log(name, "candidates", len(cfgs), "selected", [c["cfg"] for c in chosen])

# ------------------------------------------------------------------ C3 dense
E_ch, E_q, S0 = dense_scores(D, gold)
ORD = np.argsort(-S0, axis=1, kind="stable")
save_run("C3", *topn(S0), "BGE-M3 dense, 2,000-char chunks + MaxP (cache)")
log("C3 done")

# ------------------------------------------------------------------ C1, C2 BM25
t = time.time(); docs_k = [tok_kiwi(D["utext"][u]) for u in D["units"]]; q_k = [tok_kiwi(g["question"]) for g in gold]
bk = BM25(docs_k); Qk = bk.qmat(q_k); log("Kiwi tokenization", round(time.time() - t, 1), "s, vocabulary", len(bk.vocab))
docs_j = [tok_jamo(D["utext"][u]) for u in D["units"]]; q_j = [tok_jamo(g["question"]) for g in gold]
bj = BM25(docs_j); Qj = bj.qmat(q_j); log("jamo 3-gram vocabulary", len(bj.vocab))
S1 = bk.scores(Qk, 0.9, 0.4); S2 = bj.scores(Qj, 0.9, 0.4)
save_run("C1", *topn(S1), "BM25 Kiwi k1=0.9 b=0.4"); save_run("C2", *topn(S2), "BM25 jamo 3-gram k1=0.9 b=0.4")
save_run("S_C1_k1.5_b0.75", *topn(bk.scores(Qk, 1.5, 0.75)), "sensitivity"); save_run("S_C2_k1.5_b0.75", *topn(bj.scores(Qj, 1.5, 0.75)), "sensitivity")
BMG = list(itertools.product([0.6, 0.9, 1.2, 1.5, 1.8], [0.3, 0.4, 0.6, 0.75, 0.9]))
for nm, bm, Q in (("T_C1", bk, Qk), ("T_C2", bj, Qj)):
    pq = {x: r10_ndcg(top10(bm.scores(Q, *x)), GM) for x in BMG}
    cv_assemble(nm, BMG, pq, lambda x, bm=bm, Q=Q: bm.scores(Q, *x), "equal-budget tuning (grouped CV)")
log("C1, C2 done")

# ------------------------------------------------------------------ C5 RRF
R1 = np.argsort(-S1, axis=1, kind="stable")
for k, nm in ((60, "C5"), (10, "S_C5_k10"), (100, "S_C5_k100")):
    save_run(nm, *topn(rrf([ORD, R1], k)), f"RRF(C3, C1) k={k}")
log("C5 done")

# ------------------------------------------------------------------ C4 Rocchio (dense PRF)
uchunks = unit_chunks(D['owner'])
SIM = E_q @ E_ch.T
rocchio = lambda k, a: rocchio_scores(E_q, E_ch, SIM, ORD, uchunks, D['owner'], NU, k, a)
save_run("C4", *topn(rocchio(15, 0.3)), "Rocchio k=15 alpha=0.3 beta=0.7")
RG = list(itertools.product([5, 10, 15], [0.3, 0.5, 0.7]))
RS = {x: rocchio(*x) for x in RG}
cv_assemble("T_C4", RG, {x: r10_ndcg(top10(RS[x]), GM) for x in RG}, lambda x: RS[x], "equal-budget tuning (grouped CV)")
log("C4 done")

GRID, SARD, SARG, PRIOR = graph_grids()
_gsc = graph_scores
graph_scores = lambda GS, cfg: _gsc(GS, cfg, S0, ORD)

for tag, DB in (("", D), ("_v21", D21)):
    GS = graph_struct(DB); log("graph" + (tag or "_v22"), dict(GS["n"]))
    sfx = "" if not tag else "_v21"
    save_run("C6" if not tag else "S_C6_v21", *topn(graph_scores(GS, SARD)), f"SAR k=15 beta=0.3, graph{tag or ' v2.2'}")
    t = time.time(); pq = {}
    for n, x in enumerate(GRID + PRIOR):
        pq[x] = r10_ndcg(top10(graph_scores(GS, x)), GM)
        if n % 1000 == 0: log(f"graph candidates{tag}", n, "/", len(GRID) + len(PRIOR), f"({time.time() - t:.0f}s)")
    cv_assemble("G1" + sfx if not tag else "S_G1_v21", GRID, pq, lambda x: graph_scores(GS, x), "scope-weighted, 6,048 candidates, grouped CV")
    cv_assemble("G2" if not tag else "S_G2_v21", PRIOR, pq, lambda x: graph_scores(GS, x), "in-degree prior, 14 candidates, grouped CV")
    if not tag: cv_assemble("T_C6", SARG, pq, lambda x: graph_scores(GS, x), "SAR k, beta, 21 candidates, grouped CV")
json.dump(SEL, open(os.path.join(OUT, "s1_selection.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print(f"\n=== step 1 summary ({NQ} queries) ===")
for nm in ("C1", "C2", "C3", "C4", "C5", "C6", "G1", "G2"):
    M = evaluate(load_run(nm)[0], GM); print(f"{nm:4s} R@10 {M[:, J['R@10']].mean():.3f}  nDCG@10 {M[:, J['nDCG@10']].mean():.3f}  R@30 {M[:, J['R@30']].mean():.3f}  R@100 {M[:, J['R@100']].mean():.3f}")
log("step 1 done")
