"""Step 5: C7, reranking the top 30 of C5 with bge-reranker-v2-m3 at a fixed revision.
Input rule: the first 256 query tokens; the unit is split into chunks that fit the remaining length (512 - query - special tokens)
and the unit score is the maximum over its chunks. Progress is saved per query, so an interrupted run resumes where it stopped.
Usage: python s5_rerank.py"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--limit", type=int, default=0, help="smoke test: number of queries"); args = ap.parse_args()
from nn import Reranker

check_inputs()
D = load_db(DB22); gold = load_gold(D["uidx"]); NQ = len(gold); units = D["units"]
FIRST = "C5"
if not have_run(FIRST): sys.exit(f"{FIRST} ranking not found; run s1_fast.py first")
top1, sc1 = load_run(FIRST)
rr = Reranker(args.batch); log("reranker", rr.path, "revision", rr.revision, "special tokens", rr.SPECIAL, "parameters", rr.params)
if rr.revision != MODELS["rerank"][1]: sys.exit("reranker revision differs from the preregistered value")
LOG = os.path.join(WORK, "rerank_C7.jsonl")
done = {}
if os.path.exists(LOG):
    for l in open(LOG, encoding="utf-8"):
        r = json.loads(l); done[r["cid"]] = r
utok = {}
def unit_ids(u):
    if u not in utok: utok[u] = rr.ids(D["utext"][u])
    return utok[u]
log(f"C7: {len(done)}/{NQ} queries already done")
qs = range(NQ) if not args.limit else range(min(args.limit, NQ))
with open(LOG, "a", encoding="utf-8") as f:
    for n, q in enumerate(qs):
        cid = gold[q]["cid"]
        if cid in done: continue
        t0 = time.perf_counter()
        qi_full = rr.ids(gold[q]["question"]); qi = qi_full[:QMAX]; budget = MAXLEN - len(qi) - rr.SPECIAL
        cand = [units[int(u)] for u in top1[q, :DEPTH]]
        pairs, own = [], []
        for k, u in enumerate(cand):
            pi = unit_ids(u)
            for s in range(0, max(1, len(pi)), budget): pairs.append((qi, pi[s:s + budget])); own.append(k)
        sc = rr.score_pairs(pairs) if pairs else []
        best = {}
        for k, v in zip(own, sc): best[cand[k]] = max(best.get(cand[k], -1e9), v)
        scores = [best[u] for u in cand]
        rec = dict(cid=cid, query_tokens=len(qi_full), truncated=len(qi_full) > QMAX, passage_budget=budget, pairs=len(pairs),
                   reused=0, seconds=round(time.perf_counter() - t0, 3), units=cand, scores=scores)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush(); done[cid] = rec
        if len(done) % 5 == 0 or len(done) == NQ:
            secs = [r["seconds"] for r in done.values()]
            log(f"C7 {len(done)}/{NQ}  mean {np.mean(secs):.1f} s/query, remaining {(NQ - len(done)) * np.mean(secs) / 3600:.1f} h")
if len(done) < NQ: log(f"stopped at {len(done)}/{NQ}; rerun to continue"); sys.exit(0)
# final ranking: reranked top 30 (score descending, ties by first-stage order) followed by C5 ranks 31-100
top = np.zeros((NQ, TOPN), np.int64); score = np.full((NQ, TOPN), np.nan)
for q in range(NQ):
    r = done[gold[q]["cid"]]; s = np.array(r["scores"]); o = np.argsort(-s, kind="stable")
    top[q, :DEPTH] = top1[q, :DEPTH][o]; score[q, :DEPTH] = s[o]; top[q, DEPTH:] = top1[q, DEPTH:TOPN]
save_run("C7", top, score, f"C5 top {DEPTH} -> bge-reranker-v2-m3 (query {QMAX} tokens, 512-token chunks, MaxP); ranks below {DEPTH} keep the C5 order",
         dict(revision=rr.revision, params=int(rr.params), batch=args.batch, env=env_info()))
log("C7 done")
