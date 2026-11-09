"""Shared code: data loading, BM25, dense scoring, metrics, cross-validation folds, graph scoring."""
import csv, hashlib, json, math, os, re, sqlite3, sys, time, platform, collections, subprocess
import numpy as np
import scipy.sparse as sp

SEED = 20261105
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results")
RUNS = os.path.join(OUT, "runs")
WORK = os.path.join(OUT, "work")
for d in (OUT, RUNS, WORK): os.makedirs(d, exist_ok=True)

DB22 = os.path.join(DATA, "lawdb_v2_2_frozen.sqlite")
DB21 = os.path.join(DATA, "lawdb_v2_1_frozen.sqlite")
GOLD = os.path.join(DATA, "goldset_v3.csv")
CACHE = os.path.join(DATA, "cache")
PREREG = os.path.join(HERE, "PREREG_v1.0.md")
EXPECT = {DB22: "20eb10a0fd4c1ca12f1ea6aab33a64d042db9d09da8cccc2cd5ff53a9d0a70c2",
          DB21: "b989c8b661e5cb15cf40954340a0c18f01becc648dc529e2e8c18c057505533d",
          GOLD: "4e1c6daed3750e361c51b7e993225394ef943cbef7b816360af2dea0d1ff772e",
          PREREG: "537c7cad9a71155d26862ca22b6bdec092c4e648afb5529cf052be7e846efd54"}
EXPECT_EDGES = {DB22: 4345, DB21: 4206}
MODELS = {"m3": ("BAAI/bge-m3", None), "rerank": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")}
CH = 2000           # dense chunk length (characters)
TOPN = 100          # ranking depth stored per query
DEPTH = 30          # reranking depth (C7)
QMAX, MAXLEN = 256, 512   # reranker input: query tokens, total tokens

T0 = time.time()
_logf = None
def log(*a):
    global _logf
    s = f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    if _logf is None: _logf = open(os.path.join(OUT, "log.txt"), "a", encoding="utf-8")
    _logf.write(s + "\n"); _logf.flush()

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def check_inputs():
    bad = []
    for p, h in EXPECT.items():
        if not os.path.exists(p): bad.append(f"missing: {p}")
        elif sha(p) != h: bad.append(f"hash mismatch: {os.path.basename(p)}")
    if bad: sys.exit("input check failed: " + " / ".join(bad))
    log("input hashes verified (DB v2.2, DB v2.1, gold set v3, PREREG v1.0)")

# ------------------------------------------------------------------ database -> retrieval units
def load_db(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    law = {m: (n, l) for m, n, l in c.execute("SELECT mst, law_name, law_level FROM LAW")}
    def unit_of(pid): return None if not pid else (pid if pid.split(":")[1].endswith("E") else ":".join(pid.split(":")[:2]))
    def artno(k): return (int(k[1:5]), int(k[5:8])) if k.startswith("A") else (int(k[0:4]), int(k[4:6]))
    def loc(u):
        m, k = u.split(":")[:2]
        if k.endswith("E"):
            r = c.execute("SELECT byl_no, byl_branch, title FROM APPENDIX WHERE app_id=?", (u,)).fetchone()
            return f"별표 {int(r[0])}" + (f"의{int(r[1])}" if r[1] and int(r[1]) else "") + f" {r[2] or ''}"
        a, b = artno(k)
        return f"{a // 100}.{a % 100}" if law[m][1] == "NFTC" else f"제{a}조" + (f"의{b}" if b else "")
    units, utext = [], {}
    for pid, m, title in c.execute("SELECT prov_id, mst, title FROM PROVISION WHERE level='조' AND is_deleted=0 ORDER BY mst, seq").fetchall():
        u = unit_of(pid); units.append(u)
        body = " ".join(t for (t,) in c.execute("SELECT text FROM PROVISION WHERE (prov_id=? OR prov_id LIKE ?) AND is_deleted=0 ORDER BY seq, prov_id", (pid, pid + ":%")) if t)
        utext[u] = f"{law[m][0]} {loc(u)} {title or ''} {body}"
    for aid, m in c.execute("SELECT app_id, mst FROM APPENDIX WHERE is_deleted=0 ORDER BY mst, app_id").fetchall():
        units.append(aid)
        body = " ".join(t for (t,) in c.execute("SELECT text FROM APPENDIX_LINE WHERE app_id=? ORDER BY line_no", (aid,)) if t)
        utext[aid] = f"{law[m][0]} {loc(aid)} {re.sub(r'[┃│─┌┐└┘├┤┬┴┼━┏┓┗┛┣┫┳┻╋]+', ' ', body)}"
    for u in units: utext[u] = re.sub(r"<img[^>]*>(</img>)?", " ", re.sub(r"\s+", " ", utext[u])).strip()
    uidx = {u: i for i, u in enumerate(units)}
    # citation edges between units (self-loops excluded, edges from attached tables included)
    cite, deleg = set(), set()
    for frm, tp, ta, ty in c.execute("SELECT from_prov, target_prov_id, target_app_id, ref_type FROM REF_CANDIDATE WHERE resolve_level<>'none'"):
        s = unit_of(frm); t = unit_of(tp) if tp else ta
        if s in uidx and t in uidx and s != t:
            cite.add((s, t))
            if ty == "delegation": deleg.add((s, t))
    for frm, tp, ta in c.execute("SELECT from_app_id, target_prov_id, target_app_id FROM APPENDIX_REF WHERE resolve_level<>'none'"):
        t = unit_of(tp) if tp else ta
        if frm in uidx and t in uidx and frm != t: cite.add((frm, t))
    if len(cite) != EXPECT_EDGES.get(db, len(cite)): sys.exit(f"edge count mismatch: {len(cite)} (expected {EXPECT_EDGES[db]})")
    # dense chunks: 2,000 characters; continuation chunks are prefixed with the law name and location
    chunks, owner, heads = [], [], {}
    for i, u in enumerate(units):
        t = utext[u]; head = t[:t.find(" ", len(law[u.split(':')[0]][0]) + 1) + 1] if len(t) > CH else ""
        heads[u] = t[:t.find(" ", len(law[u.split(':')[0]][0]) + 1) + 1]
        for p in range(0, max(1, len(t)), CH):
            chunks.append((head if p else "") + t[p:p + CH]); owner.append(i)
    return dict(units=units, utext=utext, uidx=uidx, law=law, cite=cite, deleg=deleg, chunks=chunks,
                owner=np.array(owner), heads=heads, NU=len(units))

def load_gold(uidx):
    gold = list(csv.DictReader(open(GOLD, encoding="utf-8-sig")))
    for g in gold: g["G"] = [uidx[u] for u in g["gold_units"].split("; ") if u in uidx]
    gold = [g for g in gold if g["G"]]
    return gold

def gold_matrix(gold, NU):
    GM = np.zeros((len(gold), NU), bool)
    for q, g in enumerate(gold): GM[q, g["G"]] = True
    return GM

# ------------------------------------------------------------------ dense scores from the embedding cache
M3_BACK = "BAAI/bge-m3 (sentence-transformers dense, normalized)"
def cache_path(texts, name, back=M3_BACK):
    key = hashlib.sha256(("\n".join(texts) + back).encode()).hexdigest()[:16]
    return os.path.join(CACHE, f"{name}_{key}.npy")

def maxp(sim, owner, NU):
    """sim: NQ x Nchunk -> NQ x NU, max over the chunks of each unit"""
    order = np.argsort(owner, kind="stable"); o = owner[order]
    starts = np.r_[0, np.flatnonzero(np.diff(o)) + 1]
    S = np.full((sim.shape[0], NU), -np.inf, np.float64)
    S[:, o[starts]] = np.maximum.reduceat(sim[:, order], starts, axis=1)
    return S

def dense_scores(D, gold):
    pc, pq = cache_path(D["chunks"], "units"), cache_path([g["question"] for g in gold], "queries")
    for p in (pc, pq):
        if not os.path.exists(p): sys.exit(f"embedding cache not found: {p}")
    E_ch, E_q = np.load(pc), np.load(pq)
    return E_ch, E_q, maxp(E_q @ E_ch.T, D["owner"], D["NU"])

# ------------------------------------------------------------------ BM25 (Lucene/Anserini formula)
_kiwi = None
KEEP = re.compile(r"^(NN|NR|NP|VV|VA|XR|SL|SN|SH)")
def tok_kiwi(t):
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return [x.form for x in _kiwi.tokenize(t) if KEEP.match(x.tag)]

def jamo(s):
    out = []
    for ch in s:
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7A3:
            x = o - 0xAC00; out += [chr(0x1100 + x // 588), chr(0x1161 + (x % 588) // 28)]
            if x % 28: out.append(chr(0x11A7 + x % 28))
        else: out.append(ch)
    return "".join(out)
def tok_jamo(t):
    import unicodedata
    toks = []
    for w in re.findall(r"\w+", unicodedata.normalize("NFKC", t).lower()):
        j = jamo(w); toks += [j[i:i + 3] for i in range(len(j) - 2)] or [j]
    return toks

class BM25:
    """score(q,d) = sum_t qtf(t) * idf(t) * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl)),  idf = ln(1 + (N - df + 0.5)/(df + 0.5))"""
    def __init__(self, docs_tokens):
        self.vocab = {}
        rows, cols = [], []
        for i, toks in enumerate(docs_tokens):
            for t in toks:
                rows.append(i); cols.append(self.vocab.setdefault(t, len(self.vocab)))
        self.TF = sp.csr_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(len(docs_tokens), len(self.vocab)))
        self.TF.sum_duplicates()
        self.dl = np.asarray(self.TF.sum(1)).ravel(); self.avgdl = self.dl.mean()
        df = np.diff(self.TF.tocsc().indptr); N = len(docs_tokens)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
        self._W = {}
    def W(self, k1, b):
        key = (k1, b)
        if key not in self._W:
            T = self.TF.tocoo(); tf = T.data; K = k1 * (1 - b + b * self.dl[T.row] / self.avgdl)
            self._W[key] = sp.csr_matrix((self.idf[T.col] * tf * (k1 + 1) / (tf + K), (T.row, T.col)), shape=self.TF.shape).T.tocsr()
        return self._W[key]
    def qmat(self, q_tokens):
        rows, cols = [], []
        for i, toks in enumerate(q_tokens):
            for t in toks:
                if t in self.vocab: rows.append(i); cols.append(self.vocab[t])
        Q = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(q_tokens), len(self.vocab))); Q.sum_duplicates(); return Q
    def scores(self, Q, k1, b): return np.asarray((Q @ self.W(k1, b)).todense())

# ------------------------------------------------------------------ ranking and metrics
def topn(S, n=TOPN):
    """ties are broken by unit order (stable sort)"""
    idx = np.argsort(-S, axis=1, kind="stable")[:, :n]
    return idx, np.take_along_axis(S, idx, 1)

def rrf(rank_lists, k=60):
    """rank_lists: [NQ x NU full rankings] -> RRF scores NQ x NU, ranks start at 1"""
    NQ, NU = rank_lists[0].shape
    S = np.zeros((NQ, NU))
    r = np.arange(1, NU + 1, dtype=np.float64)
    for R in rank_lists:
        np.put_along_axis(S, R, np.take_along_axis(S, R, 1) + 1.0 / (k + r[None, :]), 1)
    return S

KEYS = ["R@5", "R@10", "R@20", "R@30", "R@100", "nDCG@10", "Complete@10", "R-Prec", "P@10", "MRR"]
DISC = 1 / np.log2(np.arange(2, 12))
def evaluate(top, GM):
    """top: NQ x >=100 unit indices -> per-query metrics NQ x len(KEYS) (binary relevance)"""
    NQ = top.shape[0]; NG = GM.sum(1)
    h = GM[np.arange(NQ)[:, None], top[:, :100]]
    idcg = np.array([DISC[:min(n, 10)].sum() for n in NG])
    rprec = np.array([h[q, :NG[q]].sum() / NG[q] for q in range(NQ)])
    first = np.where(h.any(1), h.argmax(1) + 1, 0)
    mrr = np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0)
    cols = [h[:, :5].sum(1) / NG, h[:, :10].sum(1) / NG, h[:, :20].sum(1) / NG, h[:, :30].sum(1) / NG, h.sum(1) / NG,
            (h[:, :10] * DISC).sum(1) / idcg, (h[:, :10].sum(1) == NG).astype(float), rprec, h[:, :10].sum(1) / 10, mrr]
    return np.stack(cols, 1)
J = {k: i for i, k in enumerate(KEYS)}

# ------------------------------------------------------------------ grouped cross-validation
def folds(gold):
    freq = collections.Counter(u for g in gold for u in g["G"])
    rep = np.array([sorted(g["G"], key=lambda u: (-freq[u], u))[0] for g in gold])
    ur = np.unique(rep); perm = np.random.default_rng(SEED).permutation(len(ur))
    fo = {u: perm[i] % 5 for i, u in enumerate(ur)}
    FO = np.array([fo[r] for r in rep])
    out = []
    for f in range(5):
        te = FO == f; teu = set(u for q in np.where(te)[0] for u in gold[q]["G"])
        tr = np.array([(not te[q]) and not (set(gold[q]["G"]) & teu) for q in range(len(gold))])
        out.append((te, tr))
    return rep, out

def select(pq_by_cfg, cfgs, tr):
    """highest training Recall@10; ties by nDCG@10, then by candidate order"""
    best = max(range(len(cfgs)), key=lambda i: (round(pq_by_cfg[cfgs[i]][tr, 0].mean(), 12), round(pq_by_cfg[cfgs[i]][tr, 1].mean(), 12), -i))
    return cfgs[best]

def r10_ndcg(top10, GM):
    NQ = top10.shape[0]; NG = GM.sum(1); h = GM[np.arange(NQ)[:, None], top10]
    idcg = np.array([DISC[:min(n, 10)].sum() for n in NG])
    return np.stack([h.sum(1) / NG, (h * DISC).sum(1) / idcg], 1)

def top10(S): return np.argsort(-S, axis=1, kind="stable")[:, :10]

# ------------------------------------------------------------------ run files
def save_run(name, top, score, note="", extra=None):
    np.savez_compressed(os.path.join(RUNS, f"{name}.npz"), top=top.astype(np.int32), score=score.astype(np.float64))
    meta = dict(name=name, note=note, saved=time.strftime("%Y-%m-%d %H:%M:%S")); meta.update(extra or {})
    json.dump(meta, open(os.path.join(RUNS, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
def load_run(name):
    z = np.load(os.path.join(RUNS, f"{name}.npz")); return z["top"], z["score"]
def have_run(name): return os.path.exists(os.path.join(RUNS, f"{name}.npz"))

def env_info():
    info = dict(platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(), python=sys.version.split()[0],
                numpy=np.__version__)
    try:
        import scipy; info["scipy"] = scipy.__version__
    except Exception: pass
    try:
        if os.name == "nt":
            info["cpu_name"] = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=30).stdout.strip()
        else:
            info["cpu_name"] = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "")
    except Exception: info["cpu_name"] = ""
    try:
        import psutil; info["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception: pass
    for m in ("torch", "transformers", "kiwipiepy"):
        try: info[m] = __import__(m).__version__
        except Exception: pass
    return info

def hf_snapshot(key):
    from huggingface_hub import snapshot_download
    name, rev = MODELS[key]
    path = snapshot_download(name, revision=rev) if rev else snapshot_download(name)
    return path, os.path.basename(path.rstrip("/\\"))

# ------------------------------------------------------------------ Rocchio (dense pseudo-relevance feedback)
def unit_chunks(owner):
    uc = collections.defaultdict(list)
    for j, o in enumerate(owner): uc[int(o)].append(j)
    return uc
def rocchio_scores(E_q, E_ch, SIM, ORD, uchunks, owner, NU, k, a):
    """q' = a*q + (1-a)*mean(best chunk embedding of each top-k unit), normalized, then MaxP again"""
    b = 1 - a; Qn = np.zeros_like(E_q)
    for q in range(E_q.shape[0]):
        best = [uchunks[int(u)][int(np.argmax(SIM[q, uchunks[int(u)]]))] for u in ORD[q, :k]]
        v = a * E_q[q] + b * E_ch[best].mean(0); Qn[q] = v / np.linalg.norm(v)
    return maxp(Qn @ E_ch.T, owner, NU)

# ------------------------------------------------------------------ citation-graph scoring (C6, G1, G2)
def graph_struct(DB):
    cite, deleg, uidx, NU = DB["cite"], DB["deleg"], DB["uidx"], DB["NU"]
    rev = {(t, s) for s, t in deleg} - cite; dm = lambda u: u.split(":")[0]; cat = {}
    for s, t in cite: cat[(s, t)] = "cross" if dm(s) != dm(t) else ("app" if t.endswith("E") else "same")
    for e in rev: cat[e] = "rev"
    pairs = sorted(cat); cl = sorted(cite)
    W1 = sp.csr_matrix((np.ones(len(cl)), ([uidx[s] for s, _ in cl], [uidx[t] for _, t in cl])), shape=(NU, NU))
    din = np.asarray(W1.sum(0)).ravel()
    return dict(PI=np.array([uidx[s] for s, _ in pairs]), PJ=np.array([uidx[t] for _, t in pairs]), PC=np.array([cat[e] for e in pairs]),
                P=np.log(din + 1) / np.log(din.max() + 1), n=dict(collections.Counter(cat.values())), NU=NU, _W={})
_Lf = {"ln": lambda d: np.maximum(np.log(d + 1), math.log(2)), "log2": lambda d: np.maximum(np.log2(d + 1), 1.0), "none": lambda d: np.ones_like(d)}
def _graph_W(GS, gs, ga, rho, Lk):
    key = (gs, ga, rho, Lk)
    if key not in GS["_W"]:
        PC = GS["PC"]; w = np.select([PC == "cross", PC == "same", PC == "app", PC == "rev"], [1.0, gs, ga, rho]); keep = w > 0
        W = sp.csr_matrix((w[keep], (GS["PI"][keep], GS["PJ"][keep])), shape=(GS["NU"], GS["NU"]))
        dout = np.asarray(W.sum(1)).ravel(); din = np.asarray(W.sum(0)).ravel()
        GS["_W"][key] = (W, _Lf[Lk](dout), np.where(din > 0, _Lf[Lk](din), 1.0))
    return GS["_W"][key]
def graph_scores(GS, cfg, S0, ORD):
    """S0: NQ x NU dense scores, ORD: S0 ranking. cfg = 8-tuple (propagation) or ('prior', form, beta)"""
    if cfg[0] == "prior":
        _, form, beta = cfg; P = GS["P"][None, :]
        return S0 + beta * P if form == "add" else S0 + beta * P * (1 - S0)
    gs, ga, rho, Lk, sig, comb, k, beta = cfg
    W, Lout, Lin = _graph_W(GS, gs, ga, rho, Lk); NQ = S0.shape[0]
    seeds = ORD[:, :k]; sv = np.take_along_axis(S0, seeds, 1)
    if sig == "rankdisc": sv = sv / np.log2(np.arange(k) + 2)
    Q = sp.csr_matrix(((sv / Lout[seeds]).ravel(), (np.repeat(np.arange(NQ), k), seeds.ravel())), shape=(NQ, GS["NU"]))
    B = np.asarray((Q @ W).todense()) / Lin
    F = S0 + beta * B * (1 - S0) if comb == "C1" else S0 + beta * B
    return np.where(B > 0, F, S0)
def graph_grids():
    import itertools
    GRID = list(itertools.product([0, 0.25, 0.5, 1], [0, 0.5, 1], [0, 1], ["ln", "log2", "none"], ["score", "rankdisc"], ["C1", "C2"], [10, 15, 20], [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]))
    SARD = (1, 1, 0, "ln", "score", "C1", 15, 0.3)
    SARG = [x for x in GRID if x[:6] == SARD[:6]]
    PRIOR = [("prior", f, b) for f in ("add", "res") for b in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3)]
    assert len(GRID) == 6048 and len(SARG) == 21 and len(PRIOR) == 14
    return GRID, SARD, SARG, PRIOR
