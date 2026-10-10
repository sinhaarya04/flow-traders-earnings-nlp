"""
Flow Traders earnings-language experiment.
Question: How predictable is a Flow Traders earnings letter?

Pipeline (deliberately simple + explainable):
  1. Content-word frequency ranking  -> the "top 5 words" reveal
  2. N-gram model                     -> a generated predicted opener
  3. Novelty score per quarter        -> information-decay / recycled-language finding
  4. Backtest (train on 7, predict 8) -> proof the language is forecastable
  5. NTI vs VIX correlation           -> the punchline: language is predictable, P&L isn't
"""
import re, json, random, collections, pathlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

random.seed(7)
ROOT = pathlib.Path(__file__).parent
COMM = ROOT / "data" / "commentary"
QUARTERS = ["2023Q4","2024Q1","2024Q2","2024Q4","2025Q1","2025Q2","2025Q3","2025Q4"]

# ---- load ----
docs = {q: (COMM / f"{q}.txt").read_text(encoding="utf-8") for q in QUARTERS}

# ---- tokenisation ----
def tokens(text, keep_stop=True):
    text = text.lower().replace("\u2019", "'").replace("\u201c", " ").replace("\u201d", " ")
    toks = re.findall(r"[a-z][a-z'\-]+", text)
    return toks

# generic english + release boilerplate we don't want in the "interesting words" ranking
STOP = set("""
the a an and or of to in on for with as at by from is are was were be been being this that these those
we our us it its their his her they he she you your i my me will would can could should may might must
have has had do does did not no yes so than then also more most much many few some any all both each
which who whom whose what when where why how there here we're we've our we'll into over under out up down
about across during within per vs given following able continue continued continues year quarter quarters
first second third fourth full flow traders traders' company company's results result period compared
increase increased decrease decreased due strong solid another set posted recorded closed saw see seen
across half back h1 h2 fy q1 q2 q3 q4 1q 2q 3q 4q new well
""".split())

def content_tokens(text):
    return [t for t in tokens(text) if t not in STOP and len(t) > 2]

# ---- 1. content-word frequency + document frequency ----
total = collections.Counter()
docfreq = collections.Counter()
for q in QUARTERS:
    ct = content_tokens(docs[q])
    total.update(ct)
    docfreq.update(set(ct))

ranked = sorted(total.items(), key=lambda kv: (docfreq[kv[0]], kv[1]), reverse=True)
top_words = [
    {"word": w, "total": c, "in_quarters": docfreq[w]}
    for w, c in ranked[:15]
]

# ---- 2. n-gram model (trigram w/ bigram backoff) over full tokens ----
def build_ngrams(corpus_docs):
    bi = collections.defaultdict(collections.Counter)
    tri = collections.defaultdict(collections.Counter)
    for text in corpus_docs:
        t = tokens(text)
        for i in range(len(t) - 1):
            bi[(t[i],)][t[i+1]] += 1
        for i in range(len(t) - 2):
            tri[(t[i], t[i+1])][t[i+2]] += 1
    return bi, tri

def generate(bi, tri, seed=("flow", "traders"), n=18):
    out = list(seed)
    for _ in range(n):
        ctx2 = (out[-2], out[-1])
        nxt = tri.get(ctx2)
        if not nxt:
            nxt = bi.get((out[-1],))
        if not nxt:
            break
        # greedy-ish: pick the most common continuation (deterministic, explainable)
        out.append(nxt.most_common(1)[0][0])
    return " ".join(out)

bi_all, tri_all = build_ngrams(docs.values())
predicted_opener_next = generate(bi_all, tri_all, seed=("flow", "traders"), n=20)

# ---- 3a. vocabulary recycling: % of each letter's content words seen in a PRIOR letter ----
#      this measures theme/word reuse (interpretable), not exact phrasing.
seen = set()
recycled = {}
for i, q in enumerate(QUARTERS):
    ct = set(content_tokens(docs[q]))
    if i == 0:
        recycled[q] = None  # baseline, nothing prior
    else:
        overlap = len(ct & seen) / len(ct)
        recycled[q] = round(overlap * 100, 1)
    seen |= ct
recycled_avg = round(np.mean([v for v in recycled.values() if v is not None]), 1)
novelty = {q: (None if v is None else round(100 - v, 1)) for q, v in recycled.items()}

# ---- 3b. exact-phrasing novelty (TF-IDF bigram cosine) -- secondary, shows sentences ARE rewritten
vec = TfidfVectorizer(stop_words="english", ngram_range=(2, 2), min_df=1)
X = vec.fit_transform([docs[q] for q in QUARTERS])
phrase_reuse = {}
for i, q in enumerate(QUARTERS):
    if i == 0:
        phrase_reuse[q] = None
        continue
    sims = cosine_similarity(X[i], X[:i]).ravel()
    phrase_reuse[q] = round(float(sims.max()) * 100, 1)  # max bigram-cosine to any prior
phrase_reuse_avg = round(np.mean([v for v in phrase_reuse.values() if v is not None]), 1)

# ---- 4. backtest: train on first 7, predict the 8th (2025Q4) ----
train_q = QUARTERS[:-1]           # 2023Q4 .. 2025Q3
test_q = QUARTERS[-1]             # 2025Q4
bi_tr, tri_tr = build_ngrams([docs[q] for q in train_q])
predicted_opener_bt = generate(bi_tr, tri_tr, seed=("flow", "traders"), n=20)
actual_opener_bt = " ".join(tokens(docs[test_q])[:22])

# word-level predictability: of the top-K content words learned from training,
# how many actually show up in the held-out 2025Q4 letter?
tr_total = collections.Counter()
for q in train_q:
    tr_total.update(content_tokens(docs[q]))
topK = [w for w, _ in tr_total.most_common(15)]
actual_set = set(content_tokens(docs[test_q]))
hits = [w for w in topK if w in actual_set]
predictability_pct = round(100 * len(hits) / len(topK), 1)

# ---- 5. NTI vs VIX ----
fin = pd.read_csv(ROOT / "data" / "financials.csv")
corr = round(float(np.corrcoef(fin["nti_eur_m"], fin["avg_vix"])[0, 1]), 2)

# ---- assemble results ----
results = {
    "n_quarters": len(QUARTERS),
    "quarters": QUARTERS,
    "avg_words_per_letter": round(np.mean([len(docs[q].split()) for q in QUARTERS]), 0),
    "top_words": top_words,
    "top5_reveal": [w["word"] for w in top_words[:5]],
    "predicted_opener_next_quarter": predicted_opener_next,
    "novelty_pct_by_quarter": novelty,
    "recycled_pct_by_quarter": recycled,
    "avg_recycled_pct": recycled_avg,
    "phrase_reuse_pct_by_quarter": phrase_reuse,
    "avg_phrase_reuse_pct": phrase_reuse_avg,
    "backtest": {
        "trained_on": train_q,
        "held_out": test_q,
        "predicted_opener": predicted_opener_bt,
        "actual_opener": actual_opener_bt,
        "top_words_predictability_pct": predictability_pct,
        "hits": hits,
    },
    "nti_vix_correlation": corr,
}

(ROOT / "results.json").write_text(json.dumps(results, indent=2))

# ---- pretty print ----
print("=" * 66)
print("FLOW TRADERS  ::  EARNINGS-LANGUAGE PREDICTABILITY EXPERIMENT")
print("=" * 66)
print(f"Letters analysed : {len(QUARTERS)}  ({QUARTERS[0]} -> {QUARTERS[-1]})")
print(f"Avg length       : {results['avg_words_per_letter']:.0f} words")
print(f"\n-- TOP 15 RECURRING CONTENT WORDS (word : total : appears in X/8) --")
for w in top_words:
    print(f"   {w['word']:<18} {w['total']:>3}   {w['in_quarters']}/8")
print(f"\n>> THE 5-WORD REVEAL: {results['top5_reveal']}")
print(f"\n-- VOCABULARY RECYCLING (% of letter's words seen in a prior letter) --")
for q, v in recycled.items():
    print(f"   {q}: {'n/a (baseline)' if v is None else str(v)+'% recycled'}")
print(f"   => vocabulary is on average {recycled_avg}% RECYCLED across letters")
print(f"   => but exact phrasing reuse (bigram cosine) is only {phrase_reuse_avg}%")
print(f"      i.e. SAME THEMES, freshly REWRITTEN sentences")
print(f"\n-- BACKTEST: trained on 7 quarters, predicted {test_q} --")
print(f"   Predicted opener: {predicted_opener_bt}")
print(f"   Actual opener   : {actual_opener_bt}")
print(f"   Top-word predictability: {predictability_pct}%  (hits: {hits})")
print(f"\n-- PREDICTED OPENER FOR NEXT QUARTER (2026Q1) --")
print(f"   {predicted_opener_next}")
print(f"\n-- PUNCHLINE: NTI vs Average VIX correlation = {corr} --")
print("=" * 66)
