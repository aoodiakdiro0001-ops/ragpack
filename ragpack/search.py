"""BM25 retrieval over the in-memory index.

BM25Okapi with the standard k1=1.2, b=0.75. Two deliberate choices:

* **IDF floor at 0.07** — the classic Okapi formula returns a negative IDF for
  terms present in more than ~half the corpus, which makes *common* terms
  *subtract* score. For a small corpus of notes that inverts the ranking, so
  the idf is clamped at a small positive floor instead (the same fix Lucene
  applies in BM25Similarity).
* **Path boost** — a query that matches a file name usually means the user
  wants that file, not the paragraph that merely mentions it. Weighting the
  path term counts up (×3) is a small, explainable nudge, and it is exposed
  as `--path-boost` so it can be turned off.
"""

import math
import re

from .tokenize import tokenize

K1 = 1.2
B = 0.75
IDF_FLOOR = 0.07
DEFAULT_PATH_BOOST = 3.0

_SNIP_WINDOW = 90


class Hit(object):
    __slots__ = ("chunk_id", "path", "chunk", "score", "preview", "why")

    def __init__(self, chunk_id, path, chunk, score, preview, why):
        self.chunk_id = chunk_id
        self.path = path
        self.chunk = chunk
        self.score = score
        self.preview = preview
        self.why = why

    def as_dict(self):
        return {
            "id": self.chunk_id,
            "path": self.path,
            "chunk": self.chunk,
            "score": round(self.score, 4),
            "preview": self.preview,
            "why": self.why,
        }


def idf(df, n):
    """Clamped Okapi IDF — never negative, never zero."""
    if n <= 0 or df <= 0:
        return IDF_FLOOR
    v = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    return v if v > IDF_FLOOR else IDF_FLOOR


def score_chunk(query_tokens, chunk, df, n, avgdl, path_boost=DEFAULT_PATH_BOOST,
                path_tokens=None):
    """Return (score, why) — `why` lists the terms that actually contributed."""
    if path_tokens is None:
        path_tokens = _path_tokens(chunk["path"])
    tfs = chunk["tfs"]
    if not tfs:
        return 0.0, []
    dl = chunk["len"] or 1
    norm = K1 * (1.0 - B + B * dl / (avgdl or 1.0))
    score = 0.0
    why = []
    for term in query_tokens:
        f = tfs.get(term)
        if not f:
            continue
        s = idf(df.get(term, 0), n) * f * (K1 + 1.0) / (f + norm)
        if term in path_tokens:
            s *= path_boost
        if s > 0:
            score += s
            why.append((term, round(s, 3)))
    return score, why


def _score_with_path(query_tokens, chunk, df, n, avgdl, path_boost, path_tokens):
    return score_chunk(query_tokens, chunk, df, n, avgdl, path_boost, path_tokens)


def _path_tokens(path):
    return set(tokenize(path))


def run(index, query, *, k=10, path_boost=DEFAULT_PATH_BOOST, min_score=0.0):
    """Rank the index against `query`; returns Hit list best-first."""
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    # de-duplicate but keep order: repeated query terms should not double count
    seen = set()
    ordered = []
    for t in q_tokens:
        if t not in seen:
            seen.add(t)
            ordered.append(t)

    n = len(index["chunks"])
    df = index["df"]
    avgdl = index["avgdl"] or 1.0

    # Path tokens are derived, not stored: keeping them off the chunk dict
    # means `load_index` -> `run` -> `save_index` round-trips byte-identically.
    path_cache = {}
    hits = []
    for chunk in index["chunks"]:
        ppath = chunk["path"]
        cached = path_cache.get(ppath)
        if cached is None:
            cached = path_cache[ppath] = _path_tokens(ppath)
        chunk_path_tokens = cached
        score, why = _score_with_path(ordered, chunk, df, n, avgdl, path_boost,
                                      chunk_path_tokens)
        if score >= min_score and score > 0:
            hits.append(Hit(chunk["id"], chunk["path"], chunk["chunk"],
                            score, _snippet(chunk, ordered), why))

    hits.sort(key=lambda h: (-h.score, h.path, h.chunk))
    return hits[:k]


def _snippet(chunk, q_tokens, window=_SNIP_WINDOW):
    """A short excerpt centred on the first query term occurrence."""
    preview = chunk["preview"]
    low = preview.lower()
    pos = -1
    for t in q_tokens:
        i = low.find(t.lower())
        if i >= 0 and (pos < 0 or i < pos):
            pos = i
    if pos < 0:
        return preview
    start = max(0, pos - window // 2)
    end = min(len(preview), start + window)
    frag = preview[start:end].strip()
    return ("…" if start > 0 else "") + frag + ("…" if end < len(preview) else "")


def highlight(text, q_tokens):
    """Wrap query terms in ANSI bold — used by the text reporter."""
    out = text
    for t in sorted(set(q_tokens), key=len, reverse=True):
        if len(t) < 2:
            continue
        out = re.sub(
            r"(?i)(?<![0-9A-Za-z_])" + re.escape(t) + r"(?![0-9A-Za-z_])",
            "\033[1m\\g<0>\033[0m",
            out,
        )
    return out
