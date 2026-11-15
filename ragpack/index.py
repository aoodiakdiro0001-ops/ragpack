"""Corpus walking + index construction.

An index is a plain dict so it serialises to JSON with no magic:

    {"version":1,"root":...,"files":[...],"chunks":[...],
     "df":{term:docfreq},"avgdl":float,"built_at":...}

`chunks` hold `tfs` (term -> raw count) instead of the full token list: on a
10k-file corpus that is the difference between ~40 MB and ~9 MB of JSON, and
raw counts are all BM25 needs.
"""

import hashlib
import json
import os
import re
import time

from .tokenize import tokenize

INDEX_VERSION = 1

# Extensions taken as text. Everything else is skipped unless --all-text.
TEXT_EXTS = frozenset(
    """.md .rst .txt .py .js .ts .jsx .tsx .json .yaml .yml .toml .ini
    .cfg .conf .sh .bash .zsh .fish .c .h .cpp .hpp .cc .java .kt .go .rs
    .rb .php .sql .html .htm .css .scss .vue .svelte .xml .csv .tsv .log
    .gitignore .dockerfile .tf .proto .graphql .el .elisp .clj .ex .exs
    .hs .lua .pl .r .m .swift .scala .lua .nix .diff .patch .env.example""".split()
)

SKIP_DIRS = frozenset(
    ".git .hg .svn node_modules .venv venv __pycache__ .mypy_cache "
    ".pytest_cache .tox dist build target .next .cache .trash "
    "site-packages .rvm .npm .cargo .gradle".split()
)

MAX_BYTES_DEFAULT = 1_048_576  # 1 MiB per file — past this it is data, not text

_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_BULLET = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)


def is_text_file(path):
    name = os.path.basename(path).lower()
    if name in (".gitignore", ".dockerfile", "makefile", "license", "readme"):
        return True
    ext = os.path.splitext(name)[1]
    return ext in TEXT_EXTS


def looks_binary(buf):
    """A file with NUL in its first 8 KB is not text."""
    return b"\x00" in buf[:8192]


def walk(root, *, all_text=False, max_bytes=MAX_BYTES_DEFAULT, respect_ignore=True):
    """Yield (path, absolute) for every readable text file under root."""
    root = os.path.abspath(root)
    ignores = _load_ignore(root) if respect_ignore else []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".git")
        )
        for fn in sorted(filenames):
            if fn.startswith(".") and fn not in (".gitignore", ".env.example"):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            if not all_text and not is_text_file(full):
                continue
            if _ignored(rel, ignores):
                continue
            try:
                size = os.path.getsize(full)
                if size > max_bytes:
                    continue
                with open(full, "rb") as fh:
                    buf = fh.read(max_bytes + 1)
            except (OSError, OverflowError):
                continue
            if len(buf) > max_bytes or looks_binary(buf):
                continue
            yield rel, buf.decode("utf-8", "replace")


def _load_ignore(root):
    """Patterns from .ragpackignore (one glob per line, `#` comments)."""
    path = os.path.join(root, ".ragpackignore")
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.rstrip("/"))
    return out


def _ignored(rel, patterns):
    import fnmatch

    for pat in patterns:
        if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(rel), pat):
            return True
    return False


def chunk_text(text, *, size=1200, overlap=200):
    """Split into overlapping windows that break on paragraph / line edges.

    Fixed-size windows with no overlap lose the sentence a query needs when
    it straddles a boundary; overlap recovers it cheaply. Breaking on a blank
    line first keeps bullets and code blocks from being cut in half.
    """
    if size <= 0:
        return [text]
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
            if cut > size * 0.4:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        # Resume `overlap` chars before the cut, but snap forward to a word
        # boundary — resuming mid-word would split "alpha" into "alp"+"ha" and
        # lose the match for any query that spells the word out.
        nxt = max(end - overlap, start + 1)
        while nxt < end and not text[nxt - 1].isspace():
            nxt += 1
        start = nxt
    return chunks


def _tfs(tokens):
    out = {}
    for t in tokens:
        out[t] = out.get(t, 0) + 1
    return out


def _content_hash(data):
    return hashlib.sha1(data).hexdigest()[:16]


def build_index(root, *, all_text=False, max_bytes=MAX_BYTES_DEFAULT,
                chunk_size=1200, overlap=200, progress=None):
    """Walk `root` and return a serialisable index dict."""
    files = []
    chunks = []
    df = {}
    total_len = 0
    seen_hashes = {}

    for rel, text in walk(root, all_text=all_text, max_bytes=max_bytes):
        digest = _content_hash(text.encode("utf-8", "replace"))
        files.append({"path": rel, "bytes": len(text.encode("utf-8", "replace")),
                      "sha": digest})
        for ci, piece in enumerate(chunk_text(text, size=chunk_size, overlap=overlap)):
            tokens = tokenize(piece)
            if not tokens:
                continue
            tfs = _tfs(tokens)
            # Identical content in two places is one document for scoring
            key = _content_hash(piece.encode("utf-8", "replace"))
            owner = seen_hashes.get(key)
            if owner is None:
                seen_hashes[key] = len(chunks)
                owner = len(chunks)
                chunks.append({
                    "id": len(chunks),
                    "path": rel,
                    "chunk": ci,
                    "chars": len(piece),
                    "tfs": tfs,
                    "len": len(tokens),
                    "preview": piece[:240].replace("\n", " ").strip(),
                })
                for term in tfs:
                    df[term] = df.get(term, 0) + 1
            else:
                chunks[owner]["tfs"] = {k: chunks[owner]["tfs"].get(k, 0) + v
                                        for k, v in tfs.items()}
                chunks[owner]["len"] += len(tokens)
            total_len += len(tokens)
        if progress:
            progress(rel, len(chunks))

    index = {
        "version": INDEX_VERSION,
        "root": os.path.abspath(root),
        "built_at": int(time.time()),
        "files": files,
        "chunks": chunks,
        "df": df,
        "avgdl": (total_len / len(chunks)) if chunks else 0.0,
        "stats": {
            "files": len(files),
            "chunks": len(chunks),
            "terms": len(df),
            "tokens": total_len,
        },
    }
    return index


def save_index(index, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(index, fh, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    return os.path.getsize(path)


def load_index(path):
    with open(path, encoding="utf-8") as fh:
        index = json.load(fh)
    if index.get("version") != INDEX_VERSION:
        raise ValueError(
            "index version %r != %r — re-run `ragpack index`"
            % (index.get("version"), INDEX_VERSION)
        )
    return index
