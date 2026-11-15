"""Reporters: text (EN/AR), JSON, Markdown.

Every reporter takes the same neutral result object so switching `--report
x.json` never changes what is computed — only how it is written.
"""

import json

LANG = ("en", "ar")

T = {
    "en": {
        "title": "ragpack — retrieval",
        "index_line": "Index: {root}",
        "stats": "Files: {files} · Chunks: {chunks} · Terms: {terms} · Tokens: {tokens}",
        "query": "Query: {query}",
        "results": "Top {count} of {total} matching chunks",
        "none": "No chunk matched this query.",
        "why": "why",
        "time": "Index built {when}s ago · {path}",
        "empty_index": "Index has no chunks — nothing to search.",
    },
    "ar": {
        "title": "ragpack — الاسترجاع",
        "index_line": "الفهرس: {root}",
        "stats": "ملفات: {files} · مقاطع: {chunks} · مصطلحات: {terms} · رموز: {tokens}",
        "query": "الاستعلام: {query}",
        "results": "أفضل {count} من أصل {total} مطابق",
        "none": "لا يوجد مقطع مطابق لهذا الاستعلام.",
        "why": "لماذا",
        "time": "بُني الفهرس قبل {when} ثانية · {path}",
        "empty_index": "الفهرس بلا مقاطع — لا شيء يُبحث فيه.",
    },
}


def _t(lang, key, **kw):
    table = T.get(lang, T["en"])
    return table.get(key, T["en"][key]).format(**kw)


def fmt_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%ds" % (seconds // 60, seconds % 60)
    if seconds < 86400:
        return "%dh%dm" % (seconds // 3600, (seconds % 3600) // 60)
    return "%dd%dh" % (seconds // 86400, (seconds % 86400) // 3600)


def text_report(result, *, lang="en", color=False, width=78):
    """The default terminal report. `color` is off unless stdout is a tty."""
    import time

    idx = result["index"]
    st = idx["stats"]
    out = []
    out.append(_t(lang, "title"))
    out.append("=" * min(width, 60))
    out.append(_t(lang, "index_line", root=idx["root"]))
    out.append(_t(lang, "stats", files=st["files"], chunks=st["chunks"],
                  terms=st["terms"], tokens=st["tokens"]))
    out.append("")
    out.append(_t(lang, "query", query=result["query"]))

    hits = result["hits"]
    if not hits:
        out.append(_t(lang, "none"))
        return "\n".join(out) + "\n"

    out.append(_t(lang, "results", count=len(hits), total=result["matched"]))
    out.append("")
    for i, h in enumerate(hits, 1):
        out.append("%2d. [%5.3f] %s" % (i, h["score"], h["path"]))
        if h.get("chunk"):
            out.append("     %s#%s" % ("", h["chunk"]))
        preview = h.get("preview") or ""
        if color and result.get("q_tokens"):
            from .search import highlight

            preview = highlight(preview, result["q_tokens"])
        out.append("     " + preview[:width])
        if h.get("why"):
            why = ", ".join("%s=%.2f" % (t, s) for t, s in h["why"][:5])
            out.append("     %s: %s" % (_t(lang, "why"), why))
        out.append("")
    out.append(_t(lang, "time", when=fmt_duration(time.time() - idx["built_at"]),
                  path=result.get("index_path", "index.json")))
    return "\n".join(out) + "\n"


def json_report(result, *, lang="en"):
    return json.dumps(
        {
            "query": result["query"],
            "matched": result["matched"],
            "returned": len(result["hits"]),
            "index": {
                "root": result["index"]["root"],
                "stats": result["index"]["stats"],
                "built_at": result["index"]["built_at"],
            },
            "results": result["hits"],
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def md_report(result, *, lang="en"):
    out = ["# ragpack — %s" % result["query"], ""]
    st = result["index"]["stats"]
    out.append("`files=%d · chunks=%d · terms=%d`" %
               (st["files"], st["chunks"], st["terms"]))
    out.append("")
    if not result["hits"]:
        out.append("_%s_" % _t(lang, "none"))
        return "\n".join(out) + "\n"
    out.append("| # | score | path | chunk | excerpt |")
    out.append("|---|------:|------|------:|---------|")
    for i, h in enumerate(result["hits"], 1):
        preview = (h.get("preview") or "").replace("|", "\\|")
        out.append("| %d | %.3f | `%s` | %s | %s |" %
                   (i, h["score"], h["path"], h.get("chunk", ""), preview[:120]))
    return "\n".join(out) + "\n"


def write_report(result, path, *, lang="en", color=False):
    """Dispatch on the file extension; returns (path, bytes)."""
    low = path.lower()
    if low.endswith(".json"):
        data = json_report(result, lang=lang)
    elif low.endswith((".md", ".markdown")):
        data = md_report(result, lang=lang)
    else:
        data = text_report(result, lang=lang, color=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(data)
    return path, len(data.encode("utf-8"))
