"""Command line interface.

    ragpack index <dir> [-o index.json] [--all-text] [--max-bytes N]
    ragpack search <query> [-i index.json] [-k 10] [--json|--report f]
    ragpack stats  [-i index.json]
    ragpack -V | --version

Exit codes: 0 = ok / results found, 1 = no results (for `search --ci`),
2 = usage or IO error. `search --ci` also fails on a missing index so CI
never passes green against a stale or absent build.
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

from . import __version__
from .index import MAX_BYTES_DEFAULT, build_index, load_index, save_index
from .report import LANG, json_report, text_report, write_report
from .search import DEFAULT_PATH_BOOST, run

DEFAULT_INDEX = "ragpack-index.json"
EXIT_OK = 0
EXIT_NO_RESULTS = 1
EXIT_ERROR = 2


def _err(msg):
    sys.stderr.write("ragpack: %s\n" % msg)


def build_parser():
    p = argparse.ArgumentParser(
        prog="ragpack",
        description="Dependency-free local retrieval (BM25) for AI agents.",
    )
    p.add_argument("-V", "--version", action="version",
                   version="ragpack %s" % __version__)
    sub = p.add_subparsers(dest="cmd")

    pi = sub.add_parser("index", help="walk a folder and build the index")
    pi.add_argument("dir", help="root directory to index")
    pi.add_argument("-o", "--output", default=DEFAULT_INDEX,
                    help="index file to write (default: %s)" % DEFAULT_INDEX)
    pi.add_argument("--all-text", action="store_true",
                    help="index every file, not only known text extensions")
    pi.add_argument("--max-bytes", type=int, default=MAX_BYTES_DEFAULT,
                    help="skip files larger than this (default: 1 MiB)")
    pi.add_argument("--chunk-size", type=int, default=1200,
                    help="target characters per chunk (default: 1200)")
    pi.add_argument("--overlap", type=int, default=200,
                    help="characters of overlap between chunks (default: 200)")
    pi.add_argument("-q", "--quiet", action="store_true")

    ps = sub.add_parser("search", help="query an existing index")
    ps.add_argument("query")
    ps.add_argument("-i", "--index", default=DEFAULT_INDEX)
    ps.add_argument("-k", type=int, default=10, help="results to return")
    ps.add_argument("--min-score", type=float, default=0.0)
    ps.add_argument("--path-boost", type=float, default=DEFAULT_PATH_BOOST,
                    help="multiplier for terms found in the file path "
                         "(default: %.1f, 0 disables)" % DEFAULT_PATH_BOOST)
    ps.add_argument("--json", action="store_true", help="print JSON to stdout")
    ps.add_argument("--report", metavar="FILE",
                    help="also write text/.json/.md to FILE")
    ps.add_argument("--lang", choices=LANG, default="en")
    ps.add_argument("--color", action="store_true")
    ps.add_argument("--ci", action="store_true",
                    help="exit 1 when nothing matches (CI gate)")

    pt = sub.add_parser("stats", help="print index statistics")
    pt.add_argument("-i", "--index", default=DEFAULT_INDEX)
    pt.add_argument("--json", action="store_true")
    pt.add_argument("--lang", choices=LANG, default="en")

    return p


def cmd_index(args):
    started = time.time()
    if not os.path.isdir(args.dir):
        _err("not a directory: %s" % args.dir)
        return EXIT_ERROR
    last = [0.0]

    def progress(rel, n):
        now = time.time()
        if not args.quiet and now - last[0] > 0.5:
            last[0] = now
            sys.stderr.write("\r  ... %s (%d chunks)" % (rel, n))
            sys.stderr.flush()

    try:
        index = build_index(
            args.dir, all_text=args.all_text, max_bytes=args.max_bytes,
            chunk_size=args.chunk_size, overlap=args.overlap, progress=progress,
        )
    except (OSError, ValueError) as exc:
        _err(str(exc))
        return EXIT_ERROR
    if not args.quiet:
        sys.stderr.write("\r")
    if not index["chunks"]:
        _err("no readable text found under %s" % args.dir)
        return EXIT_ERROR
    size = save_index(index, args.output)
    st = index["stats"]
    if args.quiet:
        print("%s\t%d\t%d\t%d" % (args.output, st["files"], st["chunks"], size))
    else:
        print("ragpack: indexed %d files → %d chunks (%d terms) in %.2fs"
              % (st["files"], st["chunks"], st["terms"],
                 time.time() - started))
        print("  wrote %s (%.1f KB)" % (args.output, size / 1024.0))
    return EXIT_OK


def cmd_search(args):
    try:
        index = load_index(args.index)
    except OSError as exc:
        _err("cannot read index %s (%s) — run `ragpack index` first"
             % (args.index, exc.strerror or exc))
        return EXIT_ERROR
    except (ValueError, json.JSONDecodeError) as exc:
        _err("bad index: %s" % exc)
        return EXIT_ERROR

    if not index["chunks"]:
        _err("index has no chunks")
        return EXIT_ERROR

    from .tokenize import tokenize

    hits = run(index, args.query, k=args.k, path_boost=args.path_boost,
               min_score=args.min_score)
    result = {
        "query": args.query,
        "hits": [h.as_dict() for h in hits],
        "matched": len(hits),
        "index": index,
        "index_path": args.index,
        "q_tokens": tokenize(args.query),
    }

    if args.report:
        try:
            path, nbytes = write_report(result, args.report, lang=args.lang)
            if not args.json:
                print("ragpack: wrote %s (%d bytes)" % (path, nbytes))
        except OSError as exc:
            _err("cannot write report: %s" % exc)
            return EXIT_ERROR

    if args.json:
        sys.stdout.write(json_report(result, lang=args.lang))
    else:
        color = args.color or (sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)
        sys.stdout.write(text_report(result, lang=args.lang, color=color))

    if args.ci and not hits:
        return EXIT_NO_RESULTS
    return EXIT_OK


def cmd_stats(args):
    try:
        index = load_index(args.index)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _err("cannot read index %s (%s)" % (args.index, exc))
        return EXIT_ERROR
    st = index["stats"]
    if args.json:
        print(json.dumps({"index": args.index, "root": index["root"],
                          "built_at": index["built_at"], "stats": st},
                         ensure_ascii=False, indent=2))
        return EXIT_OK
    lang = args.lang
    labels = (("Files", "ملفات"), ("Chunks", "مقاطع"),
              ("Terms", "مصطلحات"), ("Tokens", "رموز")) if lang == "ar" else (
              ("Files", "Files"), ("Chunks", "Chunks"),
              ("Terms", "Terms"), ("Tokens", "Tokens"))
    print("%s: %s" % (("الفهرس", "Index")[lang == "en"], index["root"]))
    for key, label in labels:
        print("  %-8s %s" % (label, st[key.lower()]))
    print("  %-8s %.1f" % (("متوسط", "Avg len")[lang == "en"], index["avgdl"]))
    age = int(time.time()) - index["built_at"]
    print("  %-8s %ds" % (("العمر", "Age")[lang == "en"], age))
    return EXIT_OK


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return EXIT_ERROR
    args = parser.parse_args(argv)
    if args.cmd == "index":
        return cmd_index(args)
    if args.cmd == "search":
        return cmd_search(args)
    if args.cmd == "stats":
        return cmd_stats(args)
    parser.print_help()
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
