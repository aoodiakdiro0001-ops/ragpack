"""ragpack test suite — no network, no fixtures on disk beyond tmp dirs.

Covers: tokenization (camel/snake/unicode/stopwords), chunking (overlap and
boundary behaviour), BM25 scoring (IDF floor, path boost, ranking order),
index round-trip (byte-identical save after load+search), ignore rules,
binary/large-file skips, the three reporters in English and Arabic, and the
CLI exit codes including the `--ci` gate.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from ragpack import cli, index as indexmod, report, search, tokenize as tokmod

CORPUS = {
    "notes/bm25.md": (
        "# BM25 ranking\n\n"
        "BM25 is a probabilistic ranking function used by search engines. "
        "It scores a document by term frequency weighted against inverse "
        "document frequency. The k1 parameter controls saturation.\n"
    ),
    "notes/vector-db.md": (
        "# Vector databases\n\n"
        "A vector database stores embeddings and answers with similarity "
        "search. For a small corpus a vector database is overkill: you pay "
        "for an embeddings API and a server just to search a few hundred "
        "files. Plain BM25 over local text works offline.\n"
    ),
    "code/build_index.py": (
        "def buildIndex(path):\n"
        "    \"\"\"Build the index from path.\"\"\"\n"
        "    return index_file(path)\n\n"
        "def build_index(path):\n"
        "    return buildIndex(path)\n"
    ),
    "code/README.md": (
        "RAG without a database.\n\n"
        "This tool answers questions from your own documents using BM25 "
        "retrieval. No embeddings, no network, no server.\n"
    ),
    "data/skip.bin": b"binary\x00content\xff",
    "data/huge.txt": "x" * (indexmod.MAX_BYTES_DEFAULT + 1),
    "node_modules/ignored.md": "should never be indexed: pineapple",
    ".git/also.md": "should never be indexed either: watermelon",
}


def make_corpus(root):
    for rel, content in CORPUS.items():
        full = os.path.join(root, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        mode = "wb" if isinstance(content, bytes) else "w"
        kw = {} if isinstance(content, bytes) else {"encoding": "utf-8"}
        with open(full, mode, **kw) as fh:
            fh.write(content)


class TokenizeTests(unittest.TestCase):
    def test_camel_and_snake_split(self):
        self.assertIn("build", tokmod.tokenize("buildIndex"))
        self.assertIn("index", tokmod.tokenize("buildIndex"))
        self.assertEqual(tokmod.tokenize("build_index"), ["build", "index"])

    def test_case_and_accents_fold(self):
        self.assertEqual(tokmod.tokenize("BM25 Ranking"), ["bm25", "ranking"])
        self.assertEqual(tokmod.tokenize("ÄÖÜ"), ["äöü"])

    def test_stopwords_and_short_tokens_dropped(self):
        toks = tokmod.tokenize("the a of an index to be")
        self.assertEqual(toks, ["index"])

    def test_single_chars_and_short_numbers_dropped(self):
        self.assertEqual(tokmod.tokenize("a b c 42 7 x1"), ["42", "x1"])

    def test_empty(self):
        self.assertEqual(tokmod.tokenize(""), [])
        self.assertEqual(tokmod.tokenize(None), [])


class ChunkTests(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(indexmod.chunk_text("hello", size=100), ["hello"])

    def test_overlap_recovers_boundary(self):
        text = ("alpha " * 60).strip()
        chunks = indexmod.chunk_text(text, size=100, overlap=30)
        self.assertGreater(len(chunks), 1)
        # overlap means content repeats, so count can only grow; what must
        # never happen is a word sliced in half across two chunks
        self.assertGreaterEqual(" ".join(chunks).count("alpha"),
                                text.count("alpha"))
        for c in chunks:
            self.assertEqual(set(c.split()), {"alpha"}, c)

    def test_prefers_paragraph_break(self):
        text = ("para one " * 20 + "\n\n" + "para two " * 20).strip()
        for c in indexmod.chunk_text(text, size=200, overlap=40):
            self.assertIn("para ", c)

    def test_zero_size_returns_whole(self):
        self.assertEqual(indexmod.chunk_text("abc", size=0), ["abc"])


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="ragpack-test-")
        make_corpus(cls.root)
        cls.index = indexmod.build_index(cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_index_skips_binary_huge_and_ignored(self):
        paths = [f["path"] for f in self.index["files"]]
        self.assertNotIn("data/skip.bin", paths)
        self.assertNotIn("data/huge.txt", paths)
        self.assertNotIn("node_modules/ignored.md", paths)
        self.assertNotIn(".git/also.md", paths)
        self.assertIn("notes/bm25.md", paths)

    def test_stats_shape(self):
        st = self.index["stats"]
        for key in ("files", "chunks", "terms", "tokens"):
            self.assertGreater(st[key], 0, key)
        self.assertGreater(self.index["avgdl"], 0)

    def test_ranking_puts_relevant_doc_first(self):
        hits = search.run(self.index, "BM25 ranking function", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].path, "notes/bm25.md")

    def test_vector_db_query(self):
        hits = search.run(self.index, "vector database embeddings", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].path, "notes/vector-db.md")

    def test_camel_query_finds_python_symbol(self):
        hits = search.run(self.index, "buildIndex", k=5)
        self.assertTrue(hits)
        self.assertTrue(any(h.path.endswith("build_index.py") for h in hits))

    def test_no_match_returns_empty(self):
        self.assertEqual(search.run(self.index, "zzzz qqqq"), [])

    def test_empty_query_returns_empty(self):
        self.assertEqual(search.run(self.index, ""), [])
        self.assertEqual(search.run(self.index, "   "), [])
        self.assertEqual(search.run(self.index, "the a of an"), [])

    def test_path_boost_ranks_filename_match(self):
        with_path = search.run(self.index, "bm25 notes", k=3,
                               path_boost=search.DEFAULT_PATH_BOOST)
        without = search.run(self.index, "bm25 notes", k=3, path_boost=0.0)
        self.assertEqual(with_path[0].path, "notes/bm25.md")
        # disabling the boost must not reorder the top hit here, but must
        # strictly reduce its score
        self.assertLess(without[0].score, with_path[0].score)

    def test_idf_floor_is_never_negative(self):
        # a term in every chunk must not subtract score
        self.assertGreater(search.idf(10, 10), 0.0)
        self.assertGreaterEqual(search.idf(1, 1), search.IDF_FLOOR)
        self.assertEqual(search.idf(0, 0), search.IDF_FLOOR)

    def test_k_limits_results(self):
        self.assertLessEqual(len(search.run(self.index, "index", k=1)), 1)

    def test_min_score_filters(self):
        everything = search.run(self.index, "index search", k=100, min_score=0.0)
        strict = search.run(self.index, "index search", k=100, min_score=1e6)
        self.assertTrue(everything)
        self.assertEqual(strict, [])

    def test_why_lists_contributing_terms(self):
        hits = search.run(self.index, "BM25 ranking", k=1)
        self.assertTrue(hits[0].why)
        for term, score in hits[0].why:
            self.assertGreater(score, 0.0)
            self.assertIn(term, ("bm25", "ranking"))

    def test_snippet_marks_ellipsis(self):
        long_doc = {"id": 99, "path": "p.md", "chunk": 0, "chars": 5000,
                    "tfs": {"needle": 1}, "len": 10,
                    "preview": ("filler " * 40) + "needle" + (" tail" * 20)}
        snip = search._snippet(long_doc, ["needle"])
        self.assertIn("needle", snip)
        self.assertTrue(snip.startswith("…") or snip.endswith("…"))


class RoundTripTests(unittest.TestCase):
    def test_save_load_search_identical(self):
        root = tempfile.mkdtemp(prefix="ragpack-rt-")
        out = os.path.join(root, "idx.json")
        try:
            make_corpus(root)
            idx = indexmod.build_index(root)
            indexmod.save_index(idx, out)
            with open(out, "rb") as fh:
                before = fh.read()
            loaded = indexmod.load_index(out)
            a = [h.as_dict() for h in search.run(loaded, "vector database")]
            indexmod.save_index(loaded, out)
            with open(out, "rb") as fh:
                after = fh.read()
            self.assertEqual(before, after, "load+save must be byte-identical")
            b = [h.as_dict() for h in search.run(loaded, "vector database")]
            self.assertEqual(a, b)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_version_mismatch_rejected(self):
        root = tempfile.mkdtemp(prefix="ragpack-ver-")
        out = os.path.join(root, "idx.json")
        try:
            make_corpus(root)
            idx = indexmod.build_index(root)
            idx["version"] = 999
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(idx, fh)
            with self.assertRaises(ValueError):
                indexmod.load_index(out)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class IgnoreTests(unittest.TestCase):
    def test_ragpackignore_patterns(self):
        root = tempfile.mkdtemp(prefix="ragpack-ign-")
        try:
            make_corpus(root)
            with open(os.path.join(root, ".ragpackignore"), "w") as fh:
                fh.write("# keep notes private\nnotes/vector-db.md\n*.bin\n")
            paths = [rel for rel, _ in indexmod.walk(root)]
            self.assertNotIn("notes/vector-db.md", paths)
            self.assertIn("notes/bm25.md", paths)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_all_text_flag_indexes_binary_named_file(self):
        root = tempfile.mkdtemp(prefix="ragpack-all-")
        try:
            with open(os.path.join(root, "weird.dat"), "w") as fh:
                fh.write("plain text with no extension match")
            paths = [rel for rel, _ in indexmod.walk(root, all_text=True)]
            self.assertIn("weird.dat", paths)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="ragpack-rep-")
        make_corpus(cls.root)
        cls.index = indexmod.build_index(cls.root)
        cls.result = {
            "query": "vector database",
            "hits": [h.as_dict() for h in search.run(cls.index, "vector database", k=3)],
            "matched": 1,
            "index": cls.index,
            "index_path": "idx.json",
            "q_tokens": tokmod.tokenize("vector database"),
        }

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_text_report_english(self):
        out = report.text_report(self.result, lang="en")
        self.assertIn("ragpack — retrieval", out)
        self.assertIn("vector database", out)

    def test_text_report_arabic(self):
        out = report.text_report(self.result, lang="ar")
        self.assertIn("ragpack — الاسترجاع", out)
        self.assertIn("الاستعلام:", out)
        self.assertIn("ملفات:", out)

    def test_json_report_parses_and_has_no_internal_tokens(self):
        data = json.loads(report.json_report(self.result))
        self.assertEqual(data["query"], "vector database")
        self.assertTrue(data["results"])
        # tfs (term frequencies) are internal scoring state, not output
        self.assertNotIn("tfs", json.dumps(data))

    def test_md_report_table(self):
        out = report.md_report(self.result, lang="en")
        self.assertIn("| score |", out)
        self.assertIn("notes/vector-db.md", out)

    def test_no_results_message_both_langs(self):
        empty = dict(self.result, hits=[], matched=0)
        self.assertIn("No chunk matched", report.text_report(empty, lang="en"))
        self.assertIn("لا يوجد مقطع", report.text_report(empty, lang="ar"))

    def test_write_report_dispatches_by_extension(self):
        tmp = tempfile.mkdtemp(prefix="ragpack-wr-")
        try:
            for name in ("r.txt", "r.json", "r.md"):
                path, nbytes = report.write_report(
                    self.result, os.path.join(tmp, name))
                self.assertTrue(os.path.exists(path))
                self.assertGreater(nbytes, 0)
            with open(os.path.join(tmp, "r.json")) as fh:
                json.load(fh)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="ragpack-cli-")
        make_corpus(cls.root)
        cls.idx = os.path.join(cls.root, "idx.json")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _run(self, *argv):
        from io import StringIO
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = StringIO(), StringIO()
        try:
            code = cli.main(list(argv))
            return code, sys.stdout.getvalue(), sys.stderr.getvalue()
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    def test_index_then_search_then_stats(self):
        code, out, err = self._run("index", self.root, "-o", self.idx, "-q")
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(self.idx))

        code, out, err = self._run("search", "BM25 ranking", "-i", self.idx)
        self.assertEqual(code, 0, err)
        self.assertIn("notes/bm25.md", out)

        code, out, err = self._run("stats", "-i", self.idx)
        self.assertEqual(code, 0, err)
        self.assertIn("Chunks", out)

    def test_search_json_output(self):
        self._run("index", self.root, "-o", self.idx, "-q")
        code, out, err = self._run("search", "vector database",
                                   "-i", self.idx, "--json")
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertEqual(data["query"], "vector database")
        self.assertTrue(data["results"])

    def test_search_arabic(self):
        self._run("index", self.root, "-o", self.idx, "-q")
        code, out, err = self._run("search", "BM25", "-i", self.idx, "--lang", "ar")
        self.assertEqual(code, 0, err)
        self.assertIn("الاستعلام", out)

    def test_ci_exit_1_when_no_match(self):
        self._run("index", self.root, "-o", self.idx, "-q")
        code, out, err = self._run("search", "zzzznotfound", "-i", self.idx, "--ci")
        self.assertEqual(code, cli.EXIT_NO_RESULTS)

    def test_ci_exit_0_when_match(self):
        self._run("index", self.root, "-o", self.idx, "-q")
        code, out, err = self._run("search", "BM25", "-i", self.idx, "--ci")
        self.assertEqual(code, cli.EXIT_OK)

    def test_missing_index_is_error_not_crash(self):
        code, out, err = self._run("search", "anything",
                                   "-i", "/nonexistent/idx.json", "--ci")
        self.assertEqual(code, cli.EXIT_ERROR)
        self.assertIn("ragpack:", err)

    def test_index_of_empty_dir_errors(self):
        empty = tempfile.mkdtemp(prefix="ragpack-empty-")
        try:
            code, out, err = self._run("index", empty, "-o",
                                       os.path.join(empty, "i.json"))
            self.assertEqual(code, cli.EXIT_ERROR)
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_no_args_prints_help_and_errors(self):
        code, out, err = self._run()
        self.assertEqual(code, cli.EXIT_ERROR)

    def test_version_flag(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)

    def test_report_file_written(self):
        self._run("index", self.root, "-o", self.idx, "-q")
        target = os.path.join(self.root, "out.json")
        code, out, err = self._run("search", "BM25", "-i", self.idx,
                                   "--report", target)
        self.assertEqual(code, 0, err)
        self.assertTrue(os.path.exists(target))
        with open(target) as fh:
            self.assertEqual(json.load(fh)["query"], "BM25")


class ModuleEntryTests(unittest.TestCase):
    def test_module_invocation(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.dirname(HERE)
        proc = subprocess.run(
            [sys.executable, "-m", "ragpack", "--version"],
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(HERE),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ragpack", proc.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=1)
