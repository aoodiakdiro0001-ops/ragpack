# ragpack

**Dependency-free local retrieval (BM25) for AI agents — and for anyone who
wants to search a folder of documents without a server.**

```
$ ragpack index ./docs
ragpack: indexed 58 files → 156 chunks (3580 terms) in 0.05s

$ ragpack search "sleep ratio darkwake" -k 3
 1. [12.275] sleeplens.md
     …`pmset -g log` live, or any saved log file: sleep/wake counts, total asleep v
     why: sleep=5.91, ratio=6.36
```

* **Zero dependencies** — Python 3.8+ standard library only. No `pip install`,
  no `numpy`, no `sentence-transformers`.
* **No network, no embeddings API, no telemetry** — there is no network code
  in this package. Your documents never leave the machine.
* **No database server** — the index is one JSON file you can diff, commit,
  or delete.
* **English + Arabic** — `--lang ar` for the report, and an Arabic stopword
  list so Arabic queries rank on nouns, not on `في`.
* **43/43 unit tests**, including a byte-identical `load → search → save`
  round-trip (the index you commit is the index you searched).

## Install

```bash
git clone https://github.com/aoodiakdiro0001-ops/ragpack
cd ragpack
python3 -m ragpack --version        # no install step — run it from the folder
# optional, to get a `ragpack` command on your PATH:
python3 -m pip install .
```

## Commands

### `ragpack index <dir>`

Walks a directory and writes an index.

| option | default | meaning |
|---|---|---|
| `-o, --output` | `ragpack-index.json` | where to write the index |
| `--all-text` | off | index every file, not only known text extensions |
| `--max-bytes` | 1 MiB | skip files larger than this |
| `--chunk-size` | 1200 | target characters per chunk |
| `--overlap` | 200 | characters of overlap between chunks |
| `-q, --quiet` | off | print `path\tfiles\tchunks\tbytes` (for scripts) |

A `.ragpackignore` file in the root takes one glob per line (`#` starts a
comment). `node_modules`, `.git`, `__pycache__`, virtualenvs and build output
are always skipped, as are binary files (NUL in the first 8 KB) and files over
`--max-bytes`.

### `ragpack search <query>`

| option | default | meaning |
|---|---|---|
| `-i, --index` | `ragpack-index.json` | index to query |
| `-k` | 10 | how many chunks to return |
| `--min-score` | 0.0 | drop anything below this |
| `--path-boost` | 3.0 | bonus when the query matches the file *name*; `0` disables |
| `--json` | off | machine-readable output on stdout |
| `--report FILE` | — | also write `.json` / `.md` / `.txt` to a file |
| `--lang` | `en` | `en` or `ar` |
| `--ci` | off | exit **1** when nothing matches, **0** otherwise |

Every hit prints a `why:` line naming the terms that contributed and how much
— so you can tell a real match from a coincidence.

### `ragpack stats`

Index size, chunk count, vocabulary, average chunk length and age. `--json`
for scripts.

## Exit codes

| code | meaning |
|---|---|
| 0 | success — and, for `search`, at least one result (or `--ci` not given) |
| 1 | `search --ci` found nothing |
| 2 | usage or I/O error (missing index, unreadable directory, bad version) |

`search --ci` exits 2 — never 0 — when the index is missing or unreadable, so
a CI job cannot go green against a stale or absent build.

## How it ranks

BM25Okapi (`k1=1.2`, `b=0.75`) over a chunk-level term-frequency map.

* **IDF floor at 0.07.** The textbook Okapi IDF goes *negative* for terms
  present in more than roughly half the corpus, which makes common terms
  subtract score and inverts the ranking on small corpora. Lucene clamps it
  for the same reason; so does ragpack.
* **Path boost ×3.** When a query matches the file name, you usually want
  that file. It is a small, printed, disableable nudge — never a hidden
  re-rank.
* **Word-boundary chunking.** Chunks break on paragraph then line then space
  edges, and the next window snaps forward to a word boundary, so a query
  never loses a word that was sliced in half.
* **Deduplicated chunks.** Identical content in two files is scored once
  (its term frequencies are merged), so a vendored copy cannot crowd out the
  original.

## Honest scope

ragpack **retrieves**. It does not generate answers, call a model, or embed
anything — pair it with whatever model you already use. It ranks English and
Arabic equally well because both are tokenized by the same code path;
languages without spaces between words (Chinese, Japanese, Thai) are **not**
segmented and will rank poorly — that is a real limitation, not a bug you
should discover in production.

## Tests

```bash
python3 tests/test_ragpack.py
```

43 tests, no network, no fixtures beyond temporary directories.

---

## العربية — ragpack

**استرجاع محلي بلا أي تبعيات (BM25) للوكلاء الذكين** — ولكي تبحث في مجلّد
مستنداتك دون خادم.

```
$ ragpack index ./docs
ragpack: indexed 58 files → 156 chunks (3580 terms) in 0.05s

$ ragpack search "الأسرار في تاريخ git" --lang ar -k 3
 1. [6.961] gha-audit.md
     لماذا: الأسرار=4.20, git=2.76
```

**صفر تبعيات:** مكتبة Python المعيارية 3.8+ فقط — بلا `pip install` وبلا
`numpy` وبلا نماذج تضمين. **بلا إنترنت وبلا API تضمين وبلا تتبّع:** لا يوجد
شفرة شبكة في هذا الحزم، ومستنداتك لا تغادر جهازك أبداً. **بلا خادم قاعدة
بيانات:** الفهرس ملف JSON واحد يمكنك تفديقه أو حذفه أو الاحتفاظ به. **عربي
وإنجليزي:** `--lang ar` للتقرير مع قائمة توقف عربية حتى يُرتَّب الاستعلام
العربي على الأسماء لا على «في». **43/43 اختباراً** من بينها جولة
`load → search → save` ببايتات متطابقة.

الأوامر: `index` لبناء الفهرس (مع `.ragpackignore` بأنماط glob، وتخطي الملفات
الثنائية والكبيرة و`node_modules` و`.git`)، و`search` للاستعلام مع خيارات
`-k` و`--min-score` و`--path-boost` و`--json` و`--report` و`--lang` و`--ci`،
و`stats` لإحصاءات الفهرس. **أكواد الخروج:** `0` نجاح (أو وجود نتائج مع
`--ci`)، `1` لا نتائج مع `--ci`، `2` خطأ استخدام أو ملف. **كيف يُرتِّب؟**
BM25Okapi بمعاملَي `k1=1.2` و`b=0.75` مع **سقف IDF عند 0.07** (الصيغة
الكتابية تصبح سالبة للمصطلحات الشائعة فيقلب الترتيب على المستندات القصيرة —
وكذلك تفعل Lucene)، و**تعزيز مسار الملف ×3** معطىً ومطبوع ويمكن تعطيله، و**تقسيم
على حدود الكلمات** حتى لا يُقطَّع مطلب في منتصفه، وإزالة تكرار المقاطع
المتطابقة.

**النطاق بصدق:** الأداة **تسترجع** ولا تُولّد إجابات ولا تستدعي نموذجاً ولا
تُضمن شيئاً — اقرنها بالنموذج الذي تستخدمه أصلاً. العربية والإنجليزية
تُعالَجان بالمسار نفسه، لكن اللغات بلا مسافات بين الكلمات (الصينية
اليابانية التايلاندية) **لا** تُقطَّع وتُرتَّب بشكل رديء — هذه حدود حقيقية
وليس علّة.

**الاختبارات:** `python3 tests/test_ragpack.py` — 43 اختباراً بلا شبكة وبلا
ملفات خارج المجلدات المؤقتة.

---

**Support:** bug fixes free for one year — 📧 baranavisat@gmail.com · ☎️ / WhatsApp +90 531 326 37 22 · wa.me/905313263722

**License:** MIT © 2026 Baran Avisat — free to use, modify and redistribute
with the copyright notice kept.
