"""Tokenization shared by indexing and querying.

The tokenizer has to work for prose *and* source code, because the corpus is
a developer's notes and repositories. Decisions that matter:

* lowercase + Unicode NFKC so "Ä" and "a" compare equal;
* split camelCase and snake_case so `buildIndex` and `build_index` both
  contain `build` and `index` — a query for "index" must find them;
* keep numbers, drop single characters (they are noise in code and prose);
* a small English stopword list, kept tiny on purpose: retrieval quality
  collapses long before token count becomes a bottleneck, and a 300-word
  list hides real terms like "not" in a security corpus.
"""

import re
import unicodedata

STOPWORDS = frozenset(
    """
    a an the and or but if then else when while for to of in on at by with
    from as is are was were be been being it its this that these those there
    here not no nor so than too very just about into over under after before
    more most some any each both all can could would should will shall may
    might must do does did done have has had having i you he she we they
    me him her us them my your his our their what which who whom whose why how
    """.split()
)

# Arabic stopwords. Kept short on purpose: only the particles and pronouns
# that appear in almost every sentence. Without them an Arabic query ranks
# by the word "في" (in) instead of by the noun the user actually asked for.
STOPWORDS_AR = frozenset(
    """
    في من الي علي عن ان هذه هذا ذلك التي الذي التي مع ما لا هو هي ثم او و
    بين كل بعض نحو لدي كما قد كان كانت يكون تكون ولا لاذا ذالك هذة التي
    عبر دون خلال منذ لدى نحول نحوها نحوه اليه اليها عليه اليها منه اليها
    فيهما فيما فيهم فيهن ليس ليسوا انت انتم نحن هن هم انا لها له لها
    """.split()
)

STOPWORDS = STOPWORDS | STOPWORDS_AR

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# \w with re.UNICODE keeps letters of every script (ä, ر, 日) and drops the
# punctuation around them. An ASCII-only class would silently delete Arabic
# and accented queries, which is how `--lang ar` search breaks.
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_TOKEN = re.compile(r"\w+", re.UNICODE)


def normalize(text):
    """NFKC-fold and lowercase. Returns str."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text):
    """Token list for `text`. Deterministic and order-preserving.

    camelCase is split *before* lowercasing — after `.lower()` the boundary
    between `build` and `Index` no longer exists, and a query for "index"
    would stop finding `buildIndex`.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFKC", text)
    out = []
    for segment in _CAMEL.split(text):
        segment = _NON_WORD.sub(" ", segment.lower())
        for chunk in segment.split():
            # snake_case only: '-' and '.' are already separators by now
            for piece in chunk.split("_"):
                if not piece:
                    continue
                if len(piece) < 2 or piece in STOPWORDS:
                    continue
                if piece.isdigit() and len(piece) < 2:
                    continue
                out.append(piece)
    return out
