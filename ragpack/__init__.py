"""ragpack — dependency-free local retrieval for AI agents.

Indexes a folder of text files and answers queries with BM25 ranking.
No network, no embeddings API, no database server: one JSON index file and
the Python standard library.

Public surface:
    ragpack.index.build_index(root, opts)   -> Index
    ragpack.search.run(index, query, opts)  -> list[Hit]
    ragpack.cli.main(argv)                 -> int (exit code)
"""

__version__ = "1.0.0"
__all__ = ["build_index", "run", "__version__"]

from .index import build_index
from .search import run
