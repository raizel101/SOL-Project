"""One shared adapter binding for retrieval and local-model transport.

The adapter retains exact-offset SQLite FTS5 retrieval and loopback-only Ollama
transport. Keeping one shared module object also makes all generation calls
interceptable by the existing mocked regression tests.
"""

from . import local_rag as RAG

__all__ = ["RAG"]
