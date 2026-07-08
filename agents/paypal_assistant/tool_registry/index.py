"""Tool Registry: a persistent vector index over every tool spec (real + decoy).

This is the "Tool Registry" box in the architecture diagram -- 500+ tools as
vectors, with category/metadata attached. It is deliberately dumb: it knows
nothing about ADK, agents, or execution. Its only job is
`search(query, top_k, categories) -> list[ToolSpec]`. The actual
"is this tool bound to the LLM's context" decision is made one layer up, in
tool_registry/semantic_toolset.py.

Two sync layers, two different directions:

  1. Code -> catalog (`sync_registry`, via `catalog_db.bootstrap_from_code`):
     specs.py/synthetic_specs.py (Python) declare which tools exist and their
     category/description/is_real. This step is idempotent and re-runs every
     time -- it upserts that data into a SQLite catalog table
     (tool_registry/catalog_db.py), which is the actual, fully-resolved
     metadata store from that point on (not a lazy re-derivation of
     docstrings at runtime). Tools removed from code get deleted from the
     catalog (and then Chroma) the same way.
  2. Catalog -> Chroma (content-hash diff): only tools whose *catalog*
     description changed since they were last embedded get (re-)embedded --
     unchanged tools cost zero embedding calls. This is what makes "add 3
     tools to a 2000-tool registry" cost 3 embedding calls instead of 2003.

Retrieval-time ToolSpec objects are built from catalog rows, with `func`
resolved via services/function_registry.py's auto-built name->callable map --
never from specs.py directly at query time. See `_validate_function_bindings`
for the explicit fail-fast check this split makes possible: a catalog row
claiming is_real=True with no matching registered function is a startup
error, not a runtime surprise.

Retrieval strategy -- dense retrieval, with an optional (off-by-default)
lexical rerank pass measured and rejected, not just assumed:
`search(..., rerank=True)` can cast a wider net and rerank it with
Reciprocal Rank Fusion against an IDF-weighted lexical (token-overlap)
signal, aimed at separating lexically-similar sibling tools (e.g.
get_dispute vs. get_dispute_status vs. list_disputes_by_customer) that dense
embeddings alone sometimes rank in the wrong order. Measured against
eval/tool_selection_eval.json: it does fix the one case it targets (a
dispute-detail-lookup query went from rank 2 to rank 1) but net-*hurts*
three other, previously-correct picks in the process (top-1 93% -> 86%) --
on this corpus the dense embeddings already carry enough signal that a
lexical pass mostly adds noise rather than resolving genuine ambiguity.
Kept implemented, tested, and available (`rerank=True`) since it may earn
its keep on a larger/noisier category where dense confidence is lower, but
`rerank=False` is the default because that's what the numbers said to do --
see eval/run_eval.py's three-way comparison. Deliberately not an LLM-based
reranker either way -- that would double the per-turn LLM cost/latency this
whole design exists to avoid.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter

import chromadb

from ..services.function_registry import FUNCTION_REGISTRY
from ..storage_paths import chroma_dir
from . import catalog_db
from .embeddings import embed_documents, embed_query
from .specs import REAL_TOOL_SPECS, ToolSpec
from .synthetic_specs import SYNTHETIC_TOOL_SPECS

logger = logging.getLogger("google_adk.tool_registry")

_COLLECTION_NAME = "tool_registry"
_RRF_K = 60  # standard Reciprocal Rank Fusion damping constant
_CANDIDATE_POOL_MULTIPLIER = 4  # cast a wider net pre-rerank, e.g. top_k=5 -> fetch 20


def _code_defined_specs() -> list[ToolSpec]:
    """The bootstrap source: what specs.py/synthetic_specs.py currently
    declare. Consulted by sync_registry() to refresh the SQLite catalog --
    NOT read at query time (see `_SPEC_BY_NAME`, built from the catalog).
    """
    return list(REAL_TOOL_SPECS) + list(SYNTHETIC_TOOL_SPECS)


def _validate_function_bindings(specs: list[ToolSpec]) -> None:
    """Fail fast at sync time, not at first tool call: every tool claiming
    is_real=True must have a matching entry in FUNCTION_REGISTRY. Catches a
    renamed/deleted business function immediately (Python itself would also
    catch a plain typo via AttributeError when specs.py builds REAL_TOOL_SPECS,
    but this check specifically covers the split between catalog data and the
    registry -- e.g. a stale catalog row for a function removed from code).
    """
    missing = [s.name for s in specs if s.is_real and s.name not in FUNCTION_REGISTRY]
    if missing:
        raise RuntimeError(
            f"tool_registry: {len(missing)} real tool(s) declared with is_real=True have no "
            f"matching callable in FUNCTION_REGISTRY (renamed or deleted function?): {missing}"
        )


def _spec_from_catalog_row(row: dict) -> ToolSpec:
    func = FUNCTION_REGISTRY.get(row["name"]) if row["is_real"] else None
    return ToolSpec(name=row["name"], category=row["category"], description=row["description"], func=func, is_real=row["is_real"])


_SPEC_BY_NAME: dict[str, ToolSpec] = {}

_client: chromadb.ClientAPI | None = None
_collection = None
last_sync_stats: dict[str, int] = {}


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=chroma_dir())
    return _client


def _content_hash(name: str, description: str) -> str:
    return hashlib.sha256(f"{name}:{description}".encode()).hexdigest()[:16]


def sync_registry(collection) -> dict[str, int]:
    """Bring the catalog in line with code, then bring Chroma in line with
    the catalog. Returns counts (embedded, deleted, unchanged) so callers
    can report what actually happened -- see scripts/seed_tool_registry.py.
    """
    global _SPEC_BY_NAME

    code_specs = _code_defined_specs()
    _validate_function_bindings(code_specs)

    # Layer 1: code -> catalog (idempotent; code is authoritative for
    # category/description/is_real -- see catalog_db.bootstrap_from_code).
    catalog_db.bootstrap_from_code([(s.name, s.category, s.description, s.is_real) for s in code_specs])
    code_names = {s.name for s in code_specs}
    catalog_names = {row["name"] for row in catalog_db.get_all_specs()}
    stale_names = [name for name in catalog_names if name not in code_names]
    if stale_names:
        catalog_db.delete_many(stale_names)

    catalog_rows = catalog_db.get_all_specs()
    _SPEC_BY_NAME = {row["name"]: _spec_from_catalog_row(row) for row in catalog_rows}

    # Layer 2: catalog -> chroma (only re-embed what actually changed).
    existing_hashes = catalog_db.get_all_hashes()
    to_upsert = [
        (row, _content_hash(row["name"], row["description"]))
        for row in catalog_rows
        if existing_hashes.get(row["name"], "") != _content_hash(row["name"], row["description"])
    ]

    if to_upsert:
        texts = [f"{row['name']}: {row['description']}" for row, _ in to_upsert]
        vectors = embed_documents(texts)
        collection.upsert(
            ids=[row["name"] for row, _ in to_upsert],
            embeddings=vectors,
            documents=texts,
            metadatas=[{"category": row["category"], "is_real": row["is_real"]} for row, _ in to_upsert],
        )
        catalog_db.mark_embedded([(row["name"], h) for row, h in to_upsert])

    if stale_names:
        collection.delete(ids=stale_names)

    return {
        "embedded": len(to_upsert),
        "deleted": len(stale_names),
        "unchanged": len(catalog_rows) - len(to_upsert),
    }


def get_collection(force_rebuild: bool = False):
    """Get the persistent tool-registry collection, incrementally synced to
    the current code-defined specs (via the catalog). Pass force_rebuild=True
    to wipe and re-embed everything from scratch (e.g. after switching
    embedding models/dimensionality).
    """
    global _collection
    client = _get_client()
    if force_rebuild:
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(_COLLECTION_NAME)
    global last_sync_stats
    last_sync_stats = sync_registry(collection)
    _collection = collection
    return collection


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _lexical_rank(query: str, candidates: list[ToolSpec]) -> dict[str, int]:
    """Rank candidates by an IDF-weighted lexical overlap with the query,
    where IDF is computed *within this candidate pool* (not the whole
    corpus). This matters: plain token-overlap (e.g. Jaccard) fails here
    because every tool in a category tends to share the category's own
    vocabulary (every disputes tool's description contains "dispute") --
    that common word would dominate a naive overlap score despite carrying
    no distinguishing signal. Weighting each shared token by
    log(1 + N/(1 + doc_freq)) down-weights words most candidates share and
    up-weights the words that actually separate them (e.g. "customer",
    "status", "evidence"), which is exactly what's needed to disambiguate
    lexically-similar sibling tools like get_dispute vs. get_dispute_status
    vs. list_disputes_by_customer.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return {s.name: i for i, s in enumerate(candidates)}

    candidate_tokens = {s.name: _tokenize(f"{s.name} {s.description}") for s in candidates}
    n = len(candidates)
    doc_freq: Counter[str] = Counter()
    for tokens in candidate_tokens.values():
        for token in query_tokens & tokens:
            doc_freq[token] += 1

    def score(spec: ToolSpec) -> float:
        shared = query_tokens & candidate_tokens[spec.name]
        return sum(math.log(1 + n / (1 + doc_freq[token])) for token in shared)

    ranked = sorted(candidates, key=score, reverse=True)
    return {s.name: i for i, s in enumerate(ranked)}


def _rrf_rerank(query: str, candidates: list[ToolSpec], top_k: int) -> list[ToolSpec]:
    """Hybrid rerank via Reciprocal Rank Fusion: blends the incoming dense
    (embedding-similarity) rank with the IDF-weighted lexical rank above,
    computed over the same candidate pool. Candidates already come in dense-
    rank order.
    """
    if len(candidates) <= 1:
        return candidates[:top_k]

    dense_rank = {s.name: i for i, s in enumerate(candidates)}
    lexical_rank = _lexical_rank(query, candidates)

    def rrf_score(spec: ToolSpec) -> float:
        return 1.0 / (_RRF_K + dense_rank[spec.name]) + 1.0 / (_RRF_K + lexical_rank[spec.name])

    return sorted(candidates, key=rrf_score, reverse=True)[:top_k]


def search(
    query: str,
    top_k: int = 5,
    categories: list[str] | None = None,
    rerank: bool = False,
) -> list[ToolSpec]:
    """Semantic top-k search over the tool registry, optionally scoped to categories.

    This is the read path the SemanticToolset calls on every LLM turn -- it
    must stay fast (one embedding call + a local ANN lookup), which is why
    the collection is pre-built rather than re-embedded per query.

    rerank=False by default -- measured, not assumed (see the module
    docstring and eval/run_eval.py): an IDF-weighted lexical rerank pass is
    implemented and available (rerank=True), but it net-hurt top-1 accuracy
    on this corpus, so plain dense ranking is what's actually used in
    production. Left available for a category where dense confidence is
    lower (e.g. many more lexically-similar tools) and a lexical signal
    might genuinely help -- re-run the eval before flipping this on.
    """
    collection = _collection if _collection is not None else get_collection()
    query_vector = embed_query(query)
    where = {"category": {"$in": categories}} if categories else None
    fetch_n = top_k * _CANDIDATE_POOL_MULTIPLIER if rerank else top_k
    results = collection.query(query_embeddings=[query_vector], n_results=fetch_n, where=where)
    ids = results["ids"][0] if results.get("ids") else []

    candidates = [_SPEC_BY_NAME[i] for i in ids if i in _SPEC_BY_NAME]
    if len(candidates) < len(ids):
        logger.warning(
            "tool_registry: %d id(s) returned by Chroma have no matching in-process "
            "ToolSpec -- the persisted index is stale relative to the running code "
            "(re-run scripts/seed_tool_registry.py)",
            len(ids) - len(candidates),
        )

    if not rerank:
        return candidates[:top_k]
    return _rrf_rerank(query, candidates, top_k)


def by_categories(categories: list[str]) -> list[ToolSpec]:
    """Direct metadata lookup, no embedding call -- used as the cold-start
    fallback (e.g. before the user has sent a message) so a domain agent
    still has its full (small) scoped tool set available immediately.
    """
    if not _SPEC_BY_NAME:
        get_collection()
    return [s for s in _SPEC_BY_NAME.values() if s.category in categories]


def registry_size() -> int:
    return len(_code_defined_specs())


def real_tool_count() -> int:
    return sum(1 for s in _code_defined_specs() if s.is_real)
