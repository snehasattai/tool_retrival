"""RAG Pipeline Tool.

A small, self-contained retrieval-augmented-generation tool: chunks every
markdown file under knowledge_base/, embeds the chunks into a dedicated
Chroma collection (separate from the tool registry), and exposes a single
`query_knowledge_base` tool that returns the top matching passages.

Design choice: this tool returns retrieved passages rather than a synthesized
answer. Generating the final answer is left to the calling agent's own LLM
turn (it already has the passages in its tool-result context and will
naturally compose an answer + can decide to also call other tools in the
same turn). This avoids a second, redundant LLM call for every RAG query --
a classic "retriever tool" pattern as opposed to a "RAG-agent-in-a-box".
"""

from __future__ import annotations

import re
from pathlib import Path

import chromadb

from ..storage_paths import REPO_ROOT, chroma_dir
from ..tool_registry.embeddings import embed_documents, embed_query
from .gateway import gateway_tool

_COLLECTION_NAME = "knowledge_base"
_KB_DIR = REPO_ROOT / "knowledge_base"

_client: chromadb.ClientAPI | None = None
_collection = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=chroma_dir())
    return _client


def _chunk_markdown(text: str, source: str) -> list[dict]:
    """Split on markdown headings into passage-sized chunks."""
    sections = re.split(r"\n(?=#{1,3} )", text.strip())
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading_match = re.match(r"#{1,3} (.+)", section)
        heading = heading_match.group(1) if heading_match else source
        chunks.append({"text": section, "heading": heading, "source": source})
    return chunks


def _seed_collection(collection) -> None:
    all_chunks: list[dict] = []
    for md_file in sorted(_KB_DIR.glob("*.md")):
        all_chunks.extend(_chunk_markdown(md_file.read_text(), md_file.stem))

    if not all_chunks:
        return

    texts = [c["text"] for c in all_chunks]
    vectors = embed_documents(texts)
    collection.add(
        ids=[f"{c['source']}::{i}" for i, c in enumerate(all_chunks)],
        embeddings=vectors,
        documents=texts,
        metadatas=[{"heading": c["heading"], "source": c["source"]} for c in all_chunks],
    )


def get_collection(force_rebuild: bool = False):
    global _collection
    client = _get_client()
    if force_rebuild:
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(_COLLECTION_NAME)
    if collection.count() == 0:
        _seed_collection(collection)
    _collection = collection
    return collection


@gateway_tool(idempotent=False)
def query_knowledge_base(query: str, top_k: int = 3) -> dict:
    """Search product docs/guides (invoicing rules, dispute policy, fees) for passages relevant to a question.

    Use this whenever the user asks a policy/how-it-works question (e.g. "how
    long do I have to respond to a dispute?") rather than asking to perform
    an action. Returns the raw passages -- synthesize the final answer
    yourself from them, and mention there is no matching documentation if the
    passages don't actually answer the question.
    """
    collection = _collection if _collection is not None else get_collection()
    if collection.count() == 0:
        return {"passages": [], "note": "knowledge base is empty"}
    query_vector = embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    passages = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    for doc, meta in zip(docs, metas):
        passages.append({"text": doc, "source": meta.get("source"), "heading": meta.get("heading")})
    return {"passages": passages}
