# utils/chroma_utils.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
from chromadb.errors import NotFoundError
from chromadb.utils import embedding_functions
import re
import hashlib

"""
Wrapper robusto per ChromaDB.

Garantisce:
- query mai con n_results<=0
- return types coerenti: retrieve_* ritorna SEMPRE list (mai None)
- filtra documenti vuoti in add
- where filtering via metadata
- ranked retrieval (ordine per similarità)
- ordered retrieval (ordine documento) basato su meta['chunk_id'] o meta['id']
"""


# ----------------------------
# Utilities
# ----------------------------

def _first_list(res: dict, key: str) -> List[Any]:
    v = res.get(key)
    if not isinstance(v, list) or not v:
        return []
    inner = v[0]
    return inner if isinstance(inner, list) else []


def _safe_int(x: Any, default: int = 10**9) -> int:
    try:
        if isinstance(x, bool):
            return default
        if isinstance(x, int):
            return x
        if isinstance(x, float) and int(x) == x:
            return int(x)
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip())
    except Exception:
        pass
    return default


def _order_key(meta: Dict[str, Any]) -> int:
    if not isinstance(meta, dict):
        return 10**9
    if "chunk_id" in meta:
        return _safe_int(meta.get("chunk_id"), default=10**9)
    return _safe_int(meta.get("id"), default=10**9)


def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not where or not isinstance(where, dict):
        return None
    clean = {k: v for k, v in where.items() if k and v is not None}
    return clean or None


def _empty_query_result() -> Dict[str, List[List[Any]]]:
    return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
def chroma_safe_collection_name(name: str, prefix: str = "col") -> str:
    """
    Converte un nome arbitrario in un nome valido per Chroma:
    - solo [a-zA-Z0-9._-]
    - lunghezza 3-512
    - deve iniziare e finire con alfanumerico
    """
    if name is None:
        name = ""

    raw = name.strip()

    # sostituisci tutto ciò che NON è ammesso con underscore
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw)

    # bordi: Chroma vuole alfanumerico all'inizio e alla fine
    safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)
    safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)

    # fallback se troppo corto
    if len(safe) < 3:
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        safe = f"{prefix}_{h}"

    # tronca se troppo lungo (hash in coda per stabilità)
    if len(safe) > 512:
        h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        safe = safe[:(512 - 1 - len(h))] + "_" + h
        safe = re.sub(r"^[^a-zA-Z0-9]+", "", safe)
        safe = re.sub(r"[^a-zA-Z0-9]+$", "", safe)
        if len(safe) < 3:
            safe = f"{prefix}_{h}"

    return safe

# ----------------------------
# Collection helpers
# ----------------------------

def default_embedding_function(model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
):
    ef = default_embedding_function(embedding_model)

    collection_name = chroma_safe_collection_name(collection_name, prefix="doc")

    try:
        return client.get_collection(name=collection_name, embedding_function=ef)
    except NotFoundError:
        return client.create_collection(name=collection_name, embedding_function=ef)



def delete_collection_if_exists(client: chromadb.PersistentClient, collection_name: str) -> bool:
    """Delete a collection if present.

    Returns True if deletion happened, False otherwise.
    """
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception:
        return False


def collection_count_safe(client: chromadb.PersistentClient, collection_name: str, *, embedding_model: str = "paraphrase-multilingual-mpnet-base-v2") -> int:
    """Best-effort count for an existing collection; returns 0 if missing/failed."""
    try:
        col = get_or_create_collection(client, collection_name, embedding_model=embedding_model)
        return int(col.count()) if col else 0
    except Exception:
        return 0


def safe_add(
    collection,
    ids: List[str],
    documents: List[str],
    metadatas: Optional[List[dict]] = None,
) -> int:
    """
    Aggiunge in modo robusto:
    - filtra documenti vuoti
    - fallback add 1-by-1 su errori (es. duplicate ids)
    """
    if not collection:
        return 0
    if not ids or not documents:
        return 0
    if metadatas is not None and len(metadatas) != len(documents):
        raise ValueError("metadatas deve avere la stessa lunghezza di documents.")

    filt_ids: List[str] = []
    filt_docs: List[str] = []
    filt_metas: List[dict] = []

    for i, (_id, doc) in enumerate(zip(ids, documents)):
        if not isinstance(_id, str) or not _id.strip():
            continue
        if not isinstance(doc, str) or not doc.strip():
            continue
        meta = (metadatas[i] if metadatas is not None else {}) or {}
        filt_ids.append(_id)
        filt_docs.append(doc)
        filt_metas.append(meta)

    if not filt_ids:
        return 0

    try:
        collection.add(ids=filt_ids, documents=filt_docs, metadatas=filt_metas)
        return len(filt_ids)
    except Exception:
        added = 0
        for _id, doc, meta in zip(filt_ids, filt_docs, filt_metas):
            try:
                collection.add(ids=[_id], documents=[doc], metadatas=[meta])
                added += 1
            except Exception:
                continue
        return added


# ----------------------------
# Query helpers
# ----------------------------

def _query_collection(
    collection,
    query: str,
    n_results: int,
    where: Optional[Dict[str, Any]] = None,
    include: Optional[List[str]] = None,
) -> Dict[str, List[List[Any]]]:
    """
    Wrapper sicuro: non lancia eccezioni, ritorna sempre un dict "tipo chroma".
    """
    empty = _empty_query_result()

    if not collection:
        return empty
    if not isinstance(query, str) or not query.strip():
        return empty

    try:
        n_results = int(n_results)
    except Exception:
        return empty

    # Chroma NON accetta n_results <= 0
    if n_results <= 0:
        return empty

    include = include or ["documents", "metadatas", "ids"]
    where = _normalize_where(where)

    # clamp su count
    try:
        cnt = collection.count()
        if not isinstance(cnt, int) or cnt <= 0:
            return empty
        n_results = min(n_results, cnt)
        if n_results <= 0:
            return empty
    except Exception:
        # se count fallisce, garantisco n_results>=1
        n_results = max(1, n_results)

    try:
        if where:
            return collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=include,
            )
        return collection.query(
            query_texts=[query],
            n_results=n_results,
            include=include,
        )
    except Exception:
        # ultimo tentativo senza where
        try:
            return collection.query(
                query_texts=[query],
                n_results=n_results,
                include=include,
            )
        except Exception:
            return empty


def retrieve_ranked_chunks_with_meta(
    collection,
    query: str,
    k: int,
    where: Optional[Dict[str, Any]] = None,
    include_distances: bool = False,
) -> List[Dict[str, Any]]:
    """
    Ranked retrieval: ordine per similarità (come ritorna Chroma).
    Ritorna SEMPRE list (mai None).
    Item schema:
      {"text": str, "metadata": dict, "id": str, "distance": float?}
    """
    if not collection:
        return []
    if not isinstance(query, str) or not query.strip():
        return []

    try:
        k = int(k)
    except Exception:
        return []
    if k <= 0:
        return []

    include = ["documents", "metadatas", "ids"]
    if include_distances:
        include.append("distances")

    res = _query_collection(collection, query=query, n_results=k, where=where, include=include)

    docs = _first_list(res, "documents")
    metas = _first_list(res, "metadatas")
    ids = _first_list(res, "ids")
    dists = _first_list(res, "distances") if include_distances else []

    if not docs:
        return []

    out: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        if not isinstance(doc, str) or not doc.strip():
            continue
        meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
        _id = ids[i] if i < len(ids) and isinstance(ids[i], str) else ""
        item = {"text": doc, "metadata": meta, "id": _id}
        if include_distances:
            try:
                item["distance"] = float(dists[i]) if i < len(dists) else None
            except Exception:
                item["distance"] = None
        out.append(item)

    return out


def retrieve_ordered_chunks(
    collection,
    query: str,
    k: int,
    with_meta: bool = False,
    where: Optional[Dict[str, Any]] = None,
    order_by: str = "auto",
) -> List[Union[str, Dict[str, Any]]]:
    """
    Ordered retrieval: prende i top-k per similarità ma li riordina secondo chunk_id/id.
    """
    if not collection:
        return []
    if not isinstance(query, str) or not query.strip():
        return []
    try:
        k = int(k)
    except Exception:
        return []
    if k <= 0:
        return []

    res = _query_collection(collection, query=query, n_results=k, where=where, include=["documents", "metadatas"])
    docs = _first_list(res, "documents")
    metas = _first_list(res, "metadatas")

    if not docs:
        return []

    retrieved: List[Tuple[int, Union[str, Dict[str, Any]]]] = []
    for doc, meta in zip(docs, metas or [{}] * len(docs)):
        if not isinstance(doc, str) or not doc.strip():
            continue
        meta = meta or {}

        if order_by == "chunk_id":
            idx = _safe_int(meta.get("chunk_id"), default=10**9)
        elif order_by == "id":
            idx = _safe_int(meta.get("id"), default=10**9)
        else:
            idx = _order_key(meta)

        item = {"text": doc, "metadata": meta} if with_meta else doc
        retrieved.append((idx, item))

    retrieved.sort(key=lambda x: x[0])
    return [x[1] for x in retrieved]


def query_in_context(
    collection,
    query: str,
    k: int = 5,
    where: Optional[Dict[str, Any]] = None,
    **filters,
) -> List[str]:
    if where is None and filters:
        where = dict(filters)
    hits = retrieve_ranked_chunks_with_meta(collection, query, k=max(int(k), 1), where=where)
    return [h.get("text", "") for h in hits if h.get("text")][:k]


# ----------------------------
# Backward-compat: setup_chromadb
# ----------------------------

def setup_chromadb(
    documents: List[str],
    collection_name: str,
    db_path: str = "./chroma_db",
    metadata_list: Optional[List[dict]] = None,
):
    """
    Compatibilità con la tua v1:
    - crea collection se non esiste
    - popola SOLO se count==0
    """
    client = chromadb.PersistentClient(path=db_path)
    collection = get_or_create_collection(client, collection_name, embedding_model="paraphrase-multilingual-mpnet-base-v2")

    if collection.count() == 0:
        if not documents:
            raise ValueError("Nessun documento fornito per la popolazione iniziale.")
        if metadata_list is not None and len(metadata_list) != len(documents):
            raise ValueError("metadata_list deve avere la stessa lunghezza di documents.")

        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []

        for i, doc in enumerate(documents):
            if not isinstance(doc, str) or not doc.strip():
                continue
            meta = (metadata_list[i] if metadata_list is not None else {"id": i}) or {}
            meta.setdefault("id", i)
            ids.append(f"{collection_name}_{i}")
            docs.append(doc)
            metas.append(meta)

        if not docs:
            raise ValueError("Nessun documento valido (non vuoto) fornito.")

        safe_add(collection, ids=ids, documents=docs, metadatas=metas)

    return collection
