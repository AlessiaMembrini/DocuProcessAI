# utils/hierarchical_rag.py
from __future__ import annotations

import re
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from utils.chroma_utils import safe_add, retrieve_ranked_chunks_with_meta


# ----------------------------
# IDs / section mapping
# ----------------------------

def _stable_id_any(prefix: str, *parts: Any) -> str:
    raw = prefix + ":" + "|".join(str(p) for p in parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def make_section_id(doc_id: str, a: int, b: int, title: str) -> str:
    return _stable_id_any("sec", doc_id, a, b, (title or "").strip()[:80])


def assign_section_id_to_span(
    doc_id: str,
    sections: List[Tuple[int, int, str, str]],
    line_start: int,
    line_end: int,
) -> Optional[str]:
    """
    Ritorna il section_id della sezione che massimizza la sovrapposizione
    con lo span [line_start, line_end]. Se non trova nulla, ritorna None.
    """
    if line_start is None or line_end is None:
        return None
    if line_end < line_start:
        line_start, line_end = line_end, line_start

    best = None
    best_overlap = 0

    for (a, b, title, _path) in sections or []:
        # overlap inclusivo
        ov = max(0, min(line_end, b) - max(line_start, a) + 1)
        if ov > best_overlap:
            best_overlap = ov
            best = make_section_id(doc_id, a, b, title)

    return best

def _section_id_for_line(doc_id: str, sections: List[Tuple[int, int, str, str]], line_no: int) -> Optional[str]:
    for (a, b, title, _path) in sections or []:
        if a <= line_no <= b:
            return make_section_id(doc_id, a, b, title)
    return None


# ----------------------------
# Section indexing (parent docs)
# ----------------------------

def add_sections_to_doc_collection(
    doc_collection,
    doc_id: str,
    sections: List[Tuple[int, int, str, str]],
    lines: List[str],
    max_chars: int = 1200,
) -> int:
    """
    Indicizza sezioni come documenti parent.
    documents = titolo + snippet della sezione (contenuto del doc corrente -> ok)
    """
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    for (a, b, title, path) in sections or []:
        sec_id = make_section_id(doc_id, a, b, title)
        txt = "\n".join(lines[a:b + 1]).strip()
        if not txt:
            continue
        doc_txt = f"{title}\n{path}\n\n{txt[:max_chars]}"

        ids.append(_stable_id_any("sec_doc", doc_id, sec_id))
        docs.append(doc_txt)
        metas.append({
            "doc_id": doc_id,
            "kind": "section",
            "section_id": sec_id,
            "title": title,
            "path": path,
            "line_start": a,
            "line_end": b,
        })

    if not ids:
        return 0
    return safe_add(doc_collection, ids=ids, documents=docs, metadatas=metas)


def _pick_top_section_ids(section_hits: List[dict], top_n: int = 3) -> List[str]:
    sec_ids: List[str] = []
    for h in section_hits or []:
        meta = (h.get("metadata") or {})
        sid = meta.get("section_id")
        if sid and sid not in sec_ids:
            sec_ids.append(sid)
        if len(sec_ids) >= top_n:
            break
    return sec_ids


def retrieve_child_chunks_in_top_sections(
    doc_collection,
    doc_id: str,
    query: str,
    *,
    child_kind: str,
    top_sections: int = 3,
    k_per_section: int = 6,
    include_distances: bool = False,
) -> List[dict]:
    """
    Parent/child retrieval:
      1) query su kind=section -> prendo top section_id
      2) per ogni section_id -> query su child_kind con filtro section_id
    """
    # 1) parents
    section_hits = retrieve_ranked_chunks_with_meta(
        doc_collection,
        query=query[:900],
        k=max(6, top_sections * 3),
        where={"doc_id": doc_id, "kind": "section"},
        include_distances=include_distances,
    )
    top_sec_ids = _pick_top_section_ids(section_hits, top_n=top_sections)
    if not top_sec_ids:
        # fallback: retrieval flat
        return retrieve_ranked_chunks_with_meta(
            doc_collection,
            query=query[:900],
            k=top_sections * k_per_section,
            where={"doc_id": doc_id, "kind": child_kind},
            include_distances=include_distances,
        )

    # 2) children per section
    out: List[dict] = []
    for sid in top_sec_ids:
        hits = retrieve_ranked_chunks_with_meta(
            doc_collection,
            query=query[:900],
            k=k_per_section,
            where={"doc_id": doc_id, "kind": child_kind, "section_id": sid},
            include_distances=include_distances,
        )
        out.extend(hits or [])

    # dedup by (kind, chunk_id) if present
    seen = set()
    deduped = []
    for h in out:
        meta = (h.get("metadata") or {})
        key = (meta.get("kind"), meta.get("chunk_id"), meta.get("term"), meta.get("section_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    return deduped


# ----------------------------
# Query decomposition for agent/definitions
# ----------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)

def _norm_agent(s: str) -> str:
    t = (s or "").strip()
    t = _WS_RE.sub(" ", t)
    return t


def _strip_punct_keep_spaces(s: str) -> str:
    t = _PUNCT_RE.sub(" ", s or "")
    t = _WS_RE.sub(" ", t).strip()
    return t


def _agent_tokens(s: str) -> List[str]:
    t = _strip_punct_keep_spaces(s)
    toks = [x for x in t.split(" ") if len(x) >= 3]
    return toks[:6]


def decompose_agent_queries(agent: str, max_q: int = 8) -> List[str]:
    """
    Sub-query semplici (no invenzioni semantiche):
      - forma originale
      - lower
      - senza punteggiatura
      - token principali (singoli + bigrammi)
    """
    base = _norm_agent(agent)
    if not base:
        return []

    q: List[str] = []
    q.append(base)
    q.append(base.lower())

    no_p = _strip_punct_keep_spaces(base)
    if no_p and no_p not in q:
        q.append(no_p)

    toks = _agent_tokens(base)
    for t in toks:
        if t not in q:
            q.append(t)
        if len(q) >= max_q:
            break

    # bigrammi token
    if len(q) < max_q and len(toks) >= 2:
        for i in range(len(toks) - 1):
            bg = f"{toks[i]} {toks[i+1]}"
            if bg not in q:
                q.append(bg)
            if len(q) >= max_q:
                break

    # dedup keep order
    seen = set()
    out = []
    for x in q:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out[:max_q]


# ----------------------------
# Definition indexing + retrieval focused on agents
# ----------------------------

def _infer_def_line_no(d: Dict[str, Any]) -> Optional[int]:
    for k in ("line_no", "line", "start_line", "line_start"):
        v = d.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def index_definitions_with_section_id(
    doc_collection,
    doc_id: str,
    definitions: List[Dict[str, Any]],
    sections: List[Tuple[int, int, str, str]],
    *,
    max_chars: int = 900,
) -> int:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []

    for i, d in enumerate(definitions or []):
        term = (d.get("term") or "").strip()
        raw = (d.get("raw_text") or d.get("definition") or "").strip()
        if not raw:
            continue

        ln = _infer_def_line_no(d)
        sec_id = _section_id_for_line(doc_id, sections, ln) if ln is not None else None

        _id = _stable_id_any("def", doc_id, i, term)
        ids.append(_id)
        docs.append(f"{term}: {raw}"[:max_chars])
        metas.append({
            "doc_id": doc_id,
            "kind": "definition",
            "term": term,
            "section_id": sec_id,
            "line_no": ln,
        })

    if not ids:
        return 0
    return safe_add(doc_collection, ids=ids, documents=docs, metadatas=metas)


def retrieve_definition_support_for_agents(
    doc_collection,
    doc_id: str,
    definitions: List[Dict[str, Any]],
    sections: List[Tuple[int, int, str, str]],
    *,
    mode: str,
    section_text: Optional[str] = None,
    allowed_agents: Optional[List[str]] = None,
    top_sections: int = 3,
    k_per_section: int = 4,
) -> Dict[str, Any]:
    """
    mode:
      - "index_only": indicizza definizioni con section_id, ritorna {"defs_added": n}
      - "retrieve": ritorna {"blocks":[...]} per agenti presenti nella sezione
    """
    if mode == "index_only":
        n = index_definitions_with_section_id(doc_collection, doc_id, definitions, sections)
        return {"defs_added": n}

    # retrieve
    txt = section_text or ""
    agents = allowed_agents or []
    agents_in_section = []
    low = txt.lower()
    for a in agents:
        a2 = (a or "").strip()
        if not a2:
            continue
        if a2.lower() in low:
            agents_in_section.append(a2)

    agents_in_section = agents_in_section[:6]  # limitiamo costo
    blocks: List[dict] = []

    for ag in agents_in_section:
        subqs = decompose_agent_queries(ag, max_q=6)
        # parent/child: prima scegli sezioni, poi definizioni
        # qui usiamo come query la subquery più informativa (prima) per scegliere sezioni;
        # poi recuperiamo definizioni con tutte le subquery nelle top sezioni.
        parent_query = subqs[0] if subqs else ag

        section_hits = retrieve_ranked_chunks_with_meta(
            doc_collection,
            query=parent_query[:400],
            k=max(6, top_sections * 3),
            where={"doc_id": doc_id, "kind": "section"},
            include_distances=False,
        )
        top_sec_ids = _pick_top_section_ids(section_hits, top_n=top_sections)

        # se non trovi sezioni, fallback flat su definizioni
        def_hits: List[dict] = []
        if not top_sec_ids:
            for q in subqs[:4]:
                def_hits.extend(retrieve_ranked_chunks_with_meta(
                    doc_collection,
                    query=q[:400],
                    k=4,
                    where={"doc_id": doc_id, "kind": "definition"},
                    include_distances=False,
                ))
        else:
            for sid in top_sec_ids:
                for q in subqs[:4]:
                    def_hits.extend(retrieve_ranked_chunks_with_meta(
                        doc_collection,
                        query=q[:400],
                        k=k_per_section,
                        where={"doc_id": doc_id, "kind": "definition", "section_id": sid},
                        include_distances=False,
                    ))

        # dedup / pick top few
        seen = set()
        taken = 0
        for h in def_hits:
            meta = (h.get("metadata") or {})
            key = (meta.get("term"), meta.get("section_id"), (h.get("text") or "")[:60])
            if key in seen:
                continue
            seen.add(key)
            t = (h.get("text") or "").strip()
            if not t:
                continue
            blocks.append({
                "evidence_id": f"{doc_id}_def_{meta.get('term', '')}_{meta.get('section_id', '')}",
                "kind": "definition",
                "text": t[:900],
                "meta": {
                    "agent_query": ag,
                    "term": meta.get("term", ""),
                    "section_id": meta.get("section_id", None),
                }
            })
            taken += 1
            if taken >= 2:
                break

    return {"blocks": blocks}
