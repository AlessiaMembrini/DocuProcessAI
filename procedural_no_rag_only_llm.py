# pipeline/procedural_no_rag_only_llm.py
from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from utils.llm_utils import call_llm
from utils.prompts_base import (
    PromptBlocks,
    prompt_is_procedural_section,
    prompt_extract_diagram_steps,
    prompt_normalize_definition,
)

from utils.definitions_rag import (
    build_allowed_agents_from_definitions,
)

from utils.agent_utils import ground_agent


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class TitleNode:
    title: str
    level: int
    start_char: int
    end_char: Optional[int] = None
    parent_index: Optional[int] = None


# ----------------------------
# Helpers
# ----------------------------

def normalize_text(text: str) -> str:
    # non strippo tutto: mi serve coerenza per indici char
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def split_into_lines(text: str) -> List[str]:
    return (text or "").split("\n")


def parse_json_from_text(s: str) -> Dict[str, Any]:
    if not s:
        return {}
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                blob = s[start : i + 1]
                try:
                    return json.loads(blob)
                except Exception:
                    return {}
    return {}


def _safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = prefix + ":" + "|".join(str(p) for p in parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


# ----------------------------
# Chunking (2000-3000 chars, close on boundary)
# ----------------------------

_BOUNDARY_RE = re.compile(r"([\.!?])(\s|\n)|(\n{2,})|(;)(\s|\n)")


def chunk_text_char_range(
    text: str,
    *,
    min_chars: int = 2000,
    max_chars: int = 3000,
) -> List[Tuple[int, int, str]]:
    """
    Chunking cieco:
    - prendo una finestra max_chars
    - se >= min_chars, provo a chiudere su un boundary (., !, ?, ;, o doppio newline)
    - se non trovo boundary, taglio a max_chars
    Ritorna: [(start_char, end_char_exclusive, chunk_text)]
    """
    t = text or ""
    n = len(t)
    out: List[Tuple[int, int, str]] = []
    i = 0

    while i < n:
        j_max = min(n, i + max_chars)
        j_min = min(n, i + min_chars)

        if j_min >= n:
            out.append((i, n, t[i:n]))
            break

        window = t[j_min:j_max]
        m = None
        for mm in _BOUNDARY_RE.finditer(window):
            m = mm
            break

        if m:
            cut = j_min + m.end()
            cut = min(cut, n)
        else:
            cut = j_max

        if cut <= i:
            cut = min(n, i + max_chars)

        out.append((i, cut, t[i:cut]))
        i = cut

    return out


# ----------------------------
# Titles-only prompt (LLM-only)
# ----------------------------

def prompt_titles_only(chunk_text: str) -> str:
    return f"""
Sei un parser di documenti. Dal testo seguente estrai SOLO i titoli/sottotitoli VISIBILI nel testo.

Vincoli:
- Considera titolo una riga/intestazione che "introduce" un blocco (es: maiuscolo, numerata, breve, senza punto finale).
- NON includere frasi di contenuto.
- Se non trovi titoli, restituisci lista vuota.

Output JSON ESATTO:
{{
  "titles": [
    {{"title": "string", "level": 1|2|3}}
  ]
}}

TESTO (chunk):
\"\"\"{chunk_text}\"\"\"
""".strip()


def locate_title_positions_in_chunk(chunk_text: str, title: str) -> List[int]:
    if not title:
        return []

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip())

    chunk_n = norm(chunk_text)
    title_n = norm(title)
    if not chunk_n or not title_n:
        return []

    offs: List[int] = []
    raw = title.strip()
    if raw:
        start = 0
        while True:
            k = chunk_text.find(raw, start)
            if k < 0:
                break
            offs.append(k)
            start = k + max(1, len(raw))
    return offs


# ----------------------------
# Outline utilities
# ----------------------------

def build_outline_hierarchy(nodes: List[TitleNode]) -> List[TitleNode]:
    stack: List[int] = []
    for i, n in enumerate(nodes):
        while stack and nodes[stack[-1]].level >= n.level:
            stack.pop()
        n.parent_index = stack[-1] if stack else None
        stack.append(i)

    for i, n in enumerate(nodes):
        end = None
        for j in range(i + 1, len(nodes)):
            if nodes[j].level <= n.level:
                end = nodes[j].start_char
                break
        n.end_char = end
    return nodes


def outline_path(nodes: List[TitleNode], idx: int) -> str:
    parts = []
    cur = idx
    while cur is not None:
        parts.append(nodes[cur].title)
        cur = nodes[cur].parent_index
    return " > ".join(reversed(parts))


# ----------------------------
# Diagram helpers
# ----------------------------

def _normalize_yes_no(label_raw: str) -> str:
    lab = (label_raw or "").strip()
    low = lab.lower()
    if low in {"si", "sì", "yes"}:
        return "SI"
    if low in {"no", "no.", "not"}:
        return "NO"
    return lab.strip()


def link_steps_with_branches(all_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(all_steps, list):
        return []
    steps = [s for s in all_steps if isinstance(s, dict) and s.get("id")]
    if not steps:
        return []

    for i, step in enumerate(steps):
        step_type = (step.get("type") or "").strip()
        if isinstance(step_type, str) and step_type.startswith("gateway_"):
            branches: List[Dict[str, Any]] = []
            j = i + 1
            while j < len(steps):
                s2 = steps[j]
                meta = (s2.get("meta") or "").strip()
                if meta.startswith("branch="):
                    label_raw = meta.split("=", 1)[1].strip()
                    if label_raw:
                        label = _normalize_yes_no(label_raw)
                        seen = {b.get("label") for b in branches if b.get("label")}
                        if label in seen:
                            k2 = 2
                            while f"{label} ({k2})" in seen:
                                k2 += 1
                            label = f"{label} ({k2})"
                        branches.append({"label": label, "target": s2["id"]})
                    j += 1
                    continue
                break

            if not branches:
                next_target = steps[i + 1]["id"] if i + 1 < len(steps) else "end"
                branches = [{"label": "", "target": next_target}]
            step["branches"] = branches
        else:
            next_target = steps[i + 1]["id"] if i + 1 < len(steps) else "end"
            step["branches"] = [{"label": "", "target": next_target}]

    return steps


def _coerce_steps_schema(raw_steps: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_steps, list):
        return []
    out: List[Dict[str, Any]] = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or str(i + 1)).strip()
        stype = str(s.get("type") or "activity_task").strip()
        agent = str(s.get("agent") or "Operatore").strip()
        desc = str(s.get("description_synthetic") or s.get("description") or "").strip()
        meta = str(s.get("meta") or "").strip()
        out.append({"id": sid, "type": stype, "agent": agent, "description_synthetic": desc, "meta": meta})
    return out


# ----------------------------
# Light heuristic prefilter (cheap)
# ----------------------------

_BULLET_RE = re.compile(r"^\s*([-\u2022\*]|\(?[a-zA-Z0-9]+\)|[0-9]+\.)\s+", re.MULTILINE)
_DEONTIC_RE = re.compile(r"\b(deve|devono|è tenuto|è obbligato|obbligatorio|entro)\b", re.IGNORECASE)


def procedural_signal_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    score = 0.0
    if _BULLET_RE.search(t):
        score += 0.30
    if _DEONTIC_RE.search(t):
        score += 0.25
    if re.search(r"\b(prima|poi|successivamente|infine|entro)\b", t, re.IGNORECASE):
        score += 0.15
    return min(1.0, score)


# ----------------------------
# DEFINITIONS (Hybrid, NO-RAG): regex/euristiche + LLM su pochi candidati
# ----------------------------

_DEF_COLON_RE = re.compile(r"^\s*([A-Z][A-Z0-9/._-]{1,30})\s*[:=]\s*(.+)\s*$")
_DEF_DASH_RE = re.compile(r"^\s*([A-Z][A-Z0-9/._-]{1,30})\s*[–\-]\s*(.+)\s*$")
_DEF_INTENDE_RE = re.compile(r"(?i)^\s*per\s+([A-Z][A-Z0-9/._-]{1,30})\s+si\s+intende\s+(.+)\s*$")
_DEF_ACRONIMO_RE = re.compile(r"(?i)\b(acronimo|sigla)\b.*\b([A-Z][A-Z0-9/._-]{1,30})\b.*\b(significa|sta per|indica)\b")

_DEF_BAD_TERM_RE = re.compile(
    r"(?i)^(art\.?|capo|sezione|paragrafo|oggetto|scopo|ambito|definizioni|glossario|acronimi|allegato)$"
)

_CONTINUATION_RE = re.compile(r"^\s*(?:[,;:\)\]]|\w|\/|-)")


def _dedup_keep_best_definition(defs: List[Dict[str, Any]], *, max_defs: int) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for d in defs or []:
        if not isinstance(d, dict):
            continue
        term = str(d.get("term") or "").strip()
        definition = str(d.get("definition") or "").strip()
        if not term or not definition:
            continue
        k = term.lower()
        cur = best.get(k)
        if cur is None or len(definition) > len(str(cur.get("definition") or "")):
            best[k] = d

    out = list(best.values())

    def _key(x: Dict[str, Any]):
        ln = x.get("line_no")
        try:
            ln_i = int(ln)
        except Exception:
            ln_i = 10**9
        return (-len(str(x.get("definition") or "")), ln_i)

    out.sort(key=_key)
    return out[:max_defs]


def _is_probable_definition_line(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) < 6:
        return False
    if _BULLET_RE.match(s):  # qui usiamo bullet multi-line, ma su singola riga va bene
        return False
    if _DEF_COLON_RE.match(s) or _DEF_DASH_RE.match(s) or _DEF_INTENDE_RE.match(s):
        return True
    low = s.lower()
    if "si intende" in low or "definizion" in low or "glossario" in low or "acronim" in low:
        return True
    if _DEF_ACRONIMO_RE.search(s):
        return True
    return False


def _build_definition_candidates_from_lines(
    lines: List[str],
    *,
    max_candidates: int = 260,
    max_join_lines: int = 2,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    n = len(lines)

    for i in range(n):
        ln = (lines[i] or "").strip()
        if not ln:
            continue
        if not _is_probable_definition_line(ln):
            continue

        acc = [ln]
        for j in range(1, max_join_lines + 1):
            if i + j >= n:
                break
            nxt = (lines[i + j] or "").strip()
            if not nxt:
                break
            # stop se sembra nuovo blocco
            if re.match(r"^\s*\d+(\.\d+){0,4}\s+\S+", nxt):
                break
            if nxt.isupper() and len(nxt) <= 90 and len(nxt.split()) >= 2:
                break
            if _BULLET_RE.match(nxt):
                break
            if _CONTINUATION_RE.match(nxt) and not (_DEF_COLON_RE.match(nxt) or _DEF_DASH_RE.match(nxt) or _DEF_INTENDE_RE.match(nxt)):
                acc.append(nxt)
                continue
            break

        snippet = " ".join(acc).strip()
        if len(snippet) < 8:
            continue
        if i in seen:
            continue
        seen.add(i)
        out.append({"line_no": i, "text": snippet})
        if len(out) >= max_candidates:
            break

    return out


def extract_definitions_hybrid_no_rag(
    *,
    lines: List[str],
    max_defs: int = 140,
    max_candidates: int = 260,
) -> List[Dict[str, Any]]:
    candidates = _build_definition_candidates_from_lines(lines, max_candidates=max_candidates, max_join_lines=2)
    if not candidates:
        return []

    candidates = candidates[:max_candidates]

    prompt = (
        "Estrai DEFINIZIONI / ACRONIMI / RUOLI da righe candidate di un documento amministrativo.\n"
        "Input: lista di elementi con line_no e text.\n\n"
        "Regole:\n"
        "- Estrai solo definizioni esplicite (es. 'X: ...', 'X = ...', 'X - ...', 'Per X si intende ...').\n"
        "- Evita intestazioni pure e titoli (es. 'OGGETTO:', 'ALLEGATO', 'ART.' se è solo titolo).\n"
        "- term max 60 caratteri; definition max 1200.\n"
        "- Ogni output deve includere SEMPRE line_no originale.\n"
        f"- Massimo {max_defs} elementi.\n\n"
        'Output SOLO JSON:\n{ "definitions": [ {"term":"...", "definition":"...", "line_no":123} ] }\n\n'
        f"CANDIDATI:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    raw = parse_json_from_text(call_llm(prompt))
    defs = raw.get("definitions", []) if isinstance(raw, dict) else []

    parsed: List[Dict[str, Any]] = []
    for d in defs or []:
        if not isinstance(d, dict):
            continue
        term = str(d.get("term") or "").strip()
        definition = str(d.get("definition") or "").strip()
        try:
            ln_i = int(d.get("line_no"))
        except Exception:
            continue

        if not term or not definition:
            continue
        if len(term) < 2 or len(term) > 60:
            continue
        if _DEF_BAD_TERM_RE.match(term.strip().lower()):
            continue
        if len(definition) < 8:
            continue

        parsed.append({"term": term, "definition": definition[:1200], "line_no": ln_i})

    return _dedup_keep_best_definition(parsed, max_defs=max_defs)


def normalize_definitions_no_rag_via_llm(
    definitions_raw: List[Dict[str, Any]],
    *,
    max_terms: int = 60,
) -> List[Dict[str, Any]]:
    if not definitions_raw:
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for d in definitions_raw[:max_terms]:
        term = str(d.get("term") or "").strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)

        evidence_blocks = [
            {
                "kind": "definition",
                "text": str(d.get("definition") or "")[:900],
                "meta": {"term": term, "line_no": d.get("line_no", None)},
            }
        ]

        p = prompt_normalize_definition(term, evidence_blocks)
        res = parse_json_from_text(call_llm(p))

        normalized = str(res.get("normalized") or "").strip()
        canonical_agent = str(res.get("canonical_agent") or "").strip()
        aliases = res.get("aliases") if isinstance(res.get("aliases"), list) else []

        clean_aliases: List[str] = []
        for a in aliases[:8]:
            if not isinstance(a, str):
                continue
            a2 = a.strip()
            if not a2:
                continue
            if a2 not in clean_aliases:
                clean_aliases.append(a2)
        if term not in clean_aliases:
            clean_aliases.insert(0, term)

        out.append(
            {
                "term": term,
                "normalized": normalized[:160],
                "is_agent": bool(res.get("is_agent", False)),
                "canonical_agent": canonical_agent[:80],
                "aliases": clean_aliases[:8],
                "status": "ok",
            }
        )

    return out


def build_agent_alias_map_from_normalized_definitions(defs_norm: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in defs_norm or []:
        if not isinstance(d, dict):
            continue
        if not bool(d.get("is_agent", False)):
            continue
        canonical = str(d.get("canonical_agent") or "").strip()
        if not canonical:
            continue
        aliases = d.get("aliases") if isinstance(d.get("aliases"), list) else []
        term = str(d.get("term") or "").strip()
        if term:
            aliases = [term] + aliases
        for a in aliases[:10]:
            if not isinstance(a, str):
                continue
            k = a.strip().lower()
            if k and k not in out:
                out[k] = canonical
    return out


# ----------------------------
# MAIN: NO-RAG ONLY
# ----------------------------

def run_no_rag_only_pipeline_on_text(
    text: str,
    doc_id: str,
    output_root: str = "./output",
    *,
    min_chars_per_chunk: int = 2000,
    max_chars_per_chunk: int = 3000,
    max_titles_per_chunk: int = 30,
) -> Dict[str, Any]:
    """
    NO-RAG end-to-end (allineato alla struttura nuova, ma senza retrieval):
    0) definizioni: regex/euristiche + LLM su pochi candidati + normalizzazione LLM (no retrieval)
    1) chunking cieco (chars)
    2) LLM titles-only su chunk
    3) outline globale + gerarchia
    4) split sezioni per outline (char-based)
    5) per sezione: prompt_is_procedural_section + prompt_extract_diagram_steps
    """
    text = normalize_text(text)
    lines = split_into_lines(text)

    # (0) DEFINIZIONI: Hybrid + normalize (no retrieval)
    doc_definitions_raw = extract_definitions_hybrid_no_rag(lines=lines, max_defs=140, max_candidates=260)
    doc_definitions_norm = normalize_definitions_no_rag_via_llm(doc_definitions_raw, max_terms=60)
    agent_alias_map = build_agent_alias_map_from_normalized_definitions(doc_definitions_norm)
    allowed_agents = build_allowed_agents_from_definitions(doc_definitions_raw)

    # (1) chunking cieco
    chunks = chunk_text_char_range(
        text,
        min_chars=min_chars_per_chunk,
        max_chars=max_chars_per_chunk,
    )

    # (2) titles-only per chunk
    raw_title_hits: List[TitleNode] = []
    seen_key: set[str] = set()

    for (c_start, c_end, c_text) in chunks:
        p = prompt_titles_only(c_text)
        out = parse_json_from_text(call_llm(p))
        titles = out.get("titles") or []
        if not isinstance(titles, list):
            continue

        count = 0
        for t in titles:
            if not isinstance(t, dict):
                continue
            title = str(t.get("title") or "").strip()
            if not title:
                continue

            level = int(t.get("level") or 2)
            level = 1 if level <= 1 else (3 if level >= 3 else 2)

            offs = locate_title_positions_in_chunk(c_text, title)
            start_char = c_start + (offs[0] if offs else 0)

            key = f"{re.sub(r'\\s+', ' ', title.lower())}|{level}|{start_char//10}"
            if key in seen_key:
                continue
            seen_key.add(key)

            raw_title_hits.append(TitleNode(title=title, level=level, start_char=start_char))
            count += 1
            if count >= max_titles_per_chunk:
                break

    # (3) outline
    raw_title_hits.sort(key=lambda x: x.start_char)
    if not raw_title_hits:
        raw_title_hits = [TitleNode(title="DOCUMENTO", level=1, start_char=0)]
    outline = build_outline_hierarchy(raw_title_hits)

    # (4) sezioni (char-based)
    sections: List[Dict[str, Any]] = []
    for i, n in enumerate(outline):
        start = n.start_char
        end = n.end_char if n.end_char is not None else len(text)
        path = outline_path(outline, i)
        sections.append({"start_char": start, "end_char": end, "title": n.title, "path": path})

    # (5) tasks finali (classifica + steps)
    sections_flags: List[Dict[str, Any]] = []
    diagram_procedures: List[Dict[str, Any]] = []

    for sec_idx, sec in enumerate(sections):
        a = int(sec["start_char"])
        b = int(sec["end_char"])
        title = str(sec["title"])
        path = str(sec["path"])

        section_text = (text[a:b] or "").strip()
        if not section_text:
            continue

        # prefilter leggero
        heur = procedural_signal_score(section_text)
        if heur < 0.20 and not re.search(
            r"\b(procedur|modalità|istruzioni|iter|come\s+fare|compiti|attivit|verific)\b",
            title or "",
            re.IGNORECASE,
        ):
            sections_flags.append(
                {
                    "section_title": title,
                    "section_path": path,
                    "start_char": a,
                    "end_char": b,
                    "is_procedure": False,
                    "confidence": 0.0,
                    "reasons": ["prefilter_heuristic=false"],
                }
            )
            continue

        # (5.1) classificazione
        p_isproc = prompt_is_procedural_section(
            section_title=title,
            section_path=path,
            section_text=section_text,
            blocks=PromptBlocks(
                crossdoc_examples=[],
                local_similar_sections=[],
            ),
        )
        cls = parse_json_from_text(call_llm(p_isproc))
        is_proc = bool(cls.get("is_procedure", False))

        sections_flags.append(
            {
                "section_title": title,
                "section_path": path,
                "start_char": a,
                "end_char": b,
                "is_procedure": is_proc,
                "confidence": float(cls.get("confidence", 0.0) or 0.0),
                "reasons": list(cls.get("reasons", []) or []),
                "expected_missing_parts": list(cls.get("expected_missing_parts", []) or []),
            }
        )

        if not is_proc:
            continue

        # (5.2) estrazione steps
        procedure_id = f"{doc_id}_proc_{sec_idx:03d}"
        p_steps = prompt_extract_diagram_steps(
            procedure_id=procedure_id,
            section_title=title,
            subsection_title=title,
            section_path=path,
            section_text=section_text,
            blocks=PromptBlocks(
                crossdoc_examples=[],
                rag_context_blocks=[],
                definitions=doc_definitions_raw,  # evidenza per-doc
                allowed_agents=allowed_agents,
                supporting_actions=[],
            ),
        )
        extr = parse_json_from_text(call_llm(p_steps))

        raw_steps = _coerce_steps_schema(extr.get("steps"))
        linked = link_steps_with_branches(raw_steps)

        for s in linked:
            s["agent"] = ground_agent(s.get("agent", "Operatore"), agent_alias_map)

        diagram_procedures.append(
            {
                "procedure_id": procedure_id,
                "section_title": title,
                "subsection_title": title,
                "section_path": path,
                "status": str(extr.get("status", "partial")),
                "notes": list(extr.get("notes", []) or []),
                "original_text_full": section_text,
                "steps": linked,
                "points": [
                    {"point_label": str(i + 1), "original_text": st.get("description_synthetic", ""), "steps": [st]}
                    for i, st in enumerate(linked)
                ],
            }
        )

    # (6) save
    out_dir = os.path.join(output_root, doc_id)
    os.makedirs(out_dir, exist_ok=True)

    out_paths = {
        "result_json": os.path.join(out_dir, f"{doc_id}.no_rag_only.result.json"),
        "diagram_procedures_json": os.path.join(out_dir, f"{doc_id}.diagram_procedures.json"),
        "definitions_raw_json": os.path.join(out_dir, f"{doc_id}.definitions.raw.json"),
        "definitions_norm_json": os.path.join(out_dir, f"{doc_id}.definitions.normalized.json"),
        "chunks_json": os.path.join(out_dir, f"{doc_id}.chunks.json"),
    }

    _safe_write_json(out_paths["definitions_raw_json"], doc_definitions_raw)
    _safe_write_json(out_paths["definitions_norm_json"], doc_definitions_norm)
    _safe_write_json(out_paths["diagram_procedures_json"], diagram_procedures)
    _safe_write_json(out_paths["chunks_json"], [{"start_char": a, "end_char": b, "len": (b - a)} for (a, b, _) in chunks])

    result = {
        "doc_id": doc_id,
        "system_variant": "no_rag_only",
        "outline": [asdict(n) for n in outline],
        "sections": sections,
        "sections_procedural_flags": sections_flags,
        "diagram_procedures": diagram_procedures,
        "definitions_raw": doc_definitions_raw,
        "definitions_normalized": doc_definitions_norm,
        "stats": {
            "n_chars": len(text),
            "n_chunks": len(chunks),
            "n_titles": len(outline),
            "n_sections": len(sections),
            "n_proc_sections": sum(1 for x in sections_flags if x.get("is_procedure")),
            "n_diagram_procedures": len(diagram_procedures),
            "n_definitions_raw": len(doc_definitions_raw),
            "n_definitions_normalized": len(doc_definitions_norm),
            "n_agent_aliases": len(agent_alias_map),
        },
        "outputs": out_paths,
    }

    _safe_write_json(out_paths["result_json"], result)
    return result
