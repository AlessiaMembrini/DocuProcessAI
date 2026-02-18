# pipeline/procedural_rag_chroma.py
from __future__ import annotations

import os
import re
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import chromadb

from utils.llm_utils import call_llm
from utils.chroma_utils import (
    chroma_safe_collection_name,
    get_or_create_collection,
    safe_add,
    retrieve_ranked_chunks_with_meta,
)

from utils.prompts_base import (
    PromptBlocks,
    prompt_classify_titles,
    prompt_is_procedural_section,
    prompt_extract_diagram_steps,
)

from utils.definitions_rag import (
    build_allowed_agents_from_definitions,
    build_support_query,
    is_outside_section,
)

# legacy-style agent grounding (alias map)
from utils.agent_utils import ground_agent

# parent/child + query decomposition + section indexing
from utils.hierarchical_rag import (
    _section_id_for_line,
    make_section_id,
    assign_section_id_to_span,
    add_sections_to_doc_collection,
    retrieve_child_chunks_in_top_sections,
    retrieve_definition_support_for_agents,  # includes index_only + retrieve modes
)

try:
    from diagrams.run_diagrams import generate_three_diagram_sets
except Exception:
    generate_three_diagram_sets = None


# ----------------------------
# LLM safety budgets (chars ~= tokens)
# ----------------------------
LLM_MAX_INPUT_CHARS = 360_000        # hard guardrail per call (sotto 128k token con margine)
LLM_MAX_SECTION_CHARS = 35_000       # sezione passata a LLM (anti-OCR gigante)
LLM_MAX_JSONBLOCKS_CHARS = 120_000   # json.dumps(context_blocks) massimo
LLM_MAX_ITEMS_FOR_TITLES = 220       # limita items per prompt_classify_titles
LLM_MAX_DEF_CANDIDATES = 180         # limita candidates per definizioni
LLM_MAX_CAND_TEXT_CHARS = 380        # ogni candidate text massimo


# ----------------------------
# Small text utilities
# ----------------------------
def _trim_text_chars(text: str, max_chars: int, *, suffix: str = "\n...[TRUNCATED]...") -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= max_chars:
        return text
    cut = text.rfind("\n", 0, max_chars)
    if cut < int(max_chars * 0.75):
        cut = max_chars
    return text[:cut] + suffix


def _trim_section_for_llm(section_text: str, max_chars: int = LLM_MAX_SECTION_CHARS) -> str:
    """
    Mantiene inizio + fine (utile con OCR, dove info importanti possono stare alla fine).
    """
    s = (section_text or "").strip()
    if len(s) <= max_chars:
        return s
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return s[:head] + "\n...[MIDDLE TRUNCATED]...\n" + s[-tail:]


def _shrink_context_blocks_for_llm(
    blocks: List[dict],
    *,
    max_blocks: int = 12,
    max_block_text_chars: int = 650,
) -> List[dict]:
    out: List[dict] = []
    for b in (blocks or [])[:max_blocks]:
        if not isinstance(b, dict):
            continue
        nb = dict(b)
        txt = nb.get("text", "")
        if isinstance(txt, str):
            nb["text"] = _trim_text_chars(txt, max_block_text_chars)
        out.append(nb)
    return out


def _safe_json_dumps(obj: Any, *, max_chars: int) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return _trim_text_chars(s, max_chars)


def _cap_llm_prompt(prompt: str) -> str:
    return _trim_text_chars(prompt or "", LLM_MAX_INPUT_CHARS)


def call_llm_safe(prompt: str) -> str:
    """
    Hard cap sul prompt per evitare errori di context_length_exceeded.
    """
    return call_llm(_cap_llm_prompt(prompt))


def _adaptive_k(doc_lines: int, base: int, *, k_min: int, k_max: int) -> int:
    """
    k dinamico: documenti molto lunghi => k più basso.
    """
    try:
        n = int(doc_lines)
    except Exception:
        n = 0
    if n <= 0:
        return max(k_min, min(k_max, base))

    if n >= 8000:
        scale = 0.45
    elif n >= 4000:
        scale = 0.60
    elif n >= 2000:
        scale = 0.75
    else:
        scale = 1.0

    k = int(round(base * scale))
    return max(k_min, min(k_max, k))


# ----------------------------
# Data structures
# ----------------------------
@dataclass
class TitleNode:
    title: str
    level: int
    start_line: int
    end_line: Optional[int] = None
    parent_index: Optional[int] = None


@dataclass
class ProcedureRecord:
    section_title: str
    section_path: str
    start_line: int
    end_line: int
    status: str
    steps: List[str]
    notes: List[str]
    evidence_chunk_ids: List[str]


# ----------------------------
# Basic helpers
# ----------------------------
def _stable_id_any(prefix: str, *parts: Any) -> str:
    raw = prefix + ":" + "|".join(str(p) for p in parts)
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{h}"


def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join([ln.rstrip() for ln in text.split("\n")])
    return text


def split_into_lines(text: str) -> List[str]:
    return (text or "").split("\n")


def parse_json_from_text(s: str) -> Dict[str, Any]:
    """
    Parser robusto: se l'LLM ritorna testo con JSON "sporco", prova a estrarre il primo oggetto {...}.
    """
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


def _extract_json_object(text: str) -> dict:
    """
    Estrae il primo oggetto JSON {...} da una stringa.
    """
    if not text:
        return {}
    s = text.strip()
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


# ----------------------------
# Definition summarization (legacy normalization)
# ----------------------------
def summarize_definition_with_llm(term: str, definition: str, *, max_words: int = 8) -> str:
    """
    Label brevissima (6-8 parole) tipo ruolo/entità, senza verbi coniugati.
    Output: stringa.
    """
    term = (term or "").strip()
    definition = (definition or "").strip()
    if not term or not definition:
        return term or ""

    prompt = (
        "Sintetizza una definizione amministrativa in una label brevissima.\n\n"
        "Vincoli OBBLIGATORI:\n"
        f"- Massimo {max_words} parole.\n"
        "- Nessun verbo coniugato (evita: è/sono/si intende/viene/deve/devono/può).\n"
        "- Stile: nome di ruolo/attore o concetto.\n"
        "- Niente virgolette, niente elenco, niente punto finale.\n"
        "- Non ripetere inutilmente il termine se la label è già informativa.\n\n"
        "Output SOLO JSON:\n"
        '{ "summary": "..." }\n\n'
        f"TERMINE: {term}\n"
        f"DEFINIZIONE: {definition}\n"
    )

    raw = call_llm_safe(prompt)
    obj = _extract_json_object(raw)
    summary = str(obj.get("summary") or "").strip()

    if not summary:
        summary = (raw or "").strip().splitlines()[0].strip()

    summary = re.sub(r"\s+", " ", summary).strip()
    summary = summary.strip(' "\'')
    summary = summary.rstrip(".;:,")
    words = summary.split()
    if len(words) > max_words:
        summary = " ".join(words[:max_words]).strip()

    return summary or term


def _dedup_keep_best_definition(defs: List[Dict[str, Any]], *, max_defs: int) -> List[Dict[str, Any]]:
    """
    Dedup per term (case-insensitive), tenendo la definizione più informativa (più lunga).
    """
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


# ----------------------------
# Chunking (per-doc)
# ----------------------------
def build_line_chunks(lines: List[str], doc_id: str, sections=None) -> Tuple[List[str], List[dict], List[str]]:
    """
    Indicizza SOLO righe non vuote.
    Ritorna (docs, metas, ids) allineati 1:1 per safe_add.
    """
    docs: List[str] = []
    metas: List[dict] = []
    ids: List[str] = []

    for i, ln in enumerate(lines):
        t = (ln or "").strip()
        if not t:
            continue

        sec_id = _section_id_for_line(doc_id, sections or [], i)

        docs.append(t)
        metas.append(
            {
                "doc_id": doc_id,
                "kind": "line",
                "chunk_id": i,  # chunk_id==line_no (debug)
                "line_no": i,
                "section_id": sec_id,
            }
        )
        ids.append(_stable_id_any("line", doc_id, i))

    return docs, metas, ids


def build_passage_chunks(
    lines: List[str],
    doc_id: str,
    max_chars: int = 1200,
    *,
    sections: Optional[List[Tuple[int, int, str, str]]] = None,
) -> Tuple[List[str], List[dict], List[str]]:
    """
    Passage chunking per paragrafo (split su righe vuote).
    """
    passages: List[Tuple[int, int, str]] = []
    buff: List[str] = []
    start_line = 0

    def flush(end_line: int):
        nonlocal buff, start_line
        if buff:
            txt = "\n".join(buff).strip()
            if txt:
                passages.append((start_line, end_line, txt))
        buff = []

    for i, ln in enumerate(lines):
        if (ln or "").strip() == "":
            flush(i - 1)
            start_line = i + 1
        else:
            if not buff:
                start_line = i
            buff.append(ln)

    flush(len(lines) - 1)

    docs: List[str] = []
    metas: List[dict] = []
    ids: List[str] = []
    chunk_id = 0

    for (a, b, txt) in passages:

        def _emit(seg_txt: str, seg_a: int, seg_b: int):
            nonlocal chunk_id
            sec_id = assign_section_id_to_span(doc_id, sections or [], seg_a, seg_b)
            docs.append(seg_txt)
            metas.append(
                {
                    "doc_id": doc_id,
                    "kind": "passage",
                    "chunk_id": chunk_id,
                    "line_start": seg_a,
                    "line_end": seg_b,
                    "section_id": sec_id,
                }
            )
            ids.append(_stable_id_any("passage", doc_id, seg_a, seg_b, chunk_id))
            chunk_id += 1

        if len(txt) <= max_chars:
            _emit(txt, a, b)
        else:
            parts = re.split(r"(?<=[\.\;\:])\s+", txt)
            cur: List[str] = []
            cur_len = 0
            for p in parts:
                if cur_len + len(p) + 1 > max_chars and cur:
                    seg = " ".join(cur).strip()
                    if seg:
                        _emit(seg, a, b)
                    cur = []
                    cur_len = 0
                cur.append(p)
                cur_len += len(p) + 1
            if cur:
                seg = " ".join(cur).strip()
                if seg:
                    _emit(seg, a, b)

    return docs, metas, ids


# ----------------------------
# Heuristics (titles + procedural signals)
# ----------------------------
TITLE_NUM_RE = re.compile(r"^\s*(\d+(\.\d+){0,4})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*([-\u2022\*]|\(?[a-zA-Z0-9]+\)|[0-9]+\.)\s+")
DEONTIC_RE = re.compile(r"\b(deve|devono|è tenuto|è obbligato|obbligatorio|entro)\b", re.IGNORECASE)

IMPERATIVE_RE = re.compile(
    r"\b("
    r"rimuovere|pulire|lavare|risciacquare|asciugare|verificare|controllare|"
    r"compilare|inserire|registrare|inviare|consegnare|firmare|archiviare|"
    r"posizionare|chiudere|aprire|bloccare|sbloccare|selezionare|premere|"
    r"assicurarsi|accertarsi|misurare|pesare|etichettare|smaltire|sanificare"
    r")\b",
    re.IGNORECASE,
)

TITLE_KEYWORD_RE = re.compile(
    r"^\s*("
    r"art\.?\s*\d+|capo\s+[ivxlc]+|sezione\s+[ivxlc]+|paragrafo\s+\d+|"
    r"oggetto|finalit[àa]|scopo|ambito|definizioni|glossario|acronimi|"
    r"modalit[àa]|procedura|procedure|requisiti|documentazione|allegati|"
    r"responsabilit[àa]|competenze|istruzioni|fasi|passaggi|termini|"
    r"verifiche|controlli|sanzioni|modulistica"
    r")\b",
    re.IGNORECASE,
)

ALLEGATO_BARE_RE = re.compile(r"^\s*ALLEGATO\s+[A-Z0-9]+\s*$", re.IGNORECASE)
SEQ_MARKERS_RE = re.compile(r"\b(prima|poi|successivamente|infine|entro)\b", re.IGNORECASE)


def _starts_with_upper(s: str) -> bool:
    s = (s or "").strip()
    return bool(re.match(r"^[A-ZÀ-ÖØ-Þ]", s))


def _upper_ratio(s: str) -> float:
    alpha = [c for c in (s or "") if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / max(1, len(alpha))


def _norm_header_key(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t.upper()


def find_repeated_header_lines(lines: List[str], *, min_count: int = 3) -> set[str]:
    """
    Euristica: identifica righe ripetute (header/footer OCR) e le ignora in title-candidates.
    """
    counts: Dict[str, int] = {}
    for ln in lines or []:
        k = _norm_header_key(ln)
        if not k:
            continue
        if len(k) < 8:
            continue
        if re.fullmatch(r"[0-9\s]+", k):
            continue
        counts[k] = counts.get(k, 0) + 1
    return {k for k, v in counts.items() if v >= min_count}


def next_line_looks_like_heading(next_line: str) -> bool:
    s = (next_line or "").strip()
    if not s:
        return False
    if TITLE_NUM_RE.match(s):
        return True
    if TITLE_KEYWORD_RE.search(s):
        return True
    if s.endswith(":") and len(s) <= 90 and _starts_with_upper(s):
        return True
    if _upper_ratio(s) >= 0.75 and len(s.split()) >= 2 and len(s) <= 90:
        return True
    return False


def title_heuristic_score(line: str) -> float:
    s = (line or "").strip()
    if not s:
        return 0.0
    if len(s) > 160:
        return 0.0
    if BULLET_RE.match(s):
        return 0.0
    if s.count(",") >= 1:
        return 0.0

    score = 0.0
    if TITLE_NUM_RE.match(s):
        score += 0.45
    if TITLE_KEYWORD_RE.search(s):
        score += 0.35
    if _starts_with_upper(s):
        score += 0.18
    ur = _upper_ratio(s)
    if ur > 0.6:
        score += 0.18
    if s.endswith(":"):
        score += 0.18
    w = len(s.split())
    if 1 <= w <= 10:
        score += 0.10
    if re.search(r"\b(è|sono|deve|devono|può|possono|viene|vengono)\b", s, re.IGNORECASE):
        score -= 0.15
    if s.endswith(".") and w > 10:
        score -= 0.10

    return max(0.0, min(1.0, score))


def procedural_signal_score(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    score = 0.0
    if BULLET_RE.search(t):
        score += 0.30
    if DEONTIC_RE.search(t):
        score += 0.25
    if SEQ_MARKERS_RE.search(t):
        score += 0.15
    if IMPERATIVE_RE.search(t):
        score += 0.20
    return min(1.0, score)


# ----------------------------
# RAG helpers (windows)
# ----------------------------
def get_line_window(lines: List[str], center: int, radius: int = 2) -> List[Dict[str, Any]]:
    out = []
    for ln in range(max(0, center - radius), min(len(lines), center + radius + 1)):
        out.append({"line_no": ln, "text": lines[ln]})
    return out


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
                end = nodes[j].start_line - 1
                break
        n.end_line = end
    return nodes


def outline_path(nodes: List[TitleNode], idx: int) -> str:
    parts = []
    cur = idx
    while cur is not None:
        parts.append(nodes[cur].title)
        cur = nodes[cur].parent_index
    return " > ".join(reversed(parts))


# ----------------------------
# Diagram JSON helpers
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
    """
    Post-process: se trovi gateway_* seguiti da righe meta 'branch=...', crea branches.
    """
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
# Title candidates
# ----------------------------
def build_title_candidates(
    lines: List[str],
    *,
    min_score: float = 0.22,
    max_candidates: int = 900,
    header_repeat_threshold: int = 3,
) -> List[Tuple[int, str, float]]:
    repeated = find_repeated_header_lines(lines, min_count=header_repeat_threshold)

    out: List[Tuple[int, str, float]] = []
    for i, ln in enumerate(lines):
        raw = (ln or "").strip()
        if not raw:
            continue

        if _norm_header_key(raw) in repeated:
            continue

        if ALLEGATO_BARE_RE.match(raw):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if not next_line_looks_like_heading(nxt):
                continue

        sc = title_heuristic_score(raw)
        if sc >= min_score:
            out.append((i, raw, sc))

        if len(out) >= max_candidates:
            break

    return out


# ----------------------------
# DEFINITIONS (Hybrid): regex/heuristics on lines + (optional) retrieval + LLM
# ----------------------------
_DEF_COLON_RE = re.compile(r"^\s*([A-Z][A-Z0-9/._-]{1,30})\s*[:=]\s*(.+)\s*$")
_DEF_DASH_RE = re.compile(r"^\s*([A-Z][A-Z0-9/._-]{1,30})\s*[–\-]\s*(.+)\s*$")
_DEF_INTENDE_RE = re.compile(r"(?i)^\s*per\s+([A-Z][A-Z0-9/._-]{1,30})\s+si\s+intende\s+(.+)\s*$")
_DEF_ACRONIMO_RE = re.compile(
    r"(?i)\b(acronimo|sigla)\b.*\b([A-Z][A-Z0-9/._-]{1,30})\b.*\b(significa|sta per|indica)\b"
)

_DEF_BAD_TERM_RE = re.compile(
    r"(?i)^(art\.?|capo|sezione|paragrafo|oggetto|scopo|ambito|definizioni|glossario|acronimi|allegato)$"
)

_CONTINUATION_RE = re.compile(r"^\s*(?:[,;:\)\]]|\w|\/|-)")


def _is_probable_definition_line(line: str) -> bool:
    s = (line or "").strip()
    if not s or len(s) < 6:
        return False
    if BULLET_RE.match(s):
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
    """
    Passata deterministica:
    - seleziona righe che sembrano definizioni
    - concatena 0..max_join_lines righe successive se sembrano continuation
    """
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
            if TITLE_NUM_RE.match(nxt) or TITLE_KEYWORD_RE.search(nxt):
                break
            if BULLET_RE.match(nxt):
                break
            if _CONTINUATION_RE.match(nxt) and not (
                _DEF_COLON_RE.match(nxt) or _DEF_DASH_RE.match(nxt) or _DEF_INTENDE_RE.match(nxt)
            ):
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


def _retrieve_definition_line_candidates_semantic(
    doc_collection,
    doc_id: str,
    *,
    k_per_query: int = 50,
    max_out: int = 180,
) -> List[Dict[str, Any]]:
    """
    Passata semantica (booster): retrieval su kind="line".
    """
    queries = [
        "definizioni glossario acronimi si intende",
        "per X si intende",
        "acronimo significa",
        "sigla significa",
        "ruolo responsabile ufficio sportello operatore richiedente",
    ]
    hits: List[Dict[str, Any]] = []
    for q in queries:
        hits.extend(
            retrieve_ranked_chunks_with_meta(
                doc_collection,
                query=q,
                k=k_per_query,
                where={"doc_id": doc_id, "kind": "line"},
                include_distances=False,
            )
        )

    seen = set()
    out: List[Dict[str, Any]] = []
    for h in hits or []:
        meta = (h.get("metadata") or {})
        ln = meta.get("line_no")
        if ln is None:
            continue
        try:
            ln_i = int(ln)
        except Exception:
            continue
        if ln_i in seen:
            continue
        seen.add(ln_i)

        txt = (h.get("text") or "").strip()
        if not txt or len(txt) < 8:
            continue

        out.append({"line_no": ln_i, "text": txt})
        if len(out) >= max_out:
            break
    return out


def extract_definitions_hybrid(
    *,
    lines: List[str],
    doc_collection,
    doc_id: str,
    sections: List[Tuple[int, int, str, str]],
    max_defs: int = 140,
    max_candidates: int = 260,
    use_semantic_booster: bool = True,
) -> List[Dict[str, Any]]:
    """
    1) euristiche locali su linee
    2) (opzionale) retrieval semantico booster
    3) LLM su pochi candidati => {term, definition, line_no}
    4) dedup + section_id
    """
    cand_det = _build_definition_candidates_from_lines(lines, max_candidates=max_candidates, max_join_lines=2)

    cand_sem: List[Dict[str, Any]] = []
    if use_semantic_booster and doc_collection is not None:
        try:
            cand_sem = _retrieve_definition_line_candidates_semantic(
                doc_collection,
                doc_id,
                k_per_query=_adaptive_k(len(lines), 50, k_min=18, k_max=50),
                max_out=_adaptive_k(len(lines), 180, k_min=80, k_max=180),
            )
        except Exception:
            cand_sem = []

    merged: Dict[int, str] = {}
    for c in (cand_det or []):
        try:
            merged[int(c["line_no"])] = str(c.get("text") or "").strip()
        except Exception:
            continue
    for c in (cand_sem or []):
        try:
            ln = int(c["line_no"])
        except Exception:
            continue
        txt = str(c.get("text") or "").strip()
        if not txt:
            continue
        if ln not in merged or len(txt) > len(merged[ln]):
            merged[ln] = txt

    candidates = [{"line_no": ln, "text": merged[ln]} for ln in sorted(merged.keys())]
    if not candidates:
        return []

    dyn_max = min(int(max_candidates), LLM_MAX_DEF_CANDIDATES)
    candidates = candidates[:dyn_max]
    for c in candidates:
        if isinstance(c, dict) and isinstance(c.get("text"), str):
            c["text"] = _trim_text_chars(c["text"], LLM_MAX_CAND_TEXT_CHARS)

    prompt = (
        "Estrai DEFINIZIONI / ACRONIMI / RUOLI da righe candidate di un documento amministrativo.\n"
        "Input: lista di elementi con line_no e text.\n\n"
        "Regole:\n"
        "- Estrai solo definizioni esplicite (es. 'X: ...', 'X = ...', 'X - ...', 'Per X si intende ...').\n"
        "- Evita intestazioni pure e titoli (es. 'OGGETTO:', 'ALLEGATO', 'ART.' se è solo titolo).\n"
        "- term max 60 caratteri; definition max 1200.\n"
        "- Ogni output deve includere SEMPRE line_no originale.\n"
        "- Se trovi più definizioni per lo stesso term, restituisci una sola (la più informativa).\n"
        f"- Massimo {max_defs} elementi.\n\n"
        'Output SOLO JSON:\n{ "definitions": [ {"term":"...", "definition":"...", "line_no":123} ] }\n\n'
        f"CANDIDATI:\n{json.dumps(candidates, ensure_ascii=False)}"
    )
    raw = parse_json_from_text(call_llm_safe(prompt))
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

        sec_id = _section_id_for_line(doc_id, sections or [], ln_i)

        parsed.append(
            {
                "term": term,
                "definition": definition[:1200],
                "line_no": ln_i,
                "section_id": sec_id,
            }
        )

    parsed = _dedup_keep_best_definition(parsed, max_defs=max_defs)
    return parsed


# ----------------------------
# Legacy-style definition normalization (NO RAG dependency)
# ----------------------------
_ROLE_HINT_RE = re.compile(
    r"(?i)\b(ufficio|responsabile|dirigente|sportello|operatore|richiedente|utente|cittadino|ente|comune|regione|amministrazione|titolare|impresa|azienda)\b"
)


def _build_section_lookup(
    doc_id: str,
    sections: List[Tuple[int, int, str, str]],
) -> Dict[str, Dict[str, str]]:
    """
    section_id -> {"title":..., "path":...}
    """
    out: Dict[str, Dict[str, str]] = {}
    for (a, b, title, path) in sections or []:
        sid = make_section_id(doc_id, a, b, title)
        out[sid] = {"title": title, "path": path}
    return out


def normalize_definitions_legacy(
    *,
    doc_id: str,
    definitions_raw: List[Dict[str, Any]],
    sections: List[Tuple[int, int, str, str]],
    max_terms: int = 60,
) -> List[Dict[str, Any]]:
    """
    Normalizzazione definizioni in stile "vecchio":
    - Non dipende dal retrieval.
    - Sintesi breve via summarize_definition_with_llm.
    - is_agent euristico.
    - Arricchimento con section_title/section_path.
    """
    if not definitions_raw:
        return []

    sec_lookup = _build_section_lookup(doc_id, sections)

    out: List[Dict[str, Any]] = []
    seen = set()

    for d in (definitions_raw or [])[:max_terms]:
        if not isinstance(d, dict):
            continue

        term = str(d.get("term") or "").strip()
        definition = str(d.get("definition") or "").strip()
        section_id = str(d.get("section_id") or "").strip()

        if not term or not definition:
            continue

        k = term.lower()
        if k in seen:
            continue
        seen.add(k)

        summary = ""
        try:
            summary = str(summarize_definition_with_llm(term, definition) or "").strip()
        except Exception:
            summary = term

        is_agent = bool(_ROLE_HINT_RE.search(term) or _ROLE_HINT_RE.search(summary))

        canonical_agent = summary if is_agent and summary else (term if is_agent else "")
        aliases = [term]

        sec_meta = sec_lookup.get(section_id) or {}
        section_title = str(sec_meta.get("title") or "").strip()
        section_path = str(sec_meta.get("path") or "").strip()

        out.append(
            {
                "term": term,
                "normalized": summary[:160],
                "is_agent": is_agent,
                "canonical_agent": canonical_agent[:80],
                "aliases": aliases[:8],
                "status": "ok",
                "section_id": section_id or None,
                "section_title": section_title or "",
                "section_path": section_path or "",
                "line_no": d.get("line_no", None),
            }
        )

    return out


def build_agent_alias_map_from_normalized_definitions(defs_norm: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    alias(lower)->canonical_agent
    """
    out: Dict[str, str] = {}
    if not defs_norm:
        return out

    for d in defs_norm:
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
            if not k:
                continue
            if k not in out:
                out[k] = canonical

    return out


# ----------------------------
# MAIN PIPELINE
# ----------------------------
def run_pipeline_on_text(
    text: str,
    doc_id: str,
    persist_dir: str = "./chroma_db",
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2",
    use_crossdoc_patterns: bool = True,
    update_global_patterns: bool = True,
    reset_doc_collection: bool = False,
    global_patterns_collection: str = "admin_proc_global_patterns_v2_pattern_only",
    output_root: str = "./output",
    generate_diagrams: bool = True,
    evaluation_mode: bool = False,
    enable_local_sim_for_classifier: bool = False,
    disable_crossdoc_read_in_evaluation: bool = True,
    enable_parent_child_retrieval: bool = True,
    top_sections_for_child_retrieval: int = 3,
    definitions_use_semantic_booster: bool = True,
    max_normalized_definition_terms: int = 60,
) -> Dict[str, Any]:
    """
    Flusso:
      1) Titoli/Outline -> sections
      2) Indicizzazione Chroma (line/passages + sections parent)
      3) Definizioni RAW: regex/euristiche + (opz) semantic booster + LLM
      4) Indicizzazione definizioni RAW (kind=definition con section_id)
      5) Definizioni NORMALIZZATE (legacy): sintesi da RAW + mapping section/path
      6) Agent alias map (legacy)
      7) Prefiltro procedurale + Classificazione LLM sezione
      8) Retrieval contesto + Estrazione procedure + Diagrammi
    """
    # Safety: in evaluation non aggiornare pattern-bank e resetta per-doc collection
    if evaluation_mode:
        update_global_patterns = False
        reset_doc_collection = True

    text = normalize_text(text)
    lines = split_into_lines(text)

    client = chromadb.PersistentClient(path=persist_dir)

    # IMPORTANT:
    # - In chroma_utils.get_or_create_collection() il nome viene sanitizzato internamente.
    # - Qui sanitizziamo SOLO per operazioni dirette (delete + logging/stats), così i nomi combaciano.
    raw_doc_collection_name = f"admin_proc_{doc_id}"
    doc_collection_name = chroma_safe_collection_name(raw_doc_collection_name, prefix="doc")

    raw_global_name = global_patterns_collection
    global_collection_name = chroma_safe_collection_name(raw_global_name, prefix="doc")

    if reset_doc_collection:
        try:
            # delete con nome già safe (coerente con get_or_create_collection)
            client.delete_collection(name=doc_collection_name)
        except Exception:
            pass

    # get_or_create_collection sanitizza comunque: passare raw o safe produce lo stesso risultato.
    doc_collection = get_or_create_collection(client, raw_doc_collection_name, embedding_model=embedding_model)
    doc_collection_count_before = int(doc_collection.count()) if doc_collection else 0

    global_collection = None
    if use_crossdoc_patterns:
        global_collection = get_or_create_collection(client, raw_global_name, embedding_model=embedding_model)

    # ---------------------------------------------------------
    # 1) TITLES / OUTLINE
    # ---------------------------------------------------------
    candidates = build_title_candidates(lines, min_score=0.22, max_candidates=900, header_repeat_threshold=3)

    items_all = [
        {
            "line_no": line_no,
            "candidate": _trim_text_chars(cand, 160),
            "heuristic_confidence": sc,
            "context_lines": [
                {"line_no": cl.get("line_no"), "text": _trim_text_chars(str(cl.get("text") or ""), 180)}
                for cl in (get_line_window(lines, line_no, radius=2) or [])
            ],
            "features": {
                "upper_ratio": round(_upper_ratio(cand), 3),
                "starts_with_upper": bool(_starts_with_upper(cand)),
                "ends_with_colon": cand.endswith(":"),
                "has_number_prefix": bool(TITLE_NUM_RE.match(cand)),
                "is_bare_allegato": bool(ALLEGATO_BARE_RE.match(cand)),
                "word_count": len(cand.split()),
            },
        }
        for (line_no, cand, sc) in candidates
    ]
    items_all.sort(key=lambda x: float(x.get("heuristic_confidence", 0.0)), reverse=True)
    items = items_all[:LLM_MAX_ITEMS_FOR_TITLES]

    cross_title_examples: List[dict] = []
    can_read_crossdoc = bool(global_collection is not None and candidates)
    if evaluation_mode and disable_crossdoc_read_in_evaluation:
        can_read_crossdoc = False
    if can_read_crossdoc:
        _ = retrieve_ranked_chunks_with_meta(
            global_collection,
            query=" \n".join([c[1] for c in candidates[:15]])[:700],
            k=8,
            where={"label": "title_pattern"},
            include_distances=False,
        )
        # pattern-only nel tuo progetto: esempi non usati direttamente
        cross_title_examples = []

    p_titles = prompt_classify_titles(items, PromptBlocks(crossdoc_examples=cross_title_examples))
    title_nodes_raw = parse_json_from_text(call_llm_safe(p_titles))

    title_nodes: List[TitleNode] = []
    for t in title_nodes_raw.get("titles", []) or []:
        if t.get("is_title") is True:
            clean = (t.get("clean_title") or "").strip()
            if not clean:
                continue
            title_nodes.append(
                TitleNode(
                    title=clean,
                    level=int(t.get("level", 2)),
                    start_line=int(t.get("line_no")),
                )
            )

    title_nodes.sort(key=lambda n: n.start_line)
    title_nodes = build_outline_hierarchy(title_nodes)

    sections: List[Tuple[int, int, str, str]] = []
    for i, n in enumerate(title_nodes):
        start = n.start_line
        end = n.end_line if n.end_line is not None else (len(lines) - 1)
        path = outline_path(title_nodes, i)
        sections.append((start, end, n.title, path))
    if not sections:
        sections = [(0, len(lines) - 1, "DOCUMENTO", "DOCUMENTO")]

    # ---------------------------------------------------------
    # 2) Indicizzazione: line + passage + section (parent)
    # ---------------------------------------------------------
    line_docs, line_metas, line_ids = build_line_chunks(lines, doc_id, sections=sections)
    pass_docs, pass_metas, pass_ids = build_passage_chunks(lines, doc_id, sections=sections)

    doc_ids_all = line_ids + pass_ids
    doc_texts_all = line_docs + pass_docs
    doc_metas_all = line_metas + pass_metas

    added_now = safe_add(doc_collection, ids=doc_ids_all, documents=doc_texts_all, metadatas=doc_metas_all)

    sections_added = add_sections_to_doc_collection(
        doc_collection=doc_collection,
        doc_id=doc_id,
        sections=sections,
        lines=lines,
        max_chars=1200,
    )

    # ---------------------------------------------------------
    # 3) DEFINIZIONI RAW: HYBRID
    # ---------------------------------------------------------
    doc_definitions_raw = extract_definitions_hybrid(
        lines=lines,
        doc_collection=doc_collection,
        doc_id=doc_id,
        sections=sections,
        max_defs=140,
        max_candidates=260,
        use_semantic_booster=bool(definitions_use_semantic_booster),
    )

    # ---------------------------------------------------------
    # 4) Indicizza definizioni RAW come kind="definition"
    # ---------------------------------------------------------
    defs_added = 0
    try:
        defs_added = retrieve_definition_support_for_agents(
            doc_collection=doc_collection,
            doc_id=doc_id,
            definitions=doc_definitions_raw,
            sections=sections,
            mode="index_only",
        ).get("defs_added", 0)
    except Exception:
        defs_added = 0

    # ---------------------------------------------------------
    # 5) DEFINIZIONI NORMALIZZATE (LEGACY)
    # ---------------------------------------------------------
    doc_definitions_norm = normalize_definitions_legacy(
        doc_id=doc_id,
        definitions_raw=doc_definitions_raw,
        sections=sections,
        max_terms=int(max_normalized_definition_terms),
    )

    # ---------------------------------------------------------
    # 6) Allowed agents + alias map
    # ---------------------------------------------------------
    allowed_agents = build_allowed_agents_from_definitions(doc_definitions_raw)
    agent_alias_map = build_agent_alias_map_from_normalized_definitions(doc_definitions_norm)

    # ---------------------------------------------------------
    # 7) Iterate sections (prefiltro + classify + extract)
    # ---------------------------------------------------------
    procedures: List[ProcedureRecord] = []
    diagram_procedures: List[Dict[str, Any]] = []
    procedural_flags_by_section: Dict[Tuple[int, int, str], bool] = {}

    for sec_idx, (a, b, title, path) in enumerate(sections):
        section_text = "\n".join(lines[a : b + 1]).strip()
        if not section_text:
            continue

        heur = procedural_signal_score(section_text)
        title_hint = bool(
            re.search(r"\b(procedur|modalità|istruzioni|iter|compiti|attivit|verific)\b", title or "", re.IGNORECASE)
        )
        if heur < 0.20 and not title_hint:
            procedural_flags_by_section[(a, b, title)] = False
            continue

        section_text_llm = _trim_section_for_llm(section_text, LLM_MAX_SECTION_CHARS)

        local_similar_sections: List[str] = []
        if enable_local_sim_for_classifier:
            local_hits = retrieve_ranked_chunks_with_meta(
                doc_collection,
                query=section_text_llm[:700],
                k=4,
                where={"doc_id": doc_id, "kind": "passage"},
                include_distances=False,
            )
            local_similar_sections = [h.get("text", "") for h in local_hits if h.get("text")]

        cross_examples_for_cls: List[dict] = []
        can_read_crossdoc2 = bool(global_collection is not None)
        if evaluation_mode and disable_crossdoc_read_in_evaluation:
            can_read_crossdoc2 = False
        if can_read_crossdoc2:
            try:
                _ = retrieve_ranked_chunks_with_meta(
                    global_collection,
                    query=(title + "\n" + section_text_llm[:450])[:700],
                    k=4,
                    where={"label": "procedural_section_pattern"},
                    include_distances=False,
                )
                _ = retrieve_ranked_chunks_with_meta(
                    global_collection,
                    query=(title + "\n" + section_text_llm[:450])[:700],
                    k=4,
                    where={"label": "nonprocedural_section_pattern"},
                    include_distances=False,
                )
                cross_examples_for_cls = []
            except Exception:
                cross_examples_for_cls = []

        p_isproc = prompt_is_procedural_section(
            section_title=title,
            section_path=path,
            section_text=section_text_llm,
            blocks=PromptBlocks(
                crossdoc_examples=cross_examples_for_cls if cross_examples_for_cls else None,
                local_similar_sections=local_similar_sections if enable_local_sim_for_classifier else None,
            ),
        )
        cls = parse_json_from_text(call_llm_safe(p_isproc))
        is_proc = bool(cls.get("is_procedure", False))
        procedural_flags_by_section[(a, b, title)] = is_proc
        if not is_proc:
            continue

        # ---------------------------------------------------------
        # SUPPORT CONTEXT: parent/child + definizioni
        # ---------------------------------------------------------
        support_query = build_support_query(title)

        if enable_parent_child_retrieval:
            kps_pass = _adaptive_k(len(lines), 6, k_min=3, k_max=6)
            pass_hits = retrieve_child_chunks_in_top_sections(
                doc_collection=doc_collection,
                doc_id=doc_id,
                query=support_query,
                child_kind="passage",
                top_sections=top_sections_for_child_retrieval,
                k_per_section=kps_pass,
                include_distances=False,
            )
        else:
            k_pass = _adaptive_k(len(lines), 16, k_min=8, k_max=16)
            pass_hits = retrieve_ranked_chunks_with_meta(
                doc_collection,
                query=support_query,
                k=k_pass,
                where={"doc_id": doc_id, "kind": "passage"},
                include_distances=False,
            )

        pass_hits = [h for h in (pass_hits or []) if is_outside_section(h, a, b)]

        # definizioni agent-aware
        agent_support_blocks: List[dict] = []
        try:
            kps_def_support = _adaptive_k(len(lines), 4, k_min=2, k_max=4)
            agent_support_blocks = retrieve_definition_support_for_agents(
                doc_collection=doc_collection,
                doc_id=doc_id,
                definitions=doc_definitions_raw,
                sections=sections,
                mode="retrieve",
                section_text=section_text_llm,
                allowed_agents=allowed_agents,
                top_sections=top_sections_for_child_retrieval,
                k_per_section=kps_def_support,
            ).get("blocks", [])
        except Exception:
            agent_support_blocks = []

        def_hits = []
        if not agent_support_blocks:
            if enable_parent_child_retrieval:
                kps_def = _adaptive_k(len(lines), 4, k_min=2, k_max=4)
                def_hits = retrieve_child_chunks_in_top_sections(
                    doc_collection=doc_collection,
                    doc_id=doc_id,
                    query=support_query,
                    child_kind="definition",
                    top_sections=top_sections_for_child_retrieval,
                    k_per_section=kps_def,
                    include_distances=False,
                )
            else:
                k_def = _adaptive_k(len(lines), 8, k_min=4, k_max=8)
                def_hits = retrieve_ranked_chunks_with_meta(
                    doc_collection,
                    query=support_query,
                    k=k_def,
                    where={"doc_id": doc_id, "kind": "definition"},
                    include_distances=False,
                )

        context_blocks: List[dict] = []

        for cb in agent_support_blocks[:8]:
            if cb.get("text"):
                context_blocks.append(cb)

        def_seen_terms = set()
        if len(context_blocks) < 8:
            for idx, h in enumerate(def_hits or []):
                meta = h.get("metadata", {}) or {}
                term_l = (meta.get("term") or "").strip().lower()
                if term_l and term_l in def_seen_terms:
                    continue
                if term_l:
                    def_seen_terms.add(term_l)

                text_def = (h.get("text", "") or "").strip()
                if not text_def:
                    continue

                evidence_id = f"{doc_collection_name}_definition_{meta.get('term', idx)}"
                context_blocks.append(
                    {
                        "evidence_id": evidence_id,
                        "kind": "definition",
                        "text": text_def[:900],
                        "meta": {"term": meta.get("term", ""), "section_id": meta.get("section_id", None)},
                    }
                )
                if len(context_blocks) >= 8:
                    break

        for idx, h in enumerate(pass_hits or []):
            meta = h.get("metadata", {}) or {}
            text_p = (h.get("text") or "").strip()
            if not text_p:
                continue
            chunk_id = meta.get("chunk_id", idx)
            evidence_id = f"{doc_collection_name}_passage_{chunk_id}"
            context_blocks.append(
                {
                    "evidence_id": evidence_id,
                    "kind": "passage",
                    "text": text_p[:900],
                    "meta": {"section_id": meta.get("section_id", None)},
                }
            )
            if len(context_blocks) >= 12:
                break

        context_blocks_llm = _shrink_context_blocks_for_llm(context_blocks, max_blocks=12, max_block_text_chars=650)
        ctx_json = _safe_json_dumps(context_blocks_llm, max_chars=LLM_MAX_JSONBLOCKS_CHARS)

        # A) output sintetico
        prompt_min = (
            "Estrai UNA procedura come lista di step (testo breve) in ITALIANO.\n"
            "Vincoli: usa SOLO la sezione + contesto evidenziale per-doc. Non inventare.\n"
            "Output SOLO JSON:\n"
            '{ "status": "complete"|"partial", "steps": [str], "notes": [str] }\n\n'
            f"TITOLO: {title}\nPATH: {path}\n\n"
            "SEZIONE:\n"
            f"{section_text_llm}\n\n"
            "CONTESTO (definizioni/passaggi):\n"
            f"{ctx_json}"
        )
        extr_min = parse_json_from_text(call_llm_safe(prompt_min))

        procedures.append(
            ProcedureRecord(
                section_title=title,
                section_path=path,
                start_line=a,
                end_line=b,
                status=str(extr_min.get("status", "partial")),
                steps=list(extr_min.get("steps", []) or []),
                notes=list(extr_min.get("notes", []) or []),
                evidence_chunk_ids=[cb.get("evidence_id") for cb in context_blocks_llm[:12] if cb.get("evidence_id")],
            )
        )

        # B) diagram-ready
        procedure_id = f"{doc_id}_proc_{sec_idx:03d}"
        subsection_title = title

        p_steps = prompt_extract_diagram_steps(
            procedure_id=procedure_id,
            section_title=title,
            subsection_title=subsection_title,
            section_path=path,
            section_text=section_text_llm,
            blocks=PromptBlocks(
                rag_context_blocks=context_blocks_llm,
                definitions=doc_definitions_raw[:120],
                allowed_agents=allowed_agents,
                supporting_actions=None,
                crossdoc_examples=None,
            ),
        )
        extr_steps = parse_json_from_text(call_llm_safe(p_steps))

        raw_steps = _coerce_steps_schema(extr_steps.get("steps"))
        linked_steps = link_steps_with_branches(raw_steps)

        # Grounding agente usando alias_map legacy
        for s in linked_steps:
            s["agent"] = ground_agent(s.get("agent", "Operatore"), agent_alias_map)

        supporting_evidence = {
            "definitions": [
                cb.get("evidence_id")
                for cb in context_blocks_llm
                if cb.get("kind") == "definition" and cb.get("evidence_id")
            ],
            "other_passages": [
                cb.get("evidence_id")
                for cb in context_blocks_llm
                if cb.get("kind") == "passage" and cb.get("evidence_id")
            ],
        }

        diagram_procedures.append(
            {
                "procedure_id": procedure_id,
                "section_title": title,
                "subsection_title": subsection_title,
                "section_path": path,
                "status": str(extr_steps.get("status", "partial")),
                "notes": list(extr_steps.get("notes", []) or []),
                "original_text_full": section_text,
                "steps": linked_steps,
                "points": [
                    {"point_label": str(i + 1), "original_text": s.get("description_synthetic", ""), "steps": [s]}
                    for i, s in enumerate(linked_steps)
                ],
                "supporting_evidence": supporting_evidence,
            }
        )

    # ---------------------------------------------------------
    # OUTPUT saving
    # ---------------------------------------------------------
    doc_dir = os.path.join(output_root, doc_id)
    os.makedirs(doc_dir, exist_ok=True)

    out_paths = {
        "doc_dir": doc_dir,
        "diagram_procedures_json": os.path.join(doc_dir, f"{doc_id}.diagram_procedures.json"),
        "result_json": os.path.join(doc_dir, f"{doc_id}.rag.result.json"),
        "definitions_raw_json": os.path.join(doc_dir, f"{doc_id}.definitions.raw.json"),
        "definitions_norm_json": os.path.join(doc_dir, f"{doc_id}.definitions.normalized.json"),
    }

    _safe_write_json(out_paths["definitions_raw_json"], doc_definitions_raw)
    _safe_write_json(out_paths["definitions_norm_json"], doc_definitions_norm)
    _safe_write_json(out_paths["diagram_procedures_json"], diagram_procedures)

    result = {
        "doc_id": doc_id,
        "outline": [asdict(n) for n in title_nodes],
        "sections": [
            {"start": a, "end": b, "title": t, "path": p, "section_id": make_section_id(doc_id, a, b, t)}
            for (a, b, t, p) in sections
        ],
        "procedures": [asdict(p) for p in procedures],
        "diagram_procedures": diagram_procedures,
        "definitions_raw": doc_definitions_raw,
        "definitions_normalized": doc_definitions_norm,
        "stats": {
            "n_lines": len(lines),
            "n_titles": len(title_nodes),
            "n_sections": len(sections),
            "n_procedures": len(procedures),
            "n_diagram_procedures": len(diagram_procedures),
            "doc_collection_added_now": int(added_now),
            "sections_added_now": int(sections_added),
            "doc_definitions_added_now": int(defs_added),
            "doc_collection_count_before": int(doc_collection_count_before),
            "doc_collection_count": int(doc_collection.count()) if doc_collection else 0,
            "doc_collection_reset": bool(reset_doc_collection),
            "doc_collection_raw": raw_doc_collection_name,
            "doc_collection": doc_collection_name,
            "global_collection_raw": raw_global_name,
            "global_collection": global_collection_name,
            "evaluation_mode": bool(evaluation_mode),
            "use_crossdoc_patterns": bool(use_crossdoc_patterns),
            "update_global_patterns": bool(update_global_patterns),
            "enable_parent_child_retrieval": bool(enable_parent_child_retrieval),
            "top_sections_for_child_retrieval": int(top_sections_for_child_retrieval),
            "definitions_use_semantic_booster": bool(definitions_use_semantic_booster),
            "max_normalized_definition_terms": int(max_normalized_definition_terms),
            "n_definitions_raw": len(doc_definitions_raw),
            "n_definitions_normalized": len(doc_definitions_norm),
            "n_agent_aliases": len(build_agent_alias_map_from_normalized_definitions(doc_definitions_norm)),
        },
        "outputs": out_paths,
    }
    _safe_write_json(out_paths["result_json"], result)

    if generate_diagrams and generate_three_diagram_sets is not None and diagram_procedures:
        generate_three_diagram_sets(
            diagram_procedures_json_path=out_paths["diagram_procedures_json"],
            out_root=output_root,
            doc_stem=doc_id,
        )

    return result
