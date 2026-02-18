# pipeline/procedural_no_rag_guided_llm.py
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
    prompt_classify_titles,
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
    start_line: int
    end_line: Optional[int] = None
    parent_index: Optional[int] = None


# ----------------------------
# Helpers
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


# ----------------------------
# Heuristics (robuste, deterministic)
# ----------------------------

TITLE_NUM_RE = re.compile(r"^\s*(\d+(\.\d+){0,4})\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*([-\u2022\*]|\(?[a-zA-Z0-9]+\)|[0-9]+\.)\s+")
DEONTIC_RE = re.compile(r"\b(deve|devono|è tenuto|è obbligato|obbligatorio|entro)\b", re.IGNORECASE)

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
    if _upper_ratio(s) > 0.6:
        score += 0.18
    if s.endswith(":"):
        score += 0.18
    if 1 <= len(s.split()) <= 10:
        score += 0.10

    if re.search(r"\b(è|sono|deve|devono|può|possono|viene|vengono)\b", s, re.IGNORECASE):
        score -= 0.15
    if s.endswith(".") and len(s.split()) > 10:
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
    if re.search(r"\b(prima|poi|successivamente|infine|entro)\b", t, re.IGNORECASE):
        score += 0.15
    return min(1.0, score)


# ----------------------------
# Outline utilities
# ----------------------------

def get_line_window(lines: List[str], center: int, radius: int = 2) -> List[Dict[str, Any]]:
    out = []
    for ln in range(max(0, center - radius), min(len(lines), center + radius + 1)):
        out.append({"line_no": ln, "text": lines[ln]})
    return out


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
    """
    Normalizzazione senza retrieval:
    - evidenza = definition raw (kind="definition") per il singolo termine
    - LLM produce normalized + is_agent + canonical_agent + aliases
    """
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
# GUIDED NO-RAG PIPELINE
# ----------------------------

def run_no_rag_guided_pipeline_on_text(
    text: str,
    doc_id: str,
    output_root: str = "./output",
) -> Dict[str, Any]:
    """
    NO-RAG guided (allineato alla struttura nuova, ma senza retrieval):
    - euristiche per candidati titoli
    - outline (LLM) + gerarchia
    - split sezioni da outline
    - definizioni: regex/euristiche su linee + LLM su pochi candidati + normalizzazione LLM (no retrieval)
    - LLM decide procedural/non-procedural per sezione
    - LLM estrae steps per sezioni procedural
    - NESSUN retrieval: rag_context_blocks=[] e crossdoc_examples=[]
    """
    text = normalize_text(text)
    lines = split_into_lines(text)

    # (0) DEFINIZIONI: Hybrid + normalize (no retrieval)
    doc_definitions_raw = extract_definitions_hybrid_no_rag(lines=lines, max_defs=140, max_candidates=260)
    doc_definitions_norm = normalize_definitions_no_rag_via_llm(doc_definitions_raw, max_terms=60)
    agent_alias_map = build_agent_alias_map_from_normalized_definitions(doc_definitions_norm)
    allowed_agents = build_allowed_agents_from_definitions(doc_definitions_raw)

    # (1) candidates titoli
    candidates = build_title_candidates(
        lines,
        min_score=0.22,
        max_candidates=900,
        header_repeat_threshold=3,
    )

    items = [
        {
            "line_no": line_no,
            "candidate": cand,
            "heuristic_confidence": sc,
            "context_lines": get_line_window(lines, line_no, radius=2),
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

    # (2) titles con LLM
    p_titles = prompt_classify_titles(items, PromptBlocks(crossdoc_examples=[]))
    raw_titles = parse_json_from_text(call_llm(p_titles))

    title_nodes: List[TitleNode] = []
    for t in raw_titles.get("titles", []) or []:
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

    # (3) sections da outline
    sections: List[Tuple[int, int, str, str]] = []
    for i, n in enumerate(title_nodes):
        start = n.start_line
        end = n.end_line if n.end_line is not None else (len(lines) - 1)
        path = outline_path(title_nodes, i)
        sections.append((start, end, n.title, path))

    if not sections:
        sections = [(0, len(lines) - 1, "DOCUMENTO", "DOCUMENTO")]

    sections_flags: List[Dict[str, Any]] = []
    diagram_procedures: List[Dict[str, Any]] = []

    # (4) iterazione sezioni
    for sec_idx, (a, b, title, path) in enumerate(sections):
        section_text = "\n".join(lines[a : b + 1]).strip()
        if not section_text:
            continue

        # prefilter: evita chiamate inutili al LLM su sezioni chiaramente non procedurali
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
                    "start_line": a,
                    "end_line": b,
                    "is_procedure": False,
                    "confidence": 0.0,
                    "reasons": ["prefilter_heuristic=false"],
                }
            )
            continue

        # (4.1) classificazione procedura
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
                "start_line": a,
                "end_line": b,
                "is_procedure": is_proc,
                "confidence": float(cls.get("confidence", 0.0) or 0.0),
                "reasons": list(cls.get("reasons", []) or []),
                "expected_missing_parts": list(cls.get("expected_missing_parts", []) or []),
            }
        )

        if not is_proc:
            continue

        # (4.2) estrazione steps (diagram-ready) - NO RAG
        procedure_id = f"{doc_id}_guided_proc_{sec_idx:03d}"
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
        extr_steps = parse_json_from_text(call_llm(p_steps))

        raw_steps = _coerce_steps_schema(extr_steps.get("steps"))
        linked_steps = link_steps_with_branches(raw_steps)

        # grounding agent finale (alias map normalizzata)
        for s in linked_steps:
            s["agent"] = ground_agent(s.get("agent", "Operatore"), agent_alias_map)

        diagram_procedures.append(
            {
                "procedure_id": procedure_id,
                "section_title": title,
                "subsection_title": title,
                "section_path": path,
                "status": str(extr_steps.get("status", "partial")),
                "notes": list(extr_steps.get("notes", []) or []),
                "original_text_full": section_text,
                "steps": linked_steps,
                "points": [
                    {"point_label": str(i + 1), "original_text": s.get("description_synthetic", ""), "steps": [s]}
                    for i, s in enumerate(linked_steps)
                ],
            }
        )

    # (5) save
    out_dir = os.path.join(output_root, doc_id)
    os.makedirs(out_dir, exist_ok=True)

    out_paths = {
        "result_json": os.path.join(out_dir, f"{doc_id}.no_rag_guided.result.json"),
        "diagram_procedures_json": os.path.join(out_dir, f"{doc_id}.diagram_procedures.json"),
        "definitions_raw_json": os.path.join(out_dir, f"{doc_id}.definitions.raw.json"),
        "definitions_norm_json": os.path.join(out_dir, f"{doc_id}.definitions.normalized.json"),
    }

    _safe_write_json(out_paths["definitions_raw_json"], doc_definitions_raw)
    _safe_write_json(out_paths["definitions_norm_json"], doc_definitions_norm)
    _safe_write_json(out_paths["diagram_procedures_json"], diagram_procedures)

    result = {
        "doc_id": doc_id,
        "system_variant": "no_rag_guided",
        "outline": [asdict(n) for n in title_nodes],
        "sections": [{"start_line": a, "end_line": b, "title": t, "path": p} for (a, b, t, p) in sections],
        "sections_procedural_flags": sections_flags,
        "diagram_procedures": diagram_procedures,
        "definitions_raw": doc_definitions_raw,
        "definitions_normalized": doc_definitions_norm,
        "stats": {
            "n_lines": len(lines),
            "n_titles": len(title_nodes),
            "n_sections": len(sections),
            "n_proc_sections": sum(1 for x in sections_flags if x.get("is_procedure")),
            "n_diagram_procedures": len(diagram_procedures),
            "title_candidates": len(candidates),
            "n_definitions_raw": len(doc_definitions_raw),
            "n_definitions_normalized": len(doc_definitions_norm),
            "n_agent_aliases": len(agent_alias_map),
        },
        "outputs": out_paths,
    }
    _safe_write_json(out_paths["result_json"], result)
    return result
