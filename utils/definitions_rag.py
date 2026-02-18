# utils/definitions_rag.py
from __future__ import annotations
import re
import json
from typing import Any, Dict, List, Tuple, Optional

DEF_PATTERNS = [
    # "SUAP: Sportello Unico ..."
    re.compile(r"^\s*([A-Z][A-Z0-9\/\-\._ ]{1,40})\s*[:\-]\s*(.{10,250})\s*$"),
    # "Per 'X' si intende ..."
    re.compile(r"(?i)\bper\s+[\"']?(.{2,60}?)[\"']?\s+si\s+intende\s+(.{10,250})"),
    # "X = ..."
    re.compile(r"^\s*([A-Za-zÀ-ÖØ-öø-ÿ0-9\/\-\._ ]{2,60})\s*=\s*(.{10,250})\s*$"),
    #per catturare definizioni tipo "Operatore: il titolare dell'impresa che presenta la domanda" o "Operatore = il titolare dell'impresa che presenta la domanda"
    re.compile(r"^\s*([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ0-9\/\-\._ ]{1,60})\s*[:\-]\s*(.{5,1200})\s*$")

]

STOP_TERMS = {
    "art", "articolo", "comma", "capitolo", "sezione", "allegato", "tabella",
}

def _clean_term(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(" :;-–—\t\r\n")
    return t

def _clean_def(d: str) -> str:
    d = (d or "").strip()
    d = re.sub(r"\s+", " ", d)
    return d.strip()

def extract_definitions_from_lines(lines: List[str], max_defs: int = 120) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()

    for ln in lines or []:
        s = (ln or "").strip()
        if not s or len(s) < 12:
            continue

        for pat in DEF_PATTERNS:
            m = pat.search(s)
            if not m:
                continue

            term = _clean_term(m.group(1))
            definition = _clean_def(m.group(2))

            if len(term) < 2 or len(term) > 60:
                continue
            low = term.lower()
            if low in STOP_TERMS:
                continue
            if len(definition) < 8:
                continue

            key = (low, definition.lower()[:80])
            if key in seen:
                continue
            seen.add(key)

            out.append({"term": term, "definition": definition})
            if len(out) >= max_defs:
                return out

    return out

def build_allowed_agents_from_definitions(defs: List[Dict[str, str]]) -> List[str]:
    """
    Estrae agent candidate: termini che sembrano ruoli/uffici.
    Heuristica semplice: include termini con parole tipo Ufficio/Responsabile/Comune/Richiedente/Operatore, ecc.
    """
    if not defs:
        return []
    role_kw = re.compile(r"(?i)\b(ufficio|responsabile|comune|regione|richiedente|utente|cittadino|operatore|sportello|dirigente|procedimento)\b")
    agents = []
    seen = set()
    for d in defs:
        term = (d.get("term") or "").strip()
        if not term:
            continue
        if role_kw.search(term) or role_kw.search(d.get("definition") or ""):
            k = term.lower()
            if k not in seen:
                seen.add(k)
                agents.append(term)
    return agents[:40]

def match_definition(term: str, defs: List[Dict[str, str]]) -> Optional[str]:
    if not term or not defs:
        return None
    low = term.strip().lower()
    for d in defs:
        if (d.get("term") or "").strip().lower() == low:
            return d.get("definition")
    return None
# ----------------------------
# RETRIEVAL UTIL
# ----------------------------
# Scarta i chunk che cadono dentro la sezione corrente [a,b] per evitare il self-retrieval
def is_outside_section(hit: Dict[str, Any], a: int, b: int) -> bool:
    """
    Ritorna True se il chunk NON interseca la sezione [a,b].
    Serve per evitare self-retrieval.
    """
    meta = hit.get("metadata", {}) or {}
    ls = meta.get("line_start")
    le = meta.get("line_end")

    # Se non ho info di range, non blocco
    if ls is None or le is None:
        return True

    # fuori sezione se non interseca [a,b]
    return (le < a) or (ls > b)

# Costruisce una query non identica al testo della sezione per cercare definizioni, responsabilità ecc
def build_support_query(title: str) -> str:
    """
    Query semantica per cercare contesto di supporto:
    definizioni, vincoli, prerequisiti.
    """
    return (
        f"{title}\n"
        "definizioni requisiti prerequisiti responsabilità ruoli\n"
        "obbligatorio deve devono entro eccezione esclusione documento"
    )
