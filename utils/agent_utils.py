# utils/agent_utils.py
from __future__ import annotations
import re
from typing import Dict, List, Optional

BANNED_EN = {"operator", "citizen", "user", "applicant", "clerk"}

CANON_MAP = {
    "operator": "Operatore",
    "operatore": "Operatore",
    "citizen": "Cittadino",
    "cittadino": "Cittadino",
    "utente": "Richiedente",
    "user": "Richiedente",
    "applicant": "Richiedente",
    "richiedente": "Richiedente",
}

def normalize_agent_name(agent: Optional[str]) -> str:
    a = (agent or "").strip()
    if not a:
        return "Operatore"
    low = re.sub(r"\s+", " ", a.lower()).strip()
    if low in CANON_MAP:
        return CANON_MAP[low]
    if low in BANNED_EN:
        return "Operatore"
    # capitalizza pulito
    return a[:1].upper() + a[1:]


def build_agent_alias_map_from_definitions(defs: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Crea mapping alias->ruolo canonico, usando definizioni tipo:
      "Operatore = titolare d'impresa"
      "SUAP: Sportello Unico per ..."
    Output: {"suap": "SUAP (Sportello Unico ...)", "operatore":"Titolare d'impresa", ...}
    """
    out: Dict[str, str] = {}
    if not defs:
        return out

    for d in defs:
        term = (d.get("term") or "").strip()
        definition = (d.get("definition") or "").strip()
        if not term or not definition:
            continue

        t_low = term.lower()

        # caso: ACRONIMO -> espansione
        if re.fullmatch(r"[A-Z0-9]{2,12}", term) and len(definition) >= 8:
            out[t_low] = f"{term} ({definition[:80].rstrip('.')})"
            continue

        # caso: Operatore = titolare d'impresa
        if t_low in {"operatore", "richiedente", "utente", "cittadino"}:
            # se la definizione sembra un ruolo concreto, usa la definizione come canonico
            if len(definition) <= 80:
                out[t_low] = definition[:80].strip().rstrip(".")
            else:
                out[t_low] = definition[:80].strip().rstrip(".")
            continue

    return out


def ground_agent(agent: str, agent_alias_map: Dict[str, str]) -> str:
    """
    Applica normalizzazione base + sostituzioni grounding da definizioni.
    """
    base = normalize_agent_name(agent)
    low = base.lower()

    # se la base è una key nota in alias map, sostituisci
    if agent_alias_map and low in agent_alias_map:
        return agent_alias_map[low]

    # se l'agent è un acronimo e sta in alias map
    if agent_alias_map and re.fullmatch(r"[A-Z0-9]{2,12}", base):
        k = base.lower()
        if k in agent_alias_map:
            return agent_alias_map[k]

    return base
