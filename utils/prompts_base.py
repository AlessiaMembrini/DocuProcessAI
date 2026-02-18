# utils/prompts_base.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Optional


def _j(obj: Any, n: int) -> str:
    """Dump JSON con taglio (stabile) per non gonfiare prompt."""
    if not obj:
        return "[]"
    if isinstance(obj, list):
        obj = obj[:n]
    return json.dumps(obj, ensure_ascii=False)


@dataclass
class PromptBlocks:
    # cross-doc: SOLO pattern (feature), mai testo contenutistico
    crossdoc_examples: Optional[List[dict]] = None
    # (attenzione: può introdurre bias nel classificatore)
    local_similar_sections: Optional[List[str]] = None
    # contesto evidenziale (per-doc)
    rag_context_blocks: Optional[List[dict]] = None
    # definizioni per-doc
    definitions: Optional[List[dict]] = None
    allowed_agents: Optional[List[str]] = None
    supporting_actions: Optional[List[dict]] = None  # [{"action_id","label","action_text",...}]


# ----------------------------
# TITLES
# ----------------------------

def prompt_classify_titles(items: List[dict], blocks: Optional[PromptBlocks] = None) -> str:
    cross = ""
    if blocks and blocks.crossdoc_examples:
        cross = (
            "ESEMPI CROSS-DOCUMENT (SOLO PATTERN/FEATURE, NO TESTO):\n"
            "Ogni elemento contiene label/hint/pattern.\n"
            f"{_j(blocks.crossdoc_examples, 4)}\n\n"
        )

    return (
        "Task: classificare righe candidate come TITOLO di sezione.\n"
        "Regole forti (riduci falsi positivi):\n"
        "- NON titolo se bullet/step/azione ( '-', '•', '1.', '(a)' ... ).\n"
        "- NON titolo se frase lunga o discorsiva (tipico > 12 parole) o con virgole.\n"
        "- NON titolo se header/footer ripetuto/pagina.\n"
        "- 'ALLEGATO X' da solo NON è titolo, salvo introduca subito un heading.\n"
        "- Se dubbio: is_title=false.\n\n"
        "Se is_title=true:\n"
        "- level 1..6 (1 macro, 2 sezione, 3 sottosezione...)\n"
        "- clean_title: rimuovi numerazione iniziale ('1.2', 'Art. 5', 'CAPO II', 'SEZIONE I'), rimuovi ':' finale.\n"
        "- Non inventare parole.\n\n"
        "Output: SOLO JSON valido con chiave 'titles'. Un oggetto per ITEM, stesso line_no.\n"
        f"{cross}"
        "Formato:\n"
        "{ \"titles\": [ {\"line_no\":int, \"is_title\":bool, \"level\":int, \"clean_title\":str}, ... ] }\n\n"
        "ITEMS:\n"
        f"{json.dumps(items, ensure_ascii=False)}"
    )


# ----------------------------
# IS PROCEDURAL
# ----------------------------

def prompt_is_procedural_section(
    section_title: str,
    section_path: str,
    section_text: str,
    blocks: Optional[PromptBlocks] = None,
) -> str:
    """
    Classificatore con cross-doc pattern-only:
    - esempi cross-doc contengono SOLO feature (pattern), mai testo.
    - local_similar_sections (se passato) è dichiarato non autorevole.
    """
    cross_proc = ""
    cross_non = ""
    if blocks and blocks.crossdoc_examples:
        proc = [x for x in blocks.crossdoc_examples if x.get("label") == "procedural_section_pattern"]
        nonp = [x for x in blocks.crossdoc_examples if x.get("label") == "nonprocedural_section_pattern"]
        cross_proc = f"ESEMPI CROSS-DOC (pattern-only) PROCEDURALI:\n{_j(proc, 3)}\n\n" if proc else ""
        cross_non = f"ESEMPI CROSS-DOC (pattern-only) NON PROCEDURALI:\n{_j(nonp, 3)}\n\n" if nonp else ""

    local_sim = ""
    if blocks and blocks.local_similar_sections:
        sims = [s.strip() for s in blocks.local_similar_sections if isinstance(s, str) and s.strip()]
        sims = sims[:2]
        if sims:
            local_sim = (
                "NOTE (non autorevole): estratti di sezioni simili nello stesso documento (solo per lessico, NON per decidere):\n"
                f"{_j(sims, 2)}\n\n"
            )

    return (
        "Decidi se la SEZIONE è una PROCEDURA OPERATIVA.\n\n"
        "Definizione operativa:\n"
        "- PROCEDURA = nel testo si possono ricavare almeno 2 azioni operative distinte (verbo+oggetto),\n"
        "  ordinabili in un flusso (anche implicito), con agente esplicito o inferibile.\n"
        "- NON PROCEDURA = definizioni, descrizioni, normativa, requisiti senza flusso di azioni.\n\n"
        "Vincoli:\n"
        "- Usa SOLO il testo sezione.\n"
        "- Non inventare azioni.\n"
        "- Gli esempi cross-document sono SOLO pattern/feature (non contengono testo): usali come guida formale, non come contenuto.\n\n"
        "Output: SOLO JSON valido:\n"
        "{\n"
        "  \"is_procedure\": bool,\n"
        "  \"confidence\": float,\n"
        "  \"action_candidates\": [ {\"agent\": str, \"verb\": str, \"object\": str, \"evidence\": str} ],\n"
        "  \"reasons\": [str]\n"
        "}\n\n"
        f"{cross_proc}{cross_non}{local_sim}"
        f"TITOLO: {section_title}\n"
        f"PATH: {section_path}\n\n"
        "TESTO SEZIONE:\n"
        f"{section_text}\n"
    )


# ----------------------------
# EXTRACT STEPS (diagram-ready)
# ----------------------------

def prompt_extract_diagram_steps(
    procedure_id: str,
    section_title: str,
    subsection_title: str,
    section_path: str,
    section_text: str,
    blocks: Optional[PromptBlocks] = None,
) -> str:
    """
    Estrazione step rigidissima:
    - cross-doc: pattern-only (feature), mai contenuto
    - supporting_actions: consente ref=action:Ax per rimandi interni
    """
    cross_steps = ""
    defs = ""
    ctx = ""
    allowed_agents = ""
    support_actions = ""

    if blocks and blocks.crossdoc_examples:
        cross_steps = (
            "ESEMPI CROSS-DOCUMENT (SOLO PATTERN/FEATURE, NO TESTO):\n"
            f"{_j(blocks.crossdoc_examples, 3)}\n\n"
        )

    if blocks and blocks.definitions:
        defs = "DEFINIZIONI (documento corrente):\n" + _j(blocks.definitions, 16) + "\n\n"

    if blocks and blocks.allowed_agents:
        allowed_agents = (
            "AGENTI AMMESSI (usa SOLO se coerente col testo; altrimenti 'Operatore'):\n"
            f"{_j(blocks.allowed_agents, 24)}\n\n"
        )

    if blocks and blocks.supporting_actions:
        support_actions = (
            "AZIONI RIUSABILI (supporting_actions):\n"
            f"{_j(blocks.supporting_actions, 8)}\n\n"
        )

    if blocks and blocks.rag_context_blocks:
        ctx = "CONTESTO EVIDENZIALE (per-doc):\n" + _j(blocks.rag_context_blocks, 10) + "\n\n"

    return (
        "Task: estrarre STEP strutturati per BPMN (diagram-ready).\n\n"
        "Vincoli forti:\n"
        "- Fonte: SOLO testo sezione + contesto evidenziale per-doc.\n"
        "- Cross-document: SOLO pattern/feature (non contengono testo) => NON usarli come contenuto.\n"
        "- Non inventare step/rami/ruoli. Se manca info, metti una nota.\n\n"
        "Regole agent:\n"
        "- agent deve essere in ITALIANO.\n"
        "- Vietati: 'operator', 'system', 'user', 'citizen', 'applicant'.\n"
        "- Default: 'Operatore'.\n\n"
        "Tipi ammessi:\n"
        "- activity_task (default)\n"
        "- activity_subprocess (solo se esplicito sottoprocesso)\n"
        "- gateway_exclusive / gateway_parallel / gateway_inclusive / gateway_event (solo se decisione/ramo/parallelismo nel testo)\n"
        "- event_start / event_end / event_timer / event_message / event_signal / event_conditional / event_intermediate / event_link\n"
        "  (usa eventi SOLO se chiaramente eventi; altrimenti task)\n\n"
        "Riuso azioni interne:\n"
        "- Se nel testo c'è un rimando ('vedi punto...', 'come fatto in...') e nelle AZIONI RIUSABILI esiste una action_id pertinente:\n"
        "  - NON duplicare la sottoprocedura\n"
        "  - description_synthetic: '... come in <ref> (...)'\n"
        "  - meta: 'ref=action:Ax'\n\n"
        "Output: SOLO JSON valido:\n"
        "{ \"status\": \"complete\"|\"partial\", \"notes\": [str],\n"
        "  \"steps\": [ {\"id\":str,\"type\":str,\"agent\":str,\"description_synthetic\":str,\"meta\":str} ] }\n\n"
        f"{defs}{allowed_agents}{cross_steps}{support_actions}{ctx}"
        f"procedure_id: {procedure_id}\n"
        f"section_title: {section_title}\n"
        f"subsection_title: {subsection_title}\n"
        f"section_path: {section_path}\n\n"
        "TESTO SEZIONE:\n"
        f"{section_text}\n"
    )

def prompt_normalize_definition(term: str, evidence_blocks: List[dict]) -> str:
    """
    Prompt “RAG di consolidamento”: produce forma canonica + alias.
    NB: se vuoi separarlo, spostalo in utils/prompts_base.py (consigliato).
    """
    return (
        "Sei un assistente che normalizza definizioni in documenti amministrativi.\n"
        "Dato un TERMINE e un insieme di EVIDENZE (definizioni/passaggi), produci una sintesi canonica.\n\n"
        "Obiettivi:\n"
        "1) 'normalized': definizione breve, massima 120 caratteri, in italiano.\n"
        "2) 'canonical_agent': se il termine indica un ruolo/soggetto (agente), estrai una forma canonica per l'agente.\n"
        "   Esempio: TERMINE='Operatore' -> canonical_agent='Titolare dell’impresa'.\n"
        "3) 'aliases': lista di alias utili per grounding (incluso il termine stesso e varianti), max 8.\n"
        "4) 'is_agent': true/false.\n\n"
        "Vincoli:\n"
        "- Usa SOLO le evidenze fornite. Se non basta, lascia normalized vuoto e is_agent=false.\n"
        "- Non inventare conoscenza esterna.\n"
        "- canonical_agent deve essere una stringa corta (max 60 caratteri) e in Maiuscole/minuscole corrette.\n\n"
        "Output SOLO JSON:\n"
        '{ "term": "...", "normalized": "...", "is_agent": true|false, "canonical_agent": "...", "aliases": ["..."] }\n\n'
        f"TERMINE: {term}\n"
        f"EVIDENZE:\n{json.dumps(evidence_blocks[:12], ensure_ascii=False)}"
    )

