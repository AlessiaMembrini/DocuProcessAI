# diagrams/run_diagrams.py
from __future__ import annotations
import os
import json
from typing import Dict, Any, List

from diagrams.strategy import generate_all_strategy_driven_diagrams
from diagrams.agent_lanes import generate_all_agent_lanes_diagrams
from diagrams.processpiper_renderer import generate_all_bpmn


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def generate_three_diagram_sets(
    diagram_procedures_json_path: str,
    out_root: str,
    doc_stem: str,
) -> Dict[str, Any]:
    """
    OUTPUT STRUTTURA FINALE:

    out_root/doc_stem/
        ├── strategy/
        ├── agent_lanes/
        └── bpmn/

    - Nessuna cartella per procedura
    - Diagrammi organizzati SOLO per tipo
    """

    # -------------------------------------------------
    # 1) carico le procedure
    # -------------------------------------------------
    with open(diagram_procedures_json_path, "r", encoding="utf-8") as f:
        procedures = json.load(f)

    # -------------------------------------------------
    # 2) creo la struttura finale per il documento
    # -------------------------------------------------
    doc_dir = ensure_dir(os.path.join(out_root, doc_stem))

    strategy_dir = ensure_dir(os.path.join(doc_dir, "strategy"))
    agent_lanes_dir = ensure_dir(os.path.join(doc_dir, "agent_lanes"))
    bpmn_dir = ensure_dir(os.path.join(doc_dir, "bpmn"))

    # -------------------------------------------------
    # 3) cartella tecnica per JSON temporanei
    #    (input per i generatori, NON output finale)
    # -------------------------------------------------
    staging_dir = ensure_dir(os.path.join(doc_dir, "_proc_json"))

    generated = {
        "strategy_driven": [],
        "agent_lanes": [],
        "processpiper_bpmn": [],
    }

    # -------------------------------------------------
    # 4) per ogni procedura:
    #    - creo JSON temporaneo
    #    - genero diagrammi nei folder di TIPO
    # -------------------------------------------------
    for idx, proc in enumerate(procedures):
        proc_json_path = os.path.join(staging_dir, f"proc_{idx:03d}.json")
        write_json(proc_json_path, [proc])  # wrapper vogliono lista

        # strategy-driven
        res_strategy = generate_all_strategy_driven_diagrams(
            proc_json_path,
            strategy_dir,
        )
        generated["strategy_driven"].append(res_strategy)

        # agent lanes
        res_lanes = generate_all_agent_lanes_diagrams(
            proc_json_path,
            agent_lanes_dir,
        )
        generated["agent_lanes"].append(res_lanes)

        # BPMN (processpiper)
        res_bpmn = generate_all_bpmn(
            proc_json_path,
            bpmn_dir,
        )
        generated["processpiper_bpmn"].append(res_bpmn)

    return generated
