# diagrams/strategy.py
"""
Wrapper per diagrammi strategy-driven (flat):
- un diagramma per procedura (procedure_id)
- label include "Agente: descrizione"
"""

import os
import json

from diagrams.graphviz_renderer import render_graphviz
from utils.io_utils import sanitize_filename, unique_filename


def generate_all_strategy_driven_diagrams(json_file: str, output_dir: str):
    with open(json_file, "r", encoding="utf-8") as f:
        procedures = json.load(f)

    if not isinstance(procedures, list):
        print("[WARN] JSON non è una lista")
        return []

    out_dir = os.path.join(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    generated = []
    for i, proc in enumerate(procedures):
        if not isinstance(proc, dict):
            continue

        proc_id = (proc.get("procedure_id") or f"proc_{i:03d}").strip()
        sec = proc.get("section_title") or "N/A"
        sub = proc.get("subsection_title") or "N/A"
        title = f"{sec} | {sub}"

        base_name = sanitize_filename(proc_id)
        base = os.path.join(out_dir, f"{i:03d}_{base_name}_strategy_driven")
        base = unique_filename(base + ".svg").replace(".svg", "")

        pdf_path, svg_path = render_graphviz(
            title=title,
            steps=[proc],
            output_path_without_ext=base,
            layout="flat",
        )
        if pdf_path and svg_path:
            generated.append((pdf_path, svg_path))

    return generated
