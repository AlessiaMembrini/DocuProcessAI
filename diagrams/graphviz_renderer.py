# diagrams/graphviz_renderer.py
"""
Renderer Graphviz comune per:
- strategy-driven (layout flat)
- agent-lanes (layout lanes)

Supporta:
- branches: [{target, label}] con target="end"
- nodo start grafico se non esiste un event_start (anche se event_start non ha id)
- NO doppio end: se esiste un event_end nel JSON, target="end" viene collegato a quel nodo
- Event nodes: cerchi piccoli (fixedsize) MA con sizing dinamico per label lunghe
- Task nodes: padding/margine leggermente dinamico in base alle righe del label
- Safety: evita self-loop (es. end->end)

Tipi BPMN:
  - activity_task, activity_subprocess
  - gateway_exclusive, gateway_inclusive, gateway_parallel, gateway_event
  - event_start, event_end, event_timer, event_intermediate, event_message, event_signal,
    event_conditional, event_link

Genera PDF e SVG.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List, Tuple, Optional, Set

os.environ["GVBATCH"] = "1"  # Disattiva output GUI di Graphviz su Windows

from graphviz import Digraph


# =========================
# SIZE / TYPOGRAPHY TUNING
# =========================

# Event nodes: dynamic sizing by label lines (clamped)
EVENT_NODE_FIXEDSIZE: bool = True
EVENT_NODE_FONTSIZE: str = "8"
EVENT_DIAM_BASE: float = 1
EVENT_DIAM_ADD_PER_LINE: float = 0.22
EVENT_DIAM_MIN: float = 1
EVENT_DIAM_MAX: float = 2.80

# Start/End circles: dynamic sizing by label lines (clamped)
START_END_FONTSIZE: str = "8"
START_END_DIAM_BASE: float = 1
START_END_DIAM_ADD_PER_LINE: float = 0.22
START_END_DIAM_MIN: float = 1
START_END_DIAM_MAX: float = 2.80

# Task boxes: dynamic padding (margin) by label lines (clamped)
TASK_MARGIN_X_BASE: float = 0.20
TASK_MARGIN_Y_BASE: float = 0.12
TASK_MARGIN_ADD_PER_LINE: float = 0.05
TASK_MARGIN_X_MIN: float = 0.18
TASK_MARGIN_X_MAX: float = 0.55
TASK_MARGIN_Y_MIN: float = 0.10
TASK_MARGIN_Y_MAX: float = 0.40


# =========================
# COLORS
# =========================

def get_agent_color(agent_name: str, agent_index_map: dict) -> str:
    """Restituisce un colore univoco e coerente per ogni agente."""
    palette = [
        "#FFB3BA", "#BAFFC9", "#BAE1FF", "#FFFFBA", "#E0BBE4",
        "#957DAD", "#D291BC", "#FEC8D8", "#A2D9FF", "#B0E3AA"
    ]

    if agent_name not in agent_index_map:
        agent_index_map[agent_name] = len(agent_index_map)

    idx = agent_index_map[agent_name]
    if idx < len(palette):
        return palette[idx]

    hash_obj = hashlib.md5(agent_name.encode()).hexdigest()
    r = (int(hash_obj[:2], 16) % 100) + 155
    g = (int(hash_obj[2:4], 16) % 100) + 155
    b = (int(hash_obj[4:6], 16) % 100) + 155
    return f"#{r:02x}{g:02x}{b:02x}"


# =========================
# TEXT WRAP + LINE ESTIMATION
# =========================

def wrap_text_for_diagram(label: str, width: int = 35) -> str:
    """Suddivide un testo in righe per migliorare la leggibilità nei diagrammi."""
    words = (label or "").split()
    lines: List[str] = []
    current_line: List[str] = []

    for word in words:
        if len(" ".join(current_line + [word])) <= width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    # Graphviz newline must be escaped
    return "\\n".join(lines)


def _estimate_lines_for_wrapped_label(wrapped_label: str) -> int:
    """Conta righe su label già wrappato (con \\n Graphviz)."""
    if not wrapped_label:
        return 1
    return max(1, wrapped_label.count("\\n") + 1)


def _estimate_lines_for_label(label_raw: str, wrap_width: int) -> int:
    """Stima righe wrappando e contando."""
    wrapped = wrap_text_for_diagram(label_raw or "", width=wrap_width)
    return _estimate_lines_for_wrapped_label(wrapped)


def _scale_diameter_by_lines(lines: int, base: float, add_per_line: float, minv: float, maxv: float) -> str:
    d = base + max(0, lines - 1) * add_per_line
    d = max(minv, min(maxv, d))
    return f"{d:.2f}"


def _node_margin_by_lines(lines: int) -> str:
    """
    Graphviz node margin accepts: "x,y"
    Increases slightly with line count, clamped.
    """
    mx = TASK_MARGIN_X_BASE + max(0, lines - 2) * TASK_MARGIN_ADD_PER_LINE
    my = TASK_MARGIN_Y_BASE + max(0, lines - 2) * TASK_MARGIN_ADD_PER_LINE

    mx = max(TASK_MARGIN_X_MIN, min(TASK_MARGIN_X_MAX, mx))
    my = max(TASK_MARGIN_Y_MIN, min(TASK_MARGIN_Y_MAX, my))

    return f"{mx:.2f},{my:.2f}"


# =========================
# TYPE HELPERS
# =========================

def _is_gateway(step_type: str) -> bool:
    return isinstance(step_type, str) and step_type.startswith("gateway_")


def _is_event(step_type: str) -> bool:
    return isinstance(step_type, str) and step_type.startswith("event_")


def _is_hex_color(c: str) -> bool:
    return isinstance(c, str) and re.fullmatch(r"#([0-9a-fA-F]{6})", c) is not None


def _lighten_hex(color: str, amount: float = 0.55) -> str:
    """
    Schiarisce un colore esadecimale (#RRGGBB) miscelandolo con bianco.
    amount in [0,1]: 0 = nessun cambiamento, 1 = bianco.
    """
    if not _is_hex_color(color):
        return "lightblue"
    amount = max(0.0, min(1.0, float(amount)))

    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)

    r2 = int(r + (255 - r) * amount)
    g2 = int(g + (255 - g) * amount)
    b2 = int(b + (255 - b) * amount)

    return f"#{r2:02x}{g2:02x}{b2:02x}"


def _node_style_for_step(step_type: str, agent_color: str, lanes_mode: bool) -> Tuple[str, str]:
    """Ritorna (shape, fillcolor) in base al tipo BPMN."""
    if _is_gateway(step_type):
        shape = "diamond"
        if step_type == "gateway_parallel":
            return shape, "lightgreen"
        if step_type == "gateway_inclusive":
            return shape, "lightblue"
        if step_type == "gateway_event":
            return shape, "orange"
        return shape, "lightyellow"

    if _is_event(step_type):
        if step_type == "event_end":
            return "doublecircle", "red"
        if step_type == "event_start":
            return "circle", "green"
        return "circle", "white"

    if step_type == "activity_subprocess":
        return "box3d", ("lightblue" if lanes_mode else (agent_color or "lightblue"))

    base = agent_color if _is_hex_color(agent_color) else "lightblue"
    return "box", (_lighten_hex(base, amount=0.55) if lanes_mode else base)


def _event_label(step_type: str, desc: str) -> str:
    """
    Event label completo (con descrizione).
    Nota: user wants it inside the circle even if cramped.
    """
    base = step_type.replace("event_", "Event: ").replace("_", " ").title()
    if desc:
        return f"{base}\n{desc}".strip()
    return base


# =========================
# STEPS COLLECTION / MAPPING
# =========================

def _collect_steps(items_or_steps: Any) -> List[Dict[str, Any]]:
    """
    Accetta:
    - lista di steps (dict con id/type/agent/description_synthetic/branches)
    - oppure lista di procedure (dict con "steps" o "points")
    e restituisce lista piatta di steps.

    Nota: qui FILTRIAMO gli step senza id per il disegno dei nodi,
    ma per la logica "start grafico se non esiste event_start anche senza id"
    gestiamo un flag separato a livello render_graphviz (scorrendo i dati originali).
    """
    if not items_or_steps or not isinstance(items_or_steps, list):
        return []

    looks_like_procedure = any(isinstance(x, dict) and ("steps" in x or "points" in x) for x in items_or_steps)

    all_steps: List[Dict[str, Any]] = []
    if looks_like_procedure:
        for item in items_or_steps:
            if not isinstance(item, dict):
                continue
            if item.get("steps"):
                all_steps.extend(item["steps"])
            elif item.get("points"):
                for p in item.get("points") or []:
                    for s in (p.get("steps") or []):
                        all_steps.append(s)
    else:
        for s in items_or_steps:
            if isinstance(s, dict):
                all_steps.append(s)

    # Filtra step senza id (non disegnabili come nodi)
    all_steps = [s for s in all_steps if isinstance(s, dict) and s.get("id")]
    return all_steps


def _build_id_map(all_steps: List[Dict[str, Any]]) -> Dict[str, str]:
    """Mappa step_id -> node_id Graphviz."""
    id_to_node: Dict[str, str] = {}
    for s in all_steps:
        sid = s.get("id")
        if sid:
            id_to_node[sid] = f"n_{sid}"
    return id_to_node


def _flatten_any_steps(items_or_steps: Any) -> List[Dict[str, Any]]:
    """
    Variante 'loose' per cercare event_start/event_end anche quando alcuni step non hanno id.
    Non filtra per id: utile per robustezza start grafico.
    """
    if not items_or_steps or not isinstance(items_or_steps, list):
        return []

    looks_like_procedure = any(isinstance(x, dict) and ("steps" in x or "points" in x) for x in items_or_steps)
    out: List[Dict[str, Any]] = []

    if looks_like_procedure:
        for item in items_or_steps:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("steps"), list):
                out.extend([s for s in item["steps"] if isinstance(s, dict)])
            elif isinstance(item.get("points"), list):
                for p in item.get("points") or []:
                    if isinstance(p, dict) and isinstance(p.get("steps"), list):
                        out.extend([s for s in p["steps"] if isinstance(s, dict)])
    else:
        out.extend([s for s in items_or_steps if isinstance(s, dict)])

    return out


# =========================
# EDGES + START/END NODES
# =========================

def _safe_edge(dot: Digraph, src: str, tgt: str, label: Optional[str] = None) -> None:
    """Aggiunge edge evitando self-loop (src==tgt)."""
    if not src or not tgt:
        return
    if src == tgt:
        return
    dot.edge(src, tgt, label=label if label else None)


def _draw_edges(
    dot: Digraph,
    all_steps: List[Dict[str, Any]],
    id_to_node: Dict[str, str],
    *,
    has_event_start_anywhere: bool,
) -> None:
    """
    Disegna edges usando branches.

    - target="end": se esiste un event_end nel JSON (con id), collega a quello; altrimenti crea nodo End.
    - start grafico: creato solo se NON esiste alcun event_start nel JSON (anche se event_start non ha id).
    - Safety: evita cappi (end->end o in generale src==tgt).
    """
    # event_end solo se ha id mappato (nodo disegnabile)
    event_end_ids = [
        s.get("id") for s in all_steps
        if s.get("type") == "event_end" and s.get("id") in id_to_node
    ]
    event_end_node = id_to_node[event_end_ids[0]] if event_end_ids else None

    end_created = False
    all_target_ids: Set[str] = set()

    for step in all_steps:
        step_id = step.get("id")
        if step_id not in id_to_node:
            continue

        from_node = id_to_node[step_id]

        for branch in (step.get("branches") or []):
            if not isinstance(branch, dict):
                continue

            target = branch.get("target")
            label = (branch.get("label") or "").strip()

            if target and target != "end":
                all_target_ids.add(target)

            if target == "end":
                # collega a event_end se esiste, altrimenti crea nodo End fittizio
                if event_end_node:
                    # evita capello: se stai già su event_end, non creare edge verso se stesso
                    _safe_edge(dot, from_node, event_end_node, label=label)
                else:
                    if not end_created:
                        end_label = "End"
                        lines = _estimate_lines_for_label(end_label, wrap_width=12)
                        diam = _scale_diameter_by_lines(
                            lines,
                            base=START_END_DIAM_BASE,
                            add_per_line=START_END_DIAM_ADD_PER_LINE,
                            minv=START_END_DIAM_MIN,
                            maxv=START_END_DIAM_MAX,
                        )
                        dot.node(
                            "end",
                            end_label,
                            shape="doublecircle",
                            style="filled",
                            fillcolor="red",
                            fixedsize="true",
                            width=diam,
                            height=diam,
                            fontsize=START_END_FONTSIZE,
                        )
                        end_created = True
                    _safe_edge(dot, from_node, "end", label=label)

            elif target in id_to_node:
                _safe_edge(dot, from_node, id_to_node[target], label=label)

    # start grafico SOLO se non esiste event_start nel JSON (robusto anche se event_start senza id)
    if not has_event_start_anywhere:
        start_step = next((s for s in all_steps if s.get("id") and s["id"] not in all_target_ids), None)

        start_label = "Start"
        lines = _estimate_lines_for_label(start_label, wrap_width=12)
        diam = _scale_diameter_by_lines(
            lines,
            base=START_END_DIAM_BASE,
            add_per_line=START_END_DIAM_ADD_PER_LINE,
            minv=START_END_DIAM_MIN,
            maxv=START_END_DIAM_MAX,
        )

        dot.node(
            "start",
            start_label,
            shape="circle",
            style="filled",
            fillcolor="green",
            fixedsize="true",
            width=diam,
            height=diam,
            fontsize=START_END_FONTSIZE,
        )
        if start_step and start_step.get("id") in id_to_node:
            _safe_edge(dot, "start", id_to_node[start_step["id"]])


# =========================
# MAIN RENDER
# =========================

def render_graphviz(
    title: str,
    steps,  # può essere lista steps o lista procedure
    output_path_without_ext: Optional[str] = None,
    output_path_no_ext: Optional[str] = None,
    layout: Optional[str] = None,
    layout_mode: Optional[str] = None,
    lanes_label_width: int = 25,
    flat_label_width: int = 35,
    graph_rankdir: str = "TB",
    **kwargs,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Render comune.

    Compatibilità:
    - output_path_without_ext o output_path_no_ext
    - layout o layout_mode

    layout_mode:
      - "flat"  : strategy-driven (agente nel label)
      - "lanes" : agent-driven (cluster per agente)
    """
    # Compatibilità path
    if output_path_without_ext is None:
        output_path_without_ext = output_path_no_ext
    if not output_path_without_ext:
        raise ValueError("Missing output_path_without_ext/output_path_no_ext")

    # Compatibilità layout
    if layout_mode is None:
        layout_mode = layout or "flat"
    layout_mode = (layout_mode or "flat").strip().lower()
    lanes_mode = layout_mode == "lanes"

    # Robust start detection: cerca event_start anche se senza id, sul materiale originale
    raw_steps_any = _flatten_any_steps(steps)
    has_event_start_anywhere = any((s.get("type") or "").strip() == "event_start" for s in raw_steps_any)

    all_steps = _collect_steps(steps)
    if not all_steps:
        return None, None

    # Colori per agenti
    agent_set = {s.get("agent", "Operatore") or "Operatore" for s in all_steps}
    agents_sorted = sorted(agent_set)
    agent_index_map: Dict[str, int] = {}
    agent_color_map = {a: get_agent_color(a, agent_index_map) for a in agents_sorted}

    dot = Digraph(comment=title)
    dot.attr(rankdir=graph_rankdir, size="16,12", dpi="300", fontsize="12", margin="0.2")
    dot.attr(compound="true")

    id_to_node = _build_id_map(all_steps)

    if lanes_mode:
        # cluster principale
        with dot.subgraph(name="cluster_main") as main:
            main.attr(label=title, style="rounded,filled", fillcolor="white", color="black", penwidth="2")
            main.attr(labelloc="t", labeljust="c")

            # lane per agente
            lane_map = {a: f"lane_{i}" for i, a in enumerate(agents_sorted)}
            for agent in agents_sorted:
                color = agent_color_map.get(agent, "lightblue")
                with main.subgraph(name=f"cluster_{lane_map[agent]}") as lane:
                    lane.attr(label=agent, style="rounded,filled", fillcolor=color)
                    lane.attr(labelloc="t", labeljust="c")

            # nodi dentro lane
            for step in all_steps:
                sid = step.get("id")
                if sid not in id_to_node:
                    continue
                node_id = id_to_node[sid]

                agent = step.get("agent", "Operatore") or "Operatore"
                step_type = (step.get("type") or "activity_task").strip()
                desc = (step.get("description_synthetic") or step.get("text") or "").strip()

                if _is_event(step_type):
                    label_raw = _event_label(step_type, desc)
                else:
                    label_raw = desc if desc else step_type

                label = wrap_text_for_diagram(label_raw, width=lanes_label_width)
                line_count = _estimate_lines_for_wrapped_label(label)

                shape, fillcolor = _node_style_for_step(
                    step_type,
                    agent_color_map.get(agent, "lightblue"),
                    lanes_mode=True,
                )

                node_kwargs: Dict[str, str] = {
                    "shape": shape,
                    "style": "rounded,filled",
                    "fillcolor": fillcolor,
                }

                if _is_event(step_type):
                    diam = _scale_diameter_by_lines(
                        line_count,
                        base=EVENT_DIAM_BASE,
                        add_per_line=EVENT_DIAM_ADD_PER_LINE,
                        minv=EVENT_DIAM_MIN,
                        maxv=EVENT_DIAM_MAX,
                    )
                    node_kwargs.update({
                        "fixedsize": "true" if EVENT_NODE_FIXEDSIZE else "false",
                        "width": diam,
                        "height": diam,
                        "fontsize": EVENT_NODE_FONTSIZE,
                    })
                else:
                    # Task/subprocess: padding leggermente dinamico in base alle righe
                    node_kwargs["margin"] = _node_margin_by_lines(line_count)

                with main.subgraph(name=f"cluster_{lane_map.get(agent, 'lane_0')}") as lane:
                    lane.node(node_id, label, **node_kwargs)

        _draw_edges(dot, all_steps, id_to_node, has_event_start_anywhere=has_event_start_anywhere)

    else:
        # flat: nodi nel grafo principale, label include agente (tranne eventi)
        for step in all_steps:
            sid = step.get("id")
            if sid not in id_to_node:
                continue
            node_id = id_to_node[sid]

            agent = step.get("agent", "Operatore") or "Operatore"
            step_type = (step.get("type") or "activity_task").strip()
            desc = (step.get("description_synthetic") or step.get("text") or "").strip()

            if _is_event(step_type):
                label_raw = _event_label(step_type, desc)
            else:
                label_raw = f"{agent}: {desc}".strip() if desc else f"{agent}: {step_type}"

            label = wrap_text_for_diagram(label_raw, width=flat_label_width)
            line_count = _estimate_lines_for_wrapped_label(label)

            shape, fillcolor = _node_style_for_step(
                step_type,
                agent_color_map.get(agent, "lightblue"),
                lanes_mode=False,
            )

            node_kwargs2: Dict[str, str] = {
                "shape": shape,
                "style": "rounded,filled",
                "fillcolor": fillcolor,
            }

            if _is_event(step_type):
                diam = _scale_diameter_by_lines(
                    line_count,
                    base=EVENT_DIAM_BASE,
                    add_per_line=EVENT_DIAM_ADD_PER_LINE,
                    minv=EVENT_DIAM_MIN,
                    maxv=EVENT_DIAM_MAX,
                )
                node_kwargs2.update({
                    "fixedsize": "true" if EVENT_NODE_FIXEDSIZE else "false",
                    "width": diam,
                    "height": diam,
                    "fontsize": EVENT_NODE_FONTSIZE,
                })
            else:
                node_kwargs2["margin"] = _node_margin_by_lines(line_count)

            dot.node(node_id, label, **node_kwargs2)

        _draw_edges(dot, all_steps, id_to_node, has_event_start_anywhere=has_event_start_anywhere)

    # Render PDF + SVG
    pdf_path = output_path_without_ext + ".pdf"
    svg_path = output_path_without_ext + ".svg"

    dot.format = "pdf"
    dot.render(output_path_without_ext, cleanup=True)

    dot.format = "svg"
    dot.render(output_path_without_ext, cleanup=True)

    return pdf_path, svg_path
