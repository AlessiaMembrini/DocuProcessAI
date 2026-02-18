# diagrams/processpiper_renderer.py
r"""
Genera diagrammi con ProcessPiper usando PiperFlow (text2diagram), producendo file SVG.

Robustezza:
- Lane names univoci dopo sanitizzazione (evita collisioni)
- Constraint ProcessPiper: max 4 connessioni totali per elemento (in+out)
  -> fan-in limiter: merge gateway tree PRIMA del target
  -> fan-out limiter: split gateway tree DOPO la sorgente (corretto: niente gateway orfani)
- Safety:
  - rimuove self-loop (x->x)
  - rimuove edges verso nodi non dichiarati
  - guard-rail start: se start è nel DSL, deve avere almeno un edge in uscita
- Critico (bug ProcessPiper):
  - in una lane, (end) deve essere SEMPRE l’ultimo elemento stampato
    (anche quando event_end è uno step nel JSON).
- Critico (bug ProcessPiper routing):
  - evita collegamenti gateway->gateway inserendo “anchor task” intermedi.
- DEBUG:
  - salva SEMPRE il DSL completo in un file .piperflow.txt accanto allo SVG (anche se render fallisce)
"""

from __future__ import annotations

import os
import json
import re
import unicodedata
from typing import Dict, Any, List, Set, Tuple, Optional

from processpiper.text2diagram import render


# ================================
# TUNING / ESTETICA
# ================================
ENABLE_SVG_POSTPROCESS: bool = True
SVG_REMOVE_TEXT_STROKE: bool = False
SVG_VIEWBOX_PAD: int = 150

AUTO_WIDTH_MIN: int = 6500
AUTO_WIDTH_MAX: int = 24000

LANE_HEADER_MIN_PX: int = 90
LANE_HEADER_MAX_PX: int = 220

VSPACE_BETWEEN_LANES_PX: int = 4
VSPACE_BETWEEN_POOLS_PX: int = 6
VSPACE_BETWEEN_SHAPES_PX: int = 8

ENABLE_NO_POOL_COMPACT: bool = True

# ProcessPiper: max 4 connessioni totali (in+out) per elemento
MAX_TOTAL_CONNECTIONS: int = 4

MERGE_GATEWAY_LABEL: str = "Merge"
SPLIT_GATEWAY_LABEL: str = "Split"

ENABLE_START_GUARD_RAIL: bool = True

BOX_HEIGHT_BASE: int = 110
BOX_HEIGHT_ADD_PER_EXTRA_LINE: int = 30
BOX_WIDTH_MIN: int = 320

ENABLE_LANE_HEIGHT_PATCH: bool = True


# ================================
# SANITIZZAZIONE
# ================================

def sanitize_for_piperflow(text: Optional[str], fallback: str = "") -> str:
    if text is None:
        return fallback

    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    cleaned: List[str] = []
    for ch in s:
        if unicodedata.category(ch).startswith("C"):
            continue
        cleaned.append(ch)
    s = "".join(cleaned)

    # caratteri che possono rompere il DSL
    s = re.sub(r'[{}\[\]"<>|]', " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else fallback


def sanitize_title(text: Optional[str], fallback: str) -> str:
    t = sanitize_for_piperflow(text, fallback=fallback)
    t = t.replace(":", " ").strip()
    t = re.sub(r"\s+", " ", t).strip()
    return (t[:80]).strip() or fallback


def sanitize_piperflow_document(doc: str) -> str:
    if not doc:
        return ""

    d = unicodedata.normalize("NFKC", doc)
    d = (
        d.replace("’", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("–", "-")
         .replace("—", "-")
    )

    cleaned: List[str] = []
    for ch in d:
        cat0 = unicodedata.category(ch)[0]
        if cat0 == "C" and ch not in ("\n", "\t"):
            continue
        cleaned.append(ch)
    d = "".join(cleaned)

    d = d.replace("\r\n", "\n").replace("\r", "\n")
    d = "\n".join(line.rstrip() for line in d.split("\n"))
    d = re.sub(r"\n{3,}", "\n\n", d)
    return d


# ================================
# RENDER SAFETY
# ================================

def _escape_for_processpiper_python_literal(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\\", "\\\\")
    s = s.replace("'", "\\'")
    s = s.replace('"""', r'\"\"\"')
    return s


def _safe_output_path(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


# ================================
# SVG POST-PROCESS
# ================================

def _inject_svg_css(svg_path: str, css_text: str) -> None:
    try:
        with open(svg_path, "r", encoding="utf-8") as f:
            svg = f.read()

        if "<style" in svg:
            return

        style_block = f"<style type=\"text/css\"><![CDATA[\n{css_text}\n]]></style>\n"
        if "<defs>" in svg:
            svg = svg.replace("<defs>", "<defs>\n" + style_block, 1)
        else:
            m = re.search(r"<svg\b[^>]*>", svg)
            if not m:
                return
            insert_at = m.end()
            svg = svg[:insert_at] + "\n" + style_block + svg[insert_at:]

        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg)
    except Exception:
        return


def _expand_svg_viewbox(svg_path: str, pad: int = SVG_VIEWBOX_PAD) -> None:
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(svg_path)
        root = tree.getroot()

        vb = root.get("viewBox")
        if vb:
            parts = vb.strip().split()
            if len(parts) == 4:
                minx, miny, w, h = map(float, parts)
                minx -= pad
                miny -= pad
                w += 2 * pad
                h += 2 * pad
                root.set("viewBox", f"{minx:g} {miny:g} {w:g} {h:g}")

        def _bump_dim(attr: str):
            v = root.get(attr)
            if not v:
                return
            m = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(px)?\s*$", v)
            if not m:
                return
            num = float(m.group(1)) + 2 * pad
            unit = m.group(2) or ""
            root.set(attr, f"{num:g}{unit}")

        _bump_dim("width")
        _bump_dim("height")

        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
    except Exception:
        return


def _postprocess_svg(svg_path: str) -> None:
    if (not ENABLE_SVG_POSTPROCESS) or (not svg_path.lower().endswith(".svg")):
        return
    _expand_svg_viewbox(svg_path, pad=SVG_VIEWBOX_PAD)
    if SVG_REMOVE_TEXT_STROKE:
        rules = "text, tspan { stroke: none !important; }\n"
        _inject_svg_css(svg_path, rules)


# ================================
# PROCESSPIPER CONFIG + PATCH
# ================================

def _estimate_lane_header_width_px(agent_names: List[str]) -> int:
    max_len = 0
    for n in agent_names:
        max_len = max(max_len, len((n or "").strip()))
    px = int(28 + max_len * 7.2)
    return max(LANE_HEADER_MIN_PX, min(LANE_HEADER_MAX_PX, px))


def _estimate_diagram_width(num_nodes: int, num_edges: int) -> int:
    base = 5000
    w = base + num_nodes * 240 + num_edges * 90
    return max(AUTO_WIDTH_MIN, min(AUTO_WIDTH_MAX, int(w)))


def _patch_lane_set_draw_position_to_use_box_height() -> None:
    if not ENABLE_LANE_HEIGHT_PATCH:
        return
    try:
        from processpiper.lane import Lane
        from processpiper.constants import Configs
    except Exception:
        return

    if getattr(Lane.set_draw_position, "__name__", "") == "patched":
        return

    def patched(self, x: int, y: int, layout_grid):
        lane_row_count = layout_grid.get_lane_row_count(self.id)

        self.coord.x_pos = (
            x if x > 0 else
            Configs.SURFACE_LEFT_MARGIN
            + Configs.POOL_TEXT_WIDTH
            + Configs.HSPACE_BETWEEN_POOL_AND_LANE
        )
        self.coord.y_pos = y if y > 0 else Configs.SURFACE_TOP_MARGIN

        max_column_count = layout_grid.get_max_column_count()
        self.width = (
            (Configs.HSPACE_BETWEEN_SHAPES * max_column_count - 1)
            + (Configs.BOX_WIDTH * max_column_count)
            + (Configs.LANE_SHAPE_LEFT_MARGIN)
        )

        self.height = (
            (lane_row_count * Configs.BOX_HEIGHT)
            + ((lane_row_count - 1) * Configs.VSPACE_BETWEEN_SHAPES)
            + Configs.LANE_SHAPE_TOP_MARGIN
            + Configs.LANE_SHAPE_BOTTOM_MARGIN
        )

        y_pos = self.coord.y_pos + self.height + Configs.VSPACE_BETWEEN_LANES
        return self.coord.x_pos, y_pos, self.width, self.height

    patched.__name__ = "patched"
    Lane.set_draw_position = patched


def _apply_dynamic_graphics(
    piper_syntax: str,
    lane_names: List[str],
    max_label_lines: int,
) -> None:
    try:
        from processpiper.constants import Configs
    except Exception:
        return

    has_pool = bool(re.search(r"(?m)^\s*pool\s*:", piper_syntax))
    if ENABLE_NO_POOL_COMPACT and (not has_pool):
        if hasattr(Configs, "POOL_TEXT_WIDTH"):
            Configs.POOL_TEXT_WIDTH = 0
        if hasattr(Configs, "HSPACE_BETWEEN_POOL_AND_LANE"):
            Configs.HSPACE_BETWEEN_POOL_AND_LANE = 0

    lane_w = _estimate_lane_header_width_px(lane_names)
    if hasattr(Configs, "LANE_TEXT_WIDTH"):
        Configs.LANE_TEXT_WIDTH = int(lane_w)

    if hasattr(Configs, "VSPACE_BETWEEN_LANES"):
        Configs.VSPACE_BETWEEN_LANES = int(VSPACE_BETWEEN_LANES_PX)
    if hasattr(Configs, "VSPACE_BETWEEN_POOLS"):
        Configs.VSPACE_BETWEEN_POOLS = int(VSPACE_BETWEEN_POOLS_PX)
    if hasattr(Configs, "VSPACE_BETWEEN_SHAPES"):
        Configs.VSPACE_BETWEEN_SHAPES = int(VSPACE_BETWEEN_SHAPES_PX)

    base_h = int(getattr(Configs, "BOX_HEIGHT", 60))
    base_h = max(base_h, BOX_HEIGHT_BASE)

    target_h = max(
        base_h,
        BOX_HEIGHT_BASE + max(0, (max_label_lines - 2)) * BOX_HEIGHT_ADD_PER_EXTRA_LINE
    )
    if hasattr(Configs, "BOX_HEIGHT"):
        Configs.BOX_HEIGHT = int(target_h)

    if hasattr(Configs, "BOX_WIDTH"):
        current_w = int(getattr(Configs, "BOX_WIDTH", 140))
        Configs.BOX_WIDTH = int(max(current_w, BOX_WIDTH_MIN))

    if ENABLE_LANE_HEIGHT_PATCH and int(getattr(Configs, "BOX_HEIGHT", 60)) > 60:
        _patch_lane_set_draw_position_to_use_box_height()


# ================================
# LANE UNIQ
# ================================

def _make_unique_names(names: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        base = (n or "").strip() or "Operatore"
        k = seen.get(base, 0)
        if k == 0:
            out.append(base)
            seen[base] = 1
        else:
            k += 1
            seen[base] = k
            out.append(f"{base} ({k})")
    return out


# ================================
# TYPES + ALIAS
# ================================

def _is_gateway(t: str) -> bool:
    return isinstance(t, str) and t.startswith("gateway_")


def _is_start_event(t: str) -> bool:
    return t == "event_start"


def _is_end_event(t: str) -> bool:
    return t == "event_end"


def _alias_for_step(step: Dict[str, Any]) -> str:
    memo = step.get("_piper_alias")
    if isinstance(memo, str) and memo:
        return memo

    t = (step.get("type") or "").strip()
    if _is_start_event(t):
        step["_piper_alias"] = "start"
        return "start"
    if _is_end_event(t):
        step["_piper_alias"] = "end"
        return "end"

    sid = step.get("id")
    if sid:
        alias = f"s_{sid}"
        step["_piper_alias"] = alias
        return alias

    import uuid
    alias = f"s_{uuid.uuid4().hex[:8]}"
    step["_piper_alias"] = alias
    return alias


_EVENT_TAG_MAP = {
    "event_timer": "@timer",
    "event_intermediate": "@intermediate",
    "event_message": "@message",
    "event_signal": "@signal",
    "event_conditional": "@conditional",
    "event_link": "@link",
}


def _piper_gateway_line(label: str, alias: str) -> Tuple[str, str, int]:
    lab = sanitize_for_piperflow(label, fallback="Gateway")
    return f"<{lab}> as {alias}", alias, 1


def _piper_element_line(step: Dict[str, Any]) -> Tuple[str, str, int]:
    step_type = (step.get("type") or "activity_task").strip()
    alias = _alias_for_step(step)

    if _is_start_event(step_type):
        return "(start) as start", "start", 1
    if _is_end_event(step_type):
        return "(end) as end", "end", 1

    raw_desc = step.get("description_synthetic") or step.get("description") or ""
    desc = sanitize_for_piperflow(raw_desc, fallback="Azione")
    lc = 1

    if step_type in _EVENT_TAG_MAP:
        tag = _EVENT_TAG_MAP[step_type]
        return f"({tag} {desc}) as {alias}", alias, lc

    if _is_gateway(step_type):
        gw_prefix = {
            "gateway_exclusive": "",
            "gateway_parallel": "@parallel ",
            "gateway_inclusive": "@inclusive ",
            "gateway_event": "@event ",
        }.get(step_type, "")
        label = desc if desc else "Decisione"
        return f"<{gw_prefix}{label}> as {alias}", alias, lc

    if step_type == "activity_subprocess":
        return f"[@subprocess {desc}] as {alias}", alias, lc

    return f"[{desc}] as {alias}", alias, lc


# ================================
# ANCHOR TASKS (gateway->gateway workaround)
# ================================

def _is_split_gw(alias: str) -> bool:
    return isinstance(alias, str) and alias.startswith("gw_split_")


def _is_merge_gw(alias: str) -> bool:
    return isinstance(alias, str) and alias.startswith("gw_merge_")


def _is_any_gw(alias: str) -> bool:
    return _is_split_gw(alias) or _is_merge_gw(alias)


def _mk_anchor(alias_a: str, alias_b: str, k: int) -> str:
    base = f"anc_{alias_a}_{alias_b}_{k}".replace("__", "_")
    return base[:60]


def _add_edge_with_anchor_if_needed(
    cur_edges: List[Tuple[str, str, str]],
    extra_nodes: List[Tuple[str, str, int]],
    src: str,
    tgt: str,
    label: str,
    k: int,
) -> None:
    """
    Se src e tgt sono entrambi gateway (split/merge), inserisce un task vuoto intermedio:
      src -> anc -> tgt
    per evitare crash di routing in ProcessPiper.
    """
    if _is_any_gw(src) and _is_any_gw(tgt):
        anc = _mk_anchor(src, tgt, k)
        # task vuoto
        extra_nodes.append((f"[ ] as {anc}", anc, 1))
        cur_edges.append((src, anc, ""))
        cur_edges.append((anc, tgt, label or ""))
    else:
        cur_edges.append((src, tgt, label or ""))


# ================================
# EDGES BUILD
# ================================

def _build_edges_from_branches(
    steps: List[Dict[str, Any]],
    include_graphic_start: bool,
) -> List[Tuple[str, str, str]]:
    edges: List[Tuple[str, str, str]] = []
    if not steps:
        return edges

    for s in steps:
        _alias_for_step(s)

    # start grafico verso primo step
    if include_graphic_start:
        edges.append(("start", _alias_for_step(steps[0]), ""))

    valid_ids: Set[str] = {s.get("id") for s in steps if s.get("id")}
    id_to_alias: Dict[str, str] = {s["id"]: _alias_for_step(s) for s in steps if s.get("id")}

    def _resolve_target(tid: Any) -> Optional[str]:
        if tid is None:
            return None
        if tid == "end":
            return "end"
        if isinstance(tid, str) and tid in valid_ids:
            return id_to_alias.get(tid, f"s_{tid}")
        return None

    for i, step in enumerate(steps):
        src = _alias_for_step(step)
        branches = step.get("branches")

        if isinstance(branches, list) and branches:
            for br in branches:
                if not isinstance(br, dict):
                    continue
                tid = br.get("target")
                raw_label = (br.get("label") or "").strip()
                label = sanitize_for_piperflow(raw_label, fallback="") if raw_label else ""
                tgt = _resolve_target(tid)
                if tgt:
                    edges.append((src, tgt, label))
            continue

        # fallback sequenziale
        if i < len(steps) - 1:
            edges.append((src, _alias_for_step(steps[i + 1]), ""))

    return edges


def _ensure_start_connected(
    edges: List[Tuple[str, str, str]],
    steps: List[Dict[str, Any]],
    *,
    start_is_in_dsl: bool,
) -> List[Tuple[str, str, str]]:
    if (not ENABLE_START_GUARD_RAIL) or (not start_is_in_dsl) or (not steps):
        return edges
    if any(s0 == "start" for (s0, _t0, _lbl) in edges):
        return edges
    first = _alias_for_step(steps[0])
    if first != "start":
        return edges + [("start", first, "")]
    for s in steps[1:]:
        a = _alias_for_step(s)
        if a != "start":
            return edges + [("start", a, "")]
    return edges


# ================================
# MERGE TREE (FAN-IN)
# ================================

def _merge_tree_for_target(
    edges: List[Tuple[str, str, str]],
    *,
    target_alias: str,
    max_incoming: int,
    gateway_prefix: str,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, int]]]:
    in_to_target = [e for e in edges if e[1] == target_alias]
    if len(in_to_target) <= max_incoming:
        return edges, []

    other_edges = [e for e in edges if e[1] != target_alias]
    incoming_sources: List[Tuple[str, str]] = [(src, lbl) for (src, _t, lbl) in in_to_target]

    extra_nodes: List[Tuple[str, str, int]] = []
    gw_counter = 0

    def new_gw_alias() -> str:
        nonlocal gw_counter
        gw_counter += 1
        return f"{gateway_prefix}_{target_alias}_{gw_counter}"

    def chunk(lst: List[Any], n: int) -> List[List[Any]]:
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    cur_edges = list(other_edges)

    # livello 1
    groups = chunk(incoming_sources, max_incoming)
    level_nodes: List[str] = []
    for g in groups:
        gw = new_gw_alias()
        line, _, lc = _piper_gateway_line(MERGE_GATEWAY_LABEL, gw)
        extra_nodes.append((line, gw, lc))
        for (src, lbl) in g:
            cur_edges.append((src, gw, lbl or ""))
        level_nodes.append(gw)

    # livelli successivi
    while len(level_nodes) > max_incoming:
        groups2 = chunk(level_nodes, max_incoming)
        next_nodes: List[str] = []
        for g2 in groups2:
            gw = new_gw_alias()
            line, _, lc = _piper_gateway_line(MERGE_GATEWAY_LABEL, gw)
            extra_nodes.append((line, gw, lc))
            for src in g2:
                _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, src, gw, "", gw_counter)
            next_nodes.append(gw)
        level_nodes = next_nodes

    if len(level_nodes) == 1:
        _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, level_nodes[0], target_alias, "", gw_counter)
        return cur_edges, extra_nodes

    root = new_gw_alias()
    line, _, lc = _piper_gateway_line(MERGE_GATEWAY_LABEL, root)
    extra_nodes.append((line, root, lc))
    for src in level_nodes:
        _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, src, root, "", gw_counter)
    _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, root, target_alias, "", gw_counter)
    return cur_edges, extra_nodes


# ================================
# SPLIT TREE (FAN-OUT)
# ================================

def _split_tree_for_source(
    edges: List[Tuple[str, str, str]],
    *,
    source_alias: str,
    max_outgoing: int,
    gateway_prefix: str,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, int]]]:
    out_of_source = [e for e in edges if e[0] == source_alias]
    if len(out_of_source) <= max_outgoing:
        return edges, []

    other_edges = [e for e in edges if e[0] != source_alias]
    targets: List[Tuple[str, str]] = [(tgt, lbl) for (_src, tgt, lbl) in out_of_source]

    extra_nodes: List[Tuple[str, str, int]] = []
    gw_counter = 0

    def new_gw_alias() -> str:
        nonlocal gw_counter
        gw_counter += 1
        return f"{gateway_prefix}_{source_alias}_{gw_counter}"

    def chunk(lst: List[Any], n: int) -> List[List[Any]]:
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    cur_edges = list(other_edges)

    # root split: source -> root
    root = new_gw_alias()
    line, _, lc = _piper_gateway_line(SPLIT_GATEWAY_LABEL, root)
    extra_nodes.append((line, root, lc))
    cur_edges.append((source_alias, root, ""))

    current_children: List[Any] = targets

    # collassa top-down fino a <= max_outgoing dal root
    while len(current_children) > max_outgoing:
        groups = chunk(current_children, max_outgoing)
        next_level: List[str] = []
        for g in groups:
            gw = new_gw_alias()
            line, _, lc = _piper_gateway_line(SPLIT_GATEWAY_LABEL, gw)
            extra_nodes.append((line, gw, lc))

            for item in g:
                if isinstance(item, tuple) and len(item) == 2:
                    tgt, lbl = item
                    _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, gw, tgt, lbl or "", gw_counter)
                else:
                    _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, gw, str(item), "", gw_counter)

            next_level.append(gw)

        current_children = next_level

    # root -> children finali
    for item in current_children:
        if isinstance(item, tuple) and len(item) == 2:
            tgt, lbl = item
            _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, root, tgt, lbl or "", gw_counter)
        else:
            _add_edge_with_anchor_if_needed(cur_edges, extra_nodes, root, str(item), "", gw_counter)

    return cur_edges, extra_nodes


# ================================
# GLOBAL LIMITERS + SAFETY
# ================================

def _filter_edges(
    edges: List[Tuple[str, str, str]],
    declared_nodes: Set[str],
) -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []
    for s0, t0, lbl in edges:
        if not s0 or not t0:
            continue
        if s0 == t0:
            continue
        if (s0 in declared_nodes) and (t0 in declared_nodes):
            out.append((s0, t0, lbl))
    return out


def _apply_limiters_globally(
    edges: List[Tuple[str, str, str]],
    declared_nodes: Set[str],
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, int]]]:
    if not edges:
        return edges, []

    extra_all: List[Tuple[str, str, int]] = []
    cur_edges = list(edges)

    for _round in range(10):
        indeg: Dict[str, int] = {}
        outdeg: Dict[str, int] = {}
        for s0, t0, _ in cur_edges:
            outdeg[s0] = outdeg.get(s0, 0) + 1
            indeg[t0] = indeg.get(t0, 0) + 1

        changed = False

        # FAN-IN
        targets = sorted(indeg.keys(), key=lambda k: indeg.get(k, 0), reverse=True)
        for tgt in targets:
            if tgt not in declared_nodes:
                continue
            tgt_out = outdeg.get(tgt, 0)
            reserve_out = max(1, tgt_out) if tgt_out > 0 else 0
            allowed_in = max(1, MAX_TOTAL_CONNECTIONS - reserve_out)

            if indeg.get(tgt, 0) > allowed_in:
                cur_edges, extra = _merge_tree_for_target(
                    cur_edges,
                    target_alias=tgt,
                    max_incoming=allowed_in,
                    gateway_prefix="gw_merge",
                )
                if extra:
                    for _line, a, _lc in extra:
                        declared_nodes.add(a)
                    extra_all.extend(extra)
                    changed = True
                    break

        if changed:
            continue

        # FAN-OUT
        sources = sorted(outdeg.keys(), key=lambda k: outdeg.get(k, 0), reverse=True)
        for src in sources:
            if src not in declared_nodes:
                continue
            src_in = indeg.get(src, 0)
            reserve_in = max(1, src_in) if src_in > 0 else 0
            allowed_out = max(1, MAX_TOTAL_CONNECTIONS - reserve_in)

            if outdeg.get(src, 0) > allowed_out:
                cur_edges, extra = _split_tree_for_source(
                    cur_edges,
                    source_alias=src,
                    max_outgoing=allowed_out,
                    gateway_prefix="gw_split",
                )
                if extra:
                    for _line, a, _lc in extra:
                        declared_nodes.add(a)
                    extra_all.extend(extra)
                    changed = True
                    break

        if not changed:
            break

    return cur_edges, extra_all


# ================================
# PROCEDURE -> PIPERFLOW
# ================================

def procedure_to_piperflow(procedure: Dict[str, Any], idx: int) -> Tuple[str, Dict[str, Any]]:
    # ---- extract steps ----
    steps: List[Dict[str, Any]] = []
    if isinstance(procedure.get("steps"), list) and procedure["steps"]:
        steps = list(procedure["steps"])
    else:
        for point in (procedure.get("points") or []):
            steps.extend(point.get("steps") or [])

    if not steps:
        return "", {
            "num_nodes": 0,
            "num_edges": 0,
            "lane_names": [],
            "max_label_lines": 1,
            "width": AUTO_WIDTH_MIN,
        }

    for s in steps:
        _alias_for_step(s)

    raw_title = procedure.get("subsection_title") or procedure.get("section_title") or f"Procedura {idx + 1}"
    title = sanitize_title(raw_title, fallback=f"Procedura {idx + 1}")

    has_event_start = any(_is_start_event((s.get("type") or "").strip()) for s in steps)
    has_event_end = any(_is_end_event((s.get("type") or "").strip()) for s in steps)

    include_graphic_start = not has_event_start
    include_graphic_end = not has_event_end
    start_is_in_dsl = bool(include_graphic_start or has_event_start)

    # ---- agent grouping ----
    agent_to_steps: Dict[str, List[Dict[str, Any]]] = {}
    for step in steps:
        agent_raw = str(step.get("agent") or "Operatore")
        agent_to_steps.setdefault(agent_raw, []).append(step)

    agents_original_sorted = sorted(
        agent_to_steps.keys(),
        key=lambda x: (sanitize_for_piperflow(x, fallback=""), x),
    )
    agents_sanitized = [sanitize_for_piperflow(a, fallback="Operatore") for a in agents_original_sorted]
    agents_unique = _make_unique_names(agents_sanitized)
    agent_lane_map: Dict[str, str] = {orig: uniq for orig, uniq in zip(agents_original_sorted, agents_unique)}

    first_agent_orig = str(steps[0].get("agent") or "Operatore")
    first_lane = agent_lane_map.get(first_agent_orig, agents_unique[0] if agents_unique else "Operatore")

    # ---- edges base ----
    edges = _build_edges_from_branches(steps, include_graphic_start=include_graphic_start)

    # ---- declared nodes ----
    declared_nodes: Set[str] = set(_alias_for_step(s) for s in steps)
    if include_graphic_start:
        declared_nodes.add("start")
    if include_graphic_end:
        declared_nodes.add("end")

    # ---- limiters ----
    edges, extra_gateway_nodes = _apply_limiters_globally(edges, declared_nodes)

    # nota: gli anchor creati come extra_nodes sono già inclusi in extra_gateway_nodes
    # ma dobbiamo aggiungerli anche ai declared_nodes per non farli filtrare via
    for _line, a, _lc in extra_gateway_nodes:
        declared_nodes.add(a)

    # ---- start guard ----
    edges = _ensure_start_connected(edges, steps, start_is_in_dsl=start_is_in_dsl)

    # ---- safety filter ----
    edges = _filter_edges(edges, declared_nodes)

    # ---- counts ----
    node_aliases: Set[str] = set(declared_nodes)
    for s0, t0, _ in edges:
        node_aliases.add(s0)
        node_aliases.add(t0)

    num_nodes = len(node_aliases)
    num_edges = len(edges)
    width = _estimate_diagram_width(num_nodes=num_nodes, num_edges=num_edges)

    lane_names: List[str] = list(agents_unique)
    max_label_lines = 1

    # ---- DSL header ----
    lines: List[str] = []
    lines.append(f"title: {title}")
    lines.append("colourtheme: BLUEMOUNTAIN")
    lines.append(f"width: {width}")
    lines.append("")

    # Gateway lane: mettili nella first_lane
    gateway_lane = first_lane
    extra_nodes_by_lane: Dict[str, List[Tuple[str, str, int]]] = {}
    if extra_gateway_nodes:
        extra_nodes_by_lane.setdefault(gateway_lane, []).extend(extra_gateway_nodes)

    # ---- lanes ----
    for agent_orig, lane_name in zip(agents_original_sorted, agents_unique):
        lines.append(f"lane: {lane_name}")

        # graphic start (solo se serve)
        if include_graphic_start and lane_name == first_lane:
            lines.append("    (start) as start")

        # separa steps della lane: end steps stampati per ultimi
        steps_in_lane = list(agent_to_steps.get(agent_orig, []))
        normal_steps: List[Dict[str, Any]] = []
        end_steps: List[Dict[str, Any]] = []

        for st in steps_in_lane:
            a = _alias_for_step(st)
            if a == "end":
                end_steps.append(st)
            else:
                normal_steps.append(st)

        # 1) normal steps
        for step in normal_steps:
            eline, alias, lc = _piper_element_line(step)
            max_label_lines = max(max_label_lines, lc)

            # evita doppio start/end se grafici
            if alias == "start" and include_graphic_start:
                continue
            if alias == "end" and include_graphic_end:
                continue

            lines.append(f"    {eline}")

        # 2) gateway extra (incl. anchor) PRIMA dell'end (critico per ProcessPiper)
        if lane_name == gateway_lane:
            for eline, _alias, lc in extra_nodes_by_lane.get(lane_name, []):
                max_label_lines = max(max_label_lines, lc)
                lines.append(f"    {eline}")

        # 3) end steps (event_end nel JSON) SEMPRE ultimi
        for step in end_steps:
            eline, alias, lc = _piper_element_line(step)
            max_label_lines = max(max_label_lines, lc)

            if alias == "end" and include_graphic_end:
                continue

            lines.append(f"    {eline}")

        # 4) graphic end se NON esiste event_end nel JSON
        if include_graphic_end and lane_name == first_lane:
            lines.append("    (end) as end")

        lines.append("")

    # ---- flows ----
    for s0, t0, label in edges:
        flow = f"{s0}->{t0}"
        if label:
            flow += f": {label}"
        lines.append(flow)

    lines.append("")
    lines.append("footer: Generato con ProcessPiper")
    lines.append("")

    dsl = "\n".join(lines)
    meta = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "lane_names": lane_names,
        "max_label_lines": max_label_lines,
        "width": width,
        "has_event_start": has_event_start,
        "has_event_end": has_event_end,
        "include_graphic_start": include_graphic_start,
        "include_graphic_end": include_graphic_end,
        "start_is_in_dsl": start_is_in_dsl,
        "gateway_lane": gateway_lane,
        "first_lane": first_lane,
    }
    return dsl, meta


# ================================
# RENDERING
# ================================

def _write_debug_piperflow(dsl: str, svg_path: str) -> str:
    try:
        dbg_path = os.path.splitext(svg_path)[0] + ".piperflow.txt"
        with open(dbg_path, "w", encoding="utf-8") as f:
            f.write(dsl)
        return dbg_path
    except Exception:
        return ""


def generate_bpmn_from_procedure(procedure: Dict[str, Any], output_dir: str, idx: int):
    bpmn_dir = os.path.join(output_dir)
    os.makedirs(bpmn_dir, exist_ok=True)

    piper_syntax, meta = procedure_to_piperflow(procedure, idx)
    if not piper_syntax.strip():
        print(f"[SKIP] Procedura {idx} vuota")
        return None, None

    piper_syntax = sanitize_piperflow_document(piper_syntax)
    piper_syntax = piper_syntax.lstrip("\ufeff \t\r\n")

    _apply_dynamic_graphics(
        piper_syntax=piper_syntax,
        lane_names=meta.get("lane_names") or [],
        max_label_lines=int(meta.get("max_label_lines") or 1),
    )

    proc_id = (procedure.get("procedure_id") or f"proc_{idx:03d}").strip()
    base_name = sanitize_for_piperflow(proc_id, fallback=f"proc_{idx:03d}")
    base_name = re.sub(r"\s+", "_", base_name)

    svg_path = os.path.join(bpmn_dir, f"{base_name}.svg")
    safe_output = _safe_output_path(svg_path)
    safe_input = _escape_for_processpiper_python_literal(piper_syntax)
    

    try:
        render(safe_input, safe_output)
        _postprocess_svg(safe_output)
        print(f"[OK] Diagramma ProcessPiper → {safe_output}")
        return safe_output, None
    except Exception as e:
        dbg_path = _write_debug_piperflow(piper_syntax, safe_output)
        if dbg_path:
            print(f"[DEBUG] DSL salvato in: {dbg_path}")
        print(f"[ERRORE] ProcessPiper procedura {idx}: {e} vedere {dbg_path} per il DSL")
        return None, None


def generate_all_bpmn(json_file: str, output_dir: str):
    with open(json_file, "r", encoding="utf-8") as f:
        procedures = json.load(f)

    if not isinstance(procedures, list):
        print("[WARN] JSON non è una lista")
        return []

    results: List[str] = []
    for idx, proc in enumerate(procedures):
        svg_path, _ = generate_bpmn_from_procedure(proc, output_dir, idx)
        if svg_path:
            results.append(svg_path)
    return results
