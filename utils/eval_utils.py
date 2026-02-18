# utils/eval_utils.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple, Any

def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\sàèéìòóù]", "", s)
    return s

def token_set(s: str) -> set:
    return set(norm_text(s).split())

def f1_set(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    inter = len(pred & gold)
    p = inter / max(1, len(pred))
    r = inter / max(1, len(gold))
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def best_match_f1(query: str, candidates: List[str]) -> Tuple[float, int]:
    q = token_set(query)
    best = 0.0
    best_i = -1
    for i, c in enumerate(candidates or []):
        sc = f1_set(q, token_set(c))
        if sc > best:
            best = sc
            best_i = i
    return best, best_i

def outline_score(pred_titles: List[str], gold_titles: List[str]) -> Dict[str, float]:
    # match by best f1
    matched = 0
    total = max(1, len(gold_titles))
    pred_used = set()
    for g in gold_titles:
        best, bi = best_match_f1(g, pred_titles)
        if bi >= 0 and best >= 0.72 and bi not in pred_used:
            matched += 1
            pred_used.add(bi)
    recall = matched / total
    precision = matched / max(1, len(pred_titles))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}

def proc_flags_score(pred_flags: List[Dict[str, Any]], gold_flags: List[Dict[str, Any]]) -> Dict[str, float]:
    # join by section_path (più robusto del titolo)
    gmap = {x["section_path"]: bool(x["is_procedure"]) for x in gold_flags}
    pmap = {x["section_path"]: bool(x["is_procedure"]) for x in pred_flags}

    keys = sorted(set(gmap) | set(pmap))
    tp=fp=tn=fn=0
    for k in keys:
        g = gmap.get(k, False)
        p = pmap.get(k, False)
        if p and g: tp += 1
        elif p and not g: fp += 1
        elif (not p) and (not g): tn += 1
        else: fn += 1

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

def steps_score(pred_steps: List[Dict[str, Any]], gold_steps: List[Dict[str, Any]]) -> Dict[str, float]:
    # match steps by description token f1 (greedy)
    pred_desc = [x.get("description_synthetic","") for x in pred_steps]
    gold_desc = [x.get("description_synthetic","") for x in gold_steps]

    used = set()
    matched = 0
    agent_hits = 0
    for gi, g in enumerate(gold_steps):
        best, bi = best_match_f1(g.get("description_synthetic",""), pred_desc)
        if bi >= 0 and best >= 0.70 and bi not in used:
            used.add(bi)
            matched += 1
            # agent match (normalizzato minimo)
            ga = norm_text(g.get("agent",""))
            pa = norm_text(pred_steps[bi].get("agent",""))
            if ga and pa and ga == pa:
                agent_hits += 1

    recall = matched / max(1, len(gold_steps))
    precision = matched / max(1, len(pred_steps))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    agent_acc = agent_hits / max(1, matched)
    return {"precision": precision, "recall": recall, "f1": f1, "agent_acc_on_matched": agent_acc}
