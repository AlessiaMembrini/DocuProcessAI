# eval/evaluate.py
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# IO
# ============================================================

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


# ============================================================
# NORMALIZZAZIONE (generale)
# ============================================================

def normalize_whitespace(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text)
    s = s.replace("\t", " ")
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \u00A0]+", " ", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_strong(text: Optional[str]) -> str:
    s = normalize_whitespace(text).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_title_key(text: Optional[str]) -> str:
    s = normalize_whitespace(text)
    s = re.sub(r"^\s*\d+(\.\d+){0,6}\s+", "", s)  # 1.2.3 Titolo
    s = re.sub(r"^\s*pagina\s+\d+\s+di\s+\d+\s*", "", s, flags=re.IGNORECASE)
    return normalize_strong(s)


def short_excerpt(text: str, limit: int = 180) -> str:
    s = normalize_whitespace(text)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else (s[: limit - 3] + "...")


# ============================================================
# NORMALIZZAZIONE AGENTI (MINIMA, controllata)
#   - lower
#   - collassa acronimi puntati: "S.U.A.P." -> "suap"
#   - rimuove punteggiatura (ma NON stopwords)
#   - whitespace clean
# ============================================================

_ACRONYM_DOTTED_RE = re.compile(r"\b(?:[a-z]\.\s*){2,}[a-z]\.?\b", flags=re.IGNORECASE)


def _collapse_dotted_acronyms(s: str) -> str:
    """
    Collassa acronimi con punti:
      "s.u.a.p."   -> "suap"
      "a.s.l."     -> "asl"
      "c.c.i.a.a." -> "cciaa"
    """

    def repl(m: re.Match) -> str:
        return re.sub(r"[^a-zA-Z]", "", m.group(0)).lower()

    return _ACRONYM_DOTTED_RE.sub(repl, s)


def normalize_agent(text: Optional[str]) -> str:
    """
    Normalizzazione agenti (MINIMA):
    - lower
    - collassa acronimi puntati
    - rimuove punteggiatura
    - NON rimuove stopwords (così non perdi informazione distintiva tipo "servizio", "ufficio", ecc.)
    """
    s = normalize_whitespace(text).lower()
    s = _collapse_dotted_acronyms(s)
    s = re.sub(r"[^\w\s]", " ", s)       # toglie solo punteggiatura
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================
# METRICHE
# ============================================================

def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def jaccard(a: Set[str], b: Set[str]) -> float:
    a2 = set(a or set())
    b2 = set(b or set())
    union = a2 | b2
    return (len(a2 & b2) / len(union)) if union else 0.0


def prf_on_sets(ref: Set[str], pred: Set[str]) -> Dict[str, Any]:
    ref2 = set(ref or set())
    pred2 = set(pred or set())
    tp = len(ref2 & pred2)
    fp = len(pred2 - ref2)
    fn = len(ref2 - pred2)
    return {
        **prf(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ref_count": len(ref2),
        "pred_count": len(pred2),
        "matches_count": tp,
        "jaccard": jaccard(ref2, pred2),
    }


# ============================================================
# SIMILARITY + MATCHING (titoli / procedure)
# ============================================================

def _tokenize(s: str) -> List[str]:
    s = normalize_strong(s)
    return [t for t in s.split(" ") if t]


def _containment_like_score(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    inter = len(sa & sb)
    union = len(sa | sb)
    j = inter / union if union else 0.0
    small = sa if len(sa) <= len(sb) else sb
    cont = inter / len(small) if small else 0.0
    return max(j, cont)


def _coverage_score(a: str, b: str) -> float:
    """
    Coverage token-level:
      cov(a in b) = |tokens(a) ∩ tokens(b)| / |tokens(a)|
    poi simmetrico: max(cov(a in b), cov(b in a))
    """
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    sa = set(ta)
    sb = set(tb)
    inter = len(sa & sb)
    cov_a = inter / max(1, len(sa))
    cov_b = inter / max(1, len(sb))
    return float(max(cov_a, cov_b))


def _tfidf_cosine(ref: List[str], pred: List[str], *, analyzer: str, ngram_range: Tuple[int, int]) -> np.ndarray:
    if not ref or not pred:
        return np.zeros((len(ref), len(pred)), dtype=float)
    corpus = ref + pred
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, lowercase=True, min_df=1)
    X = vec.fit_transform(corpus)
    Xr = X[: len(ref)]
    Xp = X[len(ref):]
    return cosine_similarity(Xr, Xp)


def _length_penalty(a: str, b: str, alpha: float = 0.6) -> float:
    la = max(1, len(_tokenize(a)))
    lb = max(1, len(_tokenize(b)))
    ratio = min(la, lb) / max(la, lb)
    return float(ratio ** alpha)


def similarity_matrix(reference: List[str], predicted: List[str], *, title_mode: bool) -> np.ndarray:
    """
    Similarity robusta:
      - TF-IDF char_wb
      - TF-IDF word
      - token containment
      - (procedure) coverage score
    """
    if not reference or not predicted:
        return np.zeros((len(reference), len(predicted)), dtype=float)

    ref = [normalize_whitespace(x) for x in (reference or [])]
    pred = [normalize_whitespace(x) for x in (predicted or [])]

    if title_mode:
        ref_n = [normalize_title_key(x) for x in ref]
        pred_n = [normalize_title_key(x) for x in pred]
        char_range = (3, 5)
        word_range = (1, 2)
    else:
        ref_n = [normalize_strong(x) for x in ref]
        pred_n = [normalize_strong(x) for x in pred]
        char_range = (3, 6)
        word_range = (1, 2)

    sim_char = _tfidf_cosine(ref_n, pred_n, analyzer="char_wb", ngram_range=char_range)
    sim_word = _tfidf_cosine(ref_n, pred_n, analyzer="word", ngram_range=word_range)

    sim_cont = np.zeros((len(ref_n), len(pred_n)), dtype=float)
    sim_cov = np.zeros((len(ref_n), len(pred_n)), dtype=float)
    for i in range(len(ref_n)):
        for j in range(len(pred_n)):
            sim_cont[i, j] = _containment_like_score(ref_n[i], pred_n[j])
            if not title_mode:
                sim_cov[i, j] = _coverage_score(ref_n[i], pred_n[j])

    sim = np.maximum(sim_char, np.maximum(sim_word, sim_cont))
    if not title_mode:
        sim = np.maximum(sim, sim_cov)

    sim = np.clip(sim, 0.0, 1.0)

    if not title_mode:
        for i in range(len(ref_n)):
            for j in range(len(pred_n)):
                sim[i, j] *= _length_penalty(ref_n[i], pred_n[j], alpha=0.6)

    return sim


def optimal_match_1to1(
    reference: List[str],
    predicted: List[str],
    threshold: float,
    *,
    title_mode: bool,
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    if not reference:
        return 0, len(predicted), 0, []
    if not predicted:
        return 0, 0, len(reference), []

    sim = similarity_matrix(reference, predicted, title_mode=title_mode)
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)

    matches: List[Dict[str, Any]] = []
    used_pred: Set[int] = set()
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        score = float(sim[r, c])
        if score >= threshold:
            used_pred.add(c)
            matches.append(
                {
                    "ref_index": int(r),
                    "pred_index": int(c),
                    "score": round(score, 4),
                    "ref_excerpt": short_excerpt(reference[r]),
                    "pred_excerpt": short_excerpt(predicted[c]),
                }
            )

    tp = len(matches)
    fp = len(predicted) - len(used_pred)
    fn = len(reference) - tp
    return tp, fp, fn, matches


def greedy_match_1to1(
    reference: List[str],
    predicted: List[str],
    threshold: float,
    *,
    normalize_fn,
) -> Tuple[int, int, int, List[Dict[str, Any]]]:
    used_pred: Set[int] = set()
    matches: List[Dict[str, Any]] = []

    def norm(x: str) -> str:
        return normalize_fn(x)

    for ref_index, ref_text in enumerate(reference):
        ref_n = norm(ref_text)
        if not ref_n:
            continue

        best_pred_index: Optional[int] = None
        best_score = -1.0
        best_containment = False

        for pred_index, pred_text in enumerate(predicted):
            if pred_index in used_pred:
                continue

            pred_n = norm(pred_text)
            if not pred_n:
                continue

            containment = (ref_n in pred_n) or (pred_n in ref_n)
            score = SequenceMatcher(None, ref_n, pred_n).ratio()
            if containment:
                score = max(score, 0.95)

            if score > best_score:
                best_score = score
                best_pred_index = pred_index
                best_containment = containment

        if best_pred_index is not None and best_score >= threshold:
            used_pred.add(best_pred_index)
            matches.append(
                {
                    "ref_index": ref_index,
                    "pred_index": best_pred_index,
                    "score": round(best_score, 4),
                    "containment": bool(best_containment),
                    "ref_excerpt": short_excerpt(ref_text),
                    "pred_excerpt": short_excerpt(predicted[best_pred_index]),
                }
            )

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(reference) - tp
    return tp, fp, fn, matches


def eval_list_vs_list(ref: List[str], pred: List[str], threshold: float, *, title_mode: bool) -> Dict[str, Any]:
    try:
        tp, fp, fn, matches = optimal_match_1to1(ref, pred, threshold, title_mode=title_mode)
        algo = "hungarian_tfidf+coverage"
    except Exception:
        tp, fp, fn, matches = greedy_match_1to1(
            reference=ref,
            predicted=pred,
            threshold=threshold,
            normalize_fn=(normalize_title_key if title_mode else normalize_strong),
        )
        algo = "greedy_difflib"

    return {
        **prf(tp, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ref_count": len(ref),
        "pred_count": len(pred),
        "matches_count": len(matches),
        "threshold": threshold,
        "match_algo": algo,
    }


# ============================================================
# AGENTI (GOLD) - Alias groups -> canonici, pred canonicalizzati
# ============================================================

def load_gold_agents_aliases(file_path: Path) -> List[Set[str]]:
    """
    Ogni riga: alias1 | alias2 | alias3 ...
    -> gruppo di alias normalizzati MINIMAMENTE (normalize_agent).
    """
    out: List[Set[str]] = []
    if not file_path.exists():
        return out

    for line in read_text_file(file_path).splitlines():
        line = normalize_whitespace(line)
        if not line:
            continue
        parts = [normalize_agent(p) for p in line.split("|")]
        parts = [p for p in parts if p]
        if parts:
            out.append(set(parts))
    return out


def build_gold_agent_canon_map(gold_alias_groups: List[Set[str]]) -> Tuple[Dict[str, str], Set[str]]:
    """
    Ritorna:
      - alias_to_canon: alias -> canonico del suo gruppo
      - gold_canon_set: set di canonici (uno per gruppo)

    Canonico: stringa più corta del gruppo (tie-break alfabetico) per stabilità.
    """
    alias_to_canon: Dict[str, str] = {}
    canon_set: Set[str] = set()

    for g in gold_alias_groups or []:
        gg = sorted([a for a in g if a], key=lambda x: (len(x), x))
        if not gg:
            continue
        canon = gg[0]
        canon_set.add(canon)
        for a in gg:
            alias_to_canon[a] = canon

    return alias_to_canon, canon_set


def canonicalize_pred_agents_with_gold(
    pred_agents_set: Set[str],
    gold_alias_groups: List[Set[str]],
    *,
    threshold: float = 0.80,
) -> Set[str]:
    """
    Mappa ciascun agente predetto al canonico gold più simile (se >= threshold).
    Se non matcha nessun gruppo gold: lo lascia com'è (già normalizzato minimal).
    """
    pred_agents = sorted(set(pred_agents_set or set()))
    if not pred_agents:
        return set()

    if not gold_alias_groups:
        return set(pred_agents)

    alias_to_canon, _gold_canon = build_gold_agent_canon_map(gold_alias_groups)

    group_alias_lists: List[List[str]] = []
    group_canons: List[str] = []
    for g in gold_alias_groups:
        gg = sorted([a for a in g if a])
        if not gg:
            continue
        canon = sorted(gg, key=lambda x: (len(x), x))[0]
        group_alias_lists.append(gg)
        group_canons.append(canon)

    out: Set[str] = set()

    for p in pred_agents:
        if p in alias_to_canon:
            out.add(alias_to_canon[p])
            continue

        best_canon: Optional[str] = None
        best_score = 0.0

        for aliases, canon in zip(group_alias_lists, group_canons):
            # similarity max tra p e gli alias del gruppo
            sim_char = _tfidf_cosine(aliases, [p], analyzer="char_wb", ngram_range=(3, 5))
            sim_word = _tfidf_cosine(aliases, [p], analyzer="word", ngram_range=(1, 2))
            sc = float(
                max(
                    sim_char.max() if sim_char.size else 0.0,
                    sim_word.max() if sim_word.size else 0.0,
                )
            )
            if sc > best_score:
                best_score = sc
                best_canon = canon

        if best_canon is not None and best_score >= threshold:
            out.add(best_canon)
        else:
            out.add(p)

    return out


# ============================================================
# GOLD LOADERS
# ============================================================

def load_gold_procedures(file_path: Path) -> List[str]:
    txt = read_text_file(file_path).strip().strip('"').strip()
    if not txt:
        return []
    if "-----" in txt:
        return [p.strip() for p in txt.split("-----") if p.strip()]
    return [txt]


def load_gold_lines(file_path: Path) -> List[str]:
    out: List[str] = []
    for line in read_text_file(file_path).splitlines():
        line = normalize_whitespace(line)
        if line:
            out.append(line)
    return out


def load_gold_agents_by_procedure(file_path: Path) -> List[Set[str]]:
    txt = read_text_file(file_path).strip()
    if not txt:
        return []
    blocks = [b.strip() for b in txt.split("-----") if b.strip()]
    out: List[Set[str]] = []
    for b in blocks:
        raw = re.split(r"[\n,]", b)
        agents = {normalize_agent(x) for x in raw if normalize_agent(x)}
        out.append(agents)
    return out


# ============================================================
# PRED EXTRACTORS
# ============================================================

def extract_titles_from_result_json(result: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for x in (result.get("outline") or []):
        if isinstance(x, dict):
            t = normalize_whitespace(x.get("title") or "")
            if t:
                out.append(t)
        elif isinstance(x, str):
            t = normalize_whitespace(x)
            if t:
                out.append(t)
    return out


def extract_procedures_texts_from_result_json(result: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    dps = result.get("diagram_procedures") or []
    if isinstance(dps, list) and dps:
        for p in dps:
            if not isinstance(p, dict):
                continue
            txt = normalize_whitespace(p.get("original_text_full") or "")
            if txt:
                out.append(txt)
        return out

    procs = result.get("procedures") or []
    if isinstance(procs, list):
        for p in procs:
            if not isinstance(p, dict):
                continue
            txt = normalize_whitespace(p.get("original_text_full") or p.get("text") or "")
            if txt:
                out.append(txt)
    return out


def extract_procedures_with_agents_from_result_json(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dps = result.get("diagram_procedures") or []
    if not isinstance(dps, list):
        return []

    for p in dps:
        if not isinstance(p, dict):
            continue
        txt = normalize_whitespace(p.get("original_text_full") or "")
        if not txt:
            continue

        agents: Set[str] = set()
        for s in (p.get("steps") or []):
            if not isinstance(s, dict):
                continue
            a = normalize_agent(s.get("agent") or "")
            if a:
                agents.add(a)

        out.append({"text": txt, "agents": agents})
    return out


def extract_agents_set_from_result_json(result: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    dps = result.get("diagram_procedures") or []
    if isinstance(dps, list):
        for p in dps:
            if not isinstance(p, dict):
                continue
            for s in (p.get("steps") or []):
                if not isinstance(s, dict):
                    continue
                a = normalize_agent(s.get("agent") or "")
                if a:
                    out.add(a)
    return out


# ============================================================
# ONE DOCUMENT EVALUATION
# ============================================================

def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = load_json_file(path)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def evaluate_one_doc(
    doc_name: str,
    *,
    output_dir: Path,
    gold_base: Optional[Path],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    rag_dir = output_dir / doc_name
    guided_dir = output_dir / f"{doc_name}_NO_RAG_GUIDED"
    only_dir = output_dir / f"{doc_name}_NO_RAG_ONLY"

    rag_path = rag_dir / f"{doc_name}.rag.result.json"
    guided_path = guided_dir / f"{doc_name}_NO_RAG_GUIDED.no_rag_guided.result.json"
    only_path = only_dir / f"{doc_name}_NO_RAG_ONLY.no_rag_only.result.json"

    rag = _safe_load(rag_path)
    guided = _safe_load(guided_path)
    only = _safe_load(only_path)

    res: Dict[str, Any] = {
        "doc": doc_name,
        "thresholds": thresholds,
        "has_gold": False,
        "gold_vs": {},
        "pairwise": {},
        "files": {
            "rag_result": str(rag_path),
            "guided_result": str(guided_path),
            "only_result": str(only_path),
        },
    }

    preds: Dict[str, Dict[str, Any]] = {}
    for name, obj in [("rag", rag), ("no_rag_guided", guided), ("no_rag_only", only)]:
        if obj is None:
            preds[name] = {"titles": [], "procedures": [], "procedures_with_agents": [], "agents": set(), "missing": True}
        else:
            preds[name] = {
                "titles": extract_titles_from_result_json(obj),
                "procedures": extract_procedures_texts_from_result_json(obj),
                "procedures_with_agents": extract_procedures_with_agents_from_result_json(obj),
                "agents": extract_agents_set_from_result_json(obj),
                "missing": False,
            }

    # Pairwise comparisons (agents = set unici normalizzati MINIMAL)
    sys_names = ["rag", "no_rag_guided", "no_rag_only"]
    for i in range(len(sys_names)):
        for j in range(i + 1, len(sys_names)):
            a = sys_names[i]
            b = sys_names[j]
            if preds[a]["missing"] or preds[b]["missing"]:
                continue
            key = f"{a}_vs_{b}"
            res["pairwise"][key] = {
                "titles": eval_list_vs_list(preds[a]["titles"], preds[b]["titles"], thresholds["titles"], title_mode=True),
                "procedures": eval_list_vs_list(preds[a]["procedures"], preds[b]["procedures"], thresholds["procedures"], title_mode=False),
                "agents": {
                    **prf_on_sets(set(preds[a]["agents"]), set(preds[b]["agents"])),
                    "note": "agents as unique set (normalize_agent minimal)",
                },
            }

    # GOLD comparisons
    if gold_base:
        gold_dir = gold_base / doc_name
        gold_titles_path = gold_dir / "gold_titles.txt"
        gold_procs_path = gold_dir / "gold_procedures.txt"
        gold_agents_path = gold_dir / "gold_agents.txt"
        gold_agents_by_proc_path = gold_dir / "gold_agents_by_procedure.txt"

        has_any = gold_titles_path.exists() or gold_procs_path.exists() or gold_agents_path.exists()
        res["has_gold"] = bool(has_any)

        gold_titles = load_gold_lines(gold_titles_path) if gold_titles_path.exists() else []
        gold_procs = load_gold_procedures(gold_procs_path) if gold_procs_path.exists() else []
        gold_alias_groups = load_gold_agents_aliases(gold_agents_path) if gold_agents_path.exists() else []
        gold_agents_by_proc = load_gold_agents_by_procedure(gold_agents_by_proc_path) if gold_agents_by_proc_path.exists() else []

        # canonici gold (uno per alias-group)
        _alias_to_canon, gold_canon_set = build_gold_agent_canon_map(gold_alias_groups)

        for name in sys_names:
            if preds[name]["missing"]:
                continue

            block: Dict[str, Any] = {}

            if gold_titles:
                block["titles"] = eval_list_vs_list(gold_titles, preds[name]["titles"], thresholds["titles"], title_mode=True)

            if gold_procs:
                block["procedures"] = eval_list_vs_list(gold_procs, preds[name]["procedures"], thresholds["procedures"], title_mode=False)

            # AGENTS (GOLD): alias-groups -> canon set; pred canonicalizzati (fuzzy) -> set PRF/Jaccard
            if gold_alias_groups:
                thr_agents = 0.80
                pred_canon = canonicalize_pred_agents_with_gold(set(preds[name]["agents"]), gold_alias_groups, threshold=thr_agents)
                block["agents"] = {
                    **prf_on_sets(set(gold_canon_set), set(pred_canon)),
                    "threshold": thr_agents,
                    "match_algo": "gold_alias_canonicalize_then_set_prf",
                    "note": "gold agents as alias-groups -> canon set; predicted canonicalized via fuzzy match then set PRF/Jaccard",
                }

            # agents_by_procedure (macro avg su procedure matchate) - invariato (usa normalize_agent minimal)
            if gold_procs and gold_agents_by_proc and len(gold_agents_by_proc) == len(gold_procs):
                try:
                    proc_texts_pred = [x.get("text", "") for x in preds[name]["procedures_with_agents"]]
                    _tp, _fp, _fn, matches = optimal_match_1to1(
                        gold_procs, proc_texts_pred, thresholds["procedures"], title_mode=False
                    )

                    p_list: List[float] = []
                    r_list: List[float] = []
                    f1_list: List[float] = []

                    for m in matches:
                        gi = int(m["ref_index"])
                        pj = int(m["pred_index"])
                        gold_set = set(gold_agents_by_proc[gi])
                        pred_set = set((preds[name]["procedures_with_agents"][pj] or {}).get("agents") or set())
                        scores = prf_on_sets(gold_set, pred_set)
                        p_list.append(float(scores.get("precision", 0.0)))
                        r_list.append(float(scores.get("recall", 0.0)))
                        f1_list.append(float(scores.get("f1", 0.0)))

                    block["agents_by_procedure"] = {
                        "precision": sum(p_list) / len(p_list) if p_list else 0.0,
                        "recall": sum(r_list) / len(r_list) if r_list else 0.0,
                        "f1": sum(f1_list) / len(f1_list) if f1_list else 0.0,
                        "matches_count": len(matches),
                        "note": "macro avg over matched procedures (agents normalized with normalize_agent minimal)",
                    }
                except Exception as e:
                    block["agents_by_procedure"] = {"error": str(e)}

            res["gold_vs"][name] = block

    return res


# ============================================================
# AGGREGATE
# ============================================================

def aggregate(all_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    out: Dict[str, Any] = {"gold_vs": {}, "pairwise": {}}

    # gold_vs aggregation
    for d in all_docs:
        g = d.get("gold_vs", {})
        if not isinstance(g, dict):
            continue
        for sys_name, blocks in g.items():
            if not isinstance(blocks, dict):
                continue
            for block_name in ["titles", "procedures", "agents"]:
                for metric in ["precision", "recall", "f1", "jaccard"]:
                    v = blocks.get(block_name, {}).get(metric)
                    if isinstance(v, (int, float)):
                        out.setdefault("gold_vs", {}).setdefault(sys_name, {}).setdefault(block_name, {}).setdefault(metric, [])
                        out["gold_vs"][sys_name][block_name][metric].append(float(v))

    for sys_name, blocks in list(out.get("gold_vs", {}).items()):
        for block_name, metrics in list(blocks.items()):
            for metric, vals in list(metrics.items()):
                out["gold_vs"][sys_name][block_name][metric] = avg(vals) if isinstance(vals, list) else vals

    # pairwise aggregation
    for d in all_docs:
        pw = d.get("pairwise", {})
        if not isinstance(pw, dict):
            continue
        for pair_key, blocks in pw.items():
            if not isinstance(blocks, dict):
                continue
            for block_name in ["titles", "procedures", "agents"]:
                for metric in ["precision", "recall", "f1", "jaccard"]:
                    v = blocks.get(block_name, {}).get(metric)
                    if isinstance(v, (int, float)):
                        out.setdefault("pairwise", {}).setdefault(pair_key, {}).setdefault(block_name, {}).setdefault(metric, [])
                        out["pairwise"][pair_key][block_name][metric].append(float(v))

    for pair_key, blocks in list(out.get("pairwise", {}).items()):
        for block_name, metrics in list(blocks.items()):
            for metric, vals in list(metrics.items()):
                out["pairwise"][pair_key][block_name][metric] = avg(vals) if isinstance(vals, list) else vals

    return out


# ============================================================
# MAIN API
# ============================================================

def run_evaluation(
    *,
    output_dir: str,
    comparison_base: Optional[str] = None,
    out_name: str = "evaluation_summary.json",
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    if not out_dir.exists():
        raise FileNotFoundError(f"output_dir not found: {output_dir}")

    gold_base = Path(comparison_base) if comparison_base else None
    if gold_base and not gold_base.exists():
        gold_base = None

    thr = thresholds or {"titles": 0.85, "procedures": 0.70}

    doc_names = sorted(
        [
            p.name
            for p in out_dir.iterdir()
            if p.is_dir()
            and not (p.name.endswith("_NO_RAG_GUIDED") or p.name.endswith("_NO_RAG_ONLY"))
        ]
    )

    summary: Dict[str, Any] = {
        "docs_evaluated": 0,
        "docs": [],
        "aggregate": {},
        "gold_enabled": bool(gold_base),
        "thresholds": thr,
    }

    results: List[Dict[str, Any]] = []
    for doc in doc_names:
        r = evaluate_one_doc(doc, output_dir=out_dir, gold_base=gold_base, thresholds=thr)
        summary["docs"].append(r)

        # Conta valutato solo se esiste almeno un result json su disco.
        any_pred = False
        files = r.get("files", {}) or {}
        if isinstance(files, dict):
            for k in ["rag_result", "guided_result", "only_result"]:
                fp = files.get(k)
                if fp and Path(fp).exists():
                    any_pred = True
                    break

        if any_pred:
            summary["docs_evaluated"] += 1
            results.append(r)

    if results:
        summary["aggregate"] = aggregate(results)

    out_path = out_dir / out_name
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Evaluation scritto in: {out_path}")
    return summary
