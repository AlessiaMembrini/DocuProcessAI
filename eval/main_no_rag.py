# eval/main_no_rag.py
from __future__ import annotations

import os
import logging

from config import load_user_config
from utils.io_utils import read_document_file

from pipeline.procedural_no_rag_guided_llm import run_no_rag_guided_pipeline_on_text
from pipeline.procedural_no_rag_only_llm import run_no_rag_only_pipeline_on_text
from eval.evaluate import run_evaluation


logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"


def main() -> None:
    config = load_user_config()

    input_folder = config["input_path"]
    output_folder = config["output_path"]
    os.makedirs(output_folder, exist_ok=True)

    comparison_base = config.get("comparison_base") or os.path.join(".", "eval", "comparison_base")

    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".pdf", ".docx", ".txt"))
    ]

    if not files:
        print(f"[ERRORE] Nessun file valido trovato in {input_folder}")
        raise SystemExit(1)

    # ----------------------------
    # 1) ESECUZIONE NO-RAG GUIDED + ONLY
    # ----------------------------
    for path in files:
        base_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[INPUT] Documento: {base_name}")

        is_pdf = path.lower().endswith(".pdf")
        text = read_document_file(
            path,
            use_ocr=(config.get("use_ocr_for_pdf", False) if is_pdf else False),
            poppler_path=config.get("poppler_path"),
        )

        if not text or not text.strip():
            print(f"[SKIP] Testo vuoto per {base_name}")
            continue

        # GUIDED
        doc_id_guided = f"{base_name}_NO_RAG_GUIDED"
        print(f"[NO-RAG GUIDED] Run: {doc_id_guided}")
        res_g = run_no_rag_guided_pipeline_on_text(
            text=text,
            doc_id=doc_id_guided,
            output_root=output_folder,
        )
        print("[OK] Guided result:", res_g.get("outputs", {}).get("result_json"))
        print("[OK] Guided titles:", len(res_g.get("outline", [])))
        print("[OK] Guided procedures:", len(res_g.get("diagram_procedures", [])))

        # ONLY
        doc_id_only = f"{base_name}_NO_RAG_ONLY"
        print(f"[NO-RAG ONLY] Run: {doc_id_only}")
        res_o = run_no_rag_only_pipeline_on_text(
            text=text,
            doc_id=doc_id_only,
            output_root=output_folder,
            min_chars_per_chunk=int(config.get("only_llm_min_chars_per_chunk", 2000)),
            max_chars_per_chunk=int(config.get("only_llm_max_chars_per_chunk", 3000)),
            max_titles_per_chunk=int(config.get("only_llm_max_titles_per_chunk", 30)),
        )

        print("[OK] Only result:", res_o.get("outputs", {}).get("result_json"))
        print("[OK] Only titles:", len(res_o.get("outline", [])))
        print("[OK] Only procedures:", len(res_o.get("diagram_procedures", [])))

    # ----------------------------
    # 2) VALUTAZIONE (una volta alla fine)
    # ----------------------------
    print("\n[EVAL] Avvio valutazione...")
    if not os.path.isdir(comparison_base):
        print(f"[WARN] comparison_base non trovato: {comparison_base}")
        print("[WARN] Salto valutazione (gold non disponibile).")
        return

    summary = run_evaluation(
        output_dir=output_folder,
        comparison_base=comparison_base,
        thresholds={
            "titles": float(config.get("eval_titles_threshold", 0.85)),
            "procedures": float(config.get("eval_procedures_threshold", 0.70)),
        },
    )

    print("[EVAL] Completata.")
    print("[EVAL] Documenti valutati:", summary.get("docs_evaluated"))
    print("[EVAL] Gold enabled:", summary.get("gold_enabled"))
    print("[EVAL] Output:", os.path.join(output_folder, "evaluation_summary.json"))


if __name__ == "__main__":
    main()
