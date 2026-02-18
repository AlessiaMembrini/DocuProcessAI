# main.py
import os
import json
import logging
from shutil import rmtree

from config import load_user_config
from utils.io_utils import read_document_file
from pipeline.procedural_rag_chroma import run_pipeline_on_text

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("pytesseract").setLevel(logging.WARNING)
logging.getLogger("pdf2image").setLevel(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"


def main() -> None:
    config = load_user_config()
    input_folder = config["input_path"]
    output_folder = config["output_path"]
    os.makedirs(output_folder, exist_ok=True)

    files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".pdf", ".docx", ".txt"))
    ]

    if not files:
        print(f"[ERRORE] Nessun file valido trovato in {input_folder}")
        raise SystemExit(1)

    persist_dir = config.get("chroma_dir", ".chroma_admin_procedural")

    # Reset totale del DB Chroma (attenzione: cancella anche la global pattern bank).
    if config.get("delete_existing_chroma_db", False) and os.path.isdir(persist_dir):
        rmtree(persist_dir, ignore_errors=True)

    for path in files:
        base_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n[RAG] Elaborazione file: {base_name}")

        is_pdf = path.lower().endswith(".pdf")

        text = read_document_file(
            path,
            use_ocr=(config.get("use_ocr_for_pdf", False) if is_pdf else False),
            poppler_path=config.get("poppler_path"),
        )

        if not text or not text.strip():
            print(f"[SKIP] Testo vuoto per {base_name}")
            continue

        rag = run_pipeline_on_text(
            text=text,
            doc_id=base_name,
            persist_dir=persist_dir,
            use_crossdoc_patterns=bool(config.get("use_crossdoc_patterns", True)),
            # Evita leakage in evaluation: recupera esempi ma NON aggiorna la banca.
            update_global_patterns=not bool(config.get("freeze_global_patterns", False)),
            reset_doc_collection=bool(config.get("reset_doc_collections", False)),
        )
        '''
        out_rag = os.path.join(output_folder, f"{base_name}.rag.json")
        with open(out_rag, "w", encoding="utf-8") as f:
            json.dump(rag, f, ensure_ascii=False, indent=2)

        out_diag = os.path.join(output_folder, f"{base_name}.procedures.json")
        with open(out_diag, "w", encoding="utf-8") as f:
            json.dump(rag.get("diagram_procedures", []), f, ensure_ascii=False, indent=2)
        
        from diagrams.run_diagrams import generate_three_diagram_sets
        res = generate_three_diagram_sets(
            diagram_procedures_json_path=out_diag,
            out_root="./output",
            doc_stem=base_name,   # oppure doc_id
        )
        print("[OK] Salvato:", out_rag)
        print("[OK] Salvato:", out_diag)
        print("Stats:", rag.get("stats", {}))
        
        print(f"[DIAGRAMS] strategy: {len(res['strategy_driven'])} procedure")
        print(f"[DIAGRAMS] lanes:    {len(res['agent_lanes'])} procedure")
        print(f"[DIAGRAMS] bpmn:     {len(res['processpiper_bpmn'])} procedure")
        '''

if __name__ == "__main__":
    main()
