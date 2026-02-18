# diagrams/main_diagram_run.py
from __future__ import annotations

import os
from typing import List

from .run_diagrams import generate_three_diagram_sets


def _find_diagram_procedures_json_files(output_root: str) -> List[str]:
    hits: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(output_root):
        for fn in filenames:
            if fn.endswith(".diagram_procedures.json"):
                hits.append(os.path.join(dirpath, fn))
    hits.sort()
    return hits


def _infer_doc_stem(diagram_json_path: str) -> str:
    # output_root/<doc_stem>/<doc_stem>.diagram_procedures.json
    return os.path.basename(os.path.dirname(diagram_json_path))


def main() -> None:
    # lanciando da root progetto, questo punta a ./output
    output_root = os.path.abspath("./output")

    if not os.path.isdir(output_root):
        print(f"[ERRORE] output_root non esiste o non è una directory: {output_root}")
        raise SystemExit(1)

    files = _find_diagram_procedures_json_files(output_root)
    if not files:
        print(f"[WARN] Nessun *.diagram_procedures.json trovato in: {output_root}")
        return

    print(f"[INFO] Trovati {len(files)} file diagram_procedures.")

    ok = 0
    fail = 0

    for fpath in files:
        doc_stem = _infer_doc_stem(fpath)
        print(f"\n[RUN] doc_stem={doc_stem}")
        print(f"      input={fpath}")

        try:
            generated = generate_three_diagram_sets(
                diagram_procedures_json_path=fpath,
                out_root=output_root,
                doc_stem=doc_stem,
            )

            n_strategy = sum(len(x or []) for x in generated.get("strategy_driven", []))
            n_lanes = sum(len(x or []) for x in generated.get("agent_lanes", []))
            n_bpmn = sum(len(x or []) for x in generated.get("processpiper_bpmn", []))

            print(f"[OK] strategy={n_strategy} lanes={n_lanes} bpmn={n_bpmn}")
            ok += 1

        except Exception as e:
            print(f"[ERRORE] {doc_stem}: {e}")
            fail += 1

    print("\n[DONE]")
    print(f"  OK:   {ok}")
    print(f"  FAIL: {fail}")


if __name__ == "__main__":
    main()
