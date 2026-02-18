# utils/io_utils.py
import re
import os
import json
from pathlib import Path
import textwrap
from typing import Any, Union

from pypdf import PdfReader
from docx import Document


PathLike = Union[str, Path]



def read_pdf_text(path: str) -> str:
    """Estrae testo selezionabile da un PDF."""
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def read_docx_text(path: str) -> str:
    """Estrae testo da un file .docx (solo paragrafi)."""
    return "\n".join(p.text for p in Document(path).paragraphs)


def read_text_file(path: str) -> str:
    """Legge un file di testo semplice in UTF-8."""
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_pdf_with_ocr(path: str, poppler_path: str = None) -> str:
    """
    Estrae testo da PDF tramite OCR (Tesseract + pdf2image).
    Fallback automatico all'estrazione nativa in caso di errore.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        print("Conversione PDF in immagini (OCR)...")
        images = convert_from_path(path, poppler_path=poppler_path)
        full_text = ""
        for i, img in enumerate(images):
            print(f" OCR pagina {i+1}/{len(images)}")
            text = pytesseract.image_to_string(img, lang="ita+eng")
            full_text += text + "\n"
        return full_text.strip()
    except Exception as e:
        print(f"Errore OCR su {path}: {e}")
        print("Procedo con estrazione nativa...")
        return read_pdf_text(path)


def read_document_file(file_path: str, use_ocr: bool = False, poppler_path: str = None) -> str:
    """
    Legge documenti nei formati supportati: PDF, DOCX, TXT.
    Per i PDF, usa OCR se richiesto.
    """
    ext = file_path.lower()
    if ext.endswith(".pdf"):
        return read_pdf_with_ocr(file_path, poppler_path) if use_ocr else read_pdf_text(file_path)
    if ext.endswith(".docx"):
        return read_docx_text(file_path)
    return read_text_file(file_path)


def sanitize_filename(filename: str) -> str:
    """
    Rende un titolo sicuro come nome di file.
    """
    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", filename)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("._")
    return sanitized[:100]


def unique_filename(base_path: str) -> str:
    """
    Genera un nome univoco aggiungendo _1, _2... se il file esiste già.
    """
    if not os.path.exists(base_path):
        return base_path
    root, ext = os.path.splitext(base_path)
    counter = 1
    while os.path.exists(f"{root}_{counter}{ext}"):
        counter += 1
    return f"{root}_{counter}{ext}"


def save_blocks_to_txt(blocks, filepath: str) -> None:
    """Salva lista di blocchi in un file di testo formattato."""
    with open(filepath, "w", encoding="utf-8") as f:
        for b in blocks:
            section = b.get("section_title") or "(senza sezione)"
            subsection = b.get("subsection_title") or "(senza titolo)"
            f.write(f"Titolo sezione: {section}\nTitolo: {subsection}\n")
            f.write(textwrap.indent(b.get("text", ""), "  "))
            f.write("\n\n")


def save_definitions_records_txt(def_records: list, filepath: str) -> None:
    """
    Salvataggio per debug.
    Ogni riga: Term | Summary | Section | Subsection | Raw
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for r in def_records:
            term = (r.get("term") or "").replace("|", "/")
            summary = (r.get("summary") or "").replace("|", "/")
            sec = (r.get("section_title") or "").replace("|", "/")
            sub = (r.get("subsection_title") or "").replace("|", "/")
            raw = (r.get("raw") or "").replace("\n", " ").replace("|", "/")
            f.write(f"{term} | {summary} | {sec} | {sub} | {raw}\n")
