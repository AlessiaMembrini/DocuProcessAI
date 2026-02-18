# DocuProcessAI
Progetto di Tesi Magistrale

Pipeline sperimentale per l’estrazione e la modellazione di **procedure amministrative** a partire da documenti non strutturati (PDF, DOCX, TXT), con integrazione di **Large Language Models (LLM)** e meccanismo **Retrieval-Augmented Generation (RAG)**.

Il progetto è stato sviluppato come supporto sperimentale a una tesi magistrale, con l’obiettivo di analizzare l’impatto del retrieval semantico sulla completezza, coerenza e stabilità dell’estrazione procedurale.

---

## Obiettivo

I documenti amministrativi presentano:

- struttura gerarchica non uniforme  
- variabilità lessicale  
- definizioni distribuite nel testo  
- sezioni normative e discorsive mescolate  
- presenza di rumore OCR  

DocuProcessAI affronta il problema tramite una pipeline ibrida che combina:

- euristiche strutturali  
- indicizzazione vettoriale (ChromaDB)  
- retrieval gerarchico parent/child  
- classificazione e generazione tramite LLM  
- estrazione strutturata in formato JSON  
- generazione opzionale di diagrammi di processo  

---

## Architettura della pipeline

1. Pre-processing e normalizzazione del testo  
2. Ricostruzione della gerarchia documentale  
3. Segmentazione in sezioni e chunk  
4. Indicizzazione vettoriale (parent / child)  
5. Retrieval gerarchico (con fallback flat)  
6. Estrazione strutturata di procedure, agenti e definizioni  
7. Generazione diagrammi (opzionale)  

---

## Struttura del repository

```
.
├── main.py
├── config.py
├── variables.txt
│
├── pipeline/
│   ├── procedural_rag_chroma.py
│   ├── procedural_no_rag_llm.py
│   ├── extraction.py
│   ├── text_processing.py
│   └── llm_filters.py
│
├── utils/
│   ├── chroma_utils.py
│   ├── llm_utils.py
│   ├── prompts.py
│   ├── pipeline_utils.py
│   └── io_utils.py
│
├── diagrams/
│   └── main_diagram_run.py
│
├── eval/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requisiti

- Python >= 3.10  
- Accesso a un servizio LLM (OpenAI / Azure OpenAI)  
- Ambiente virtuale consigliato  

---

## Installazione

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux / Mac:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Configurazione

### File `.env`
Contiene le credenziali del servizio LLM.

### File `variables.txt`
Contiene i parametri di esecuzione della pipeline (input/output, OCR, modalità valutazione, pattern cross-document).

---

## Esecuzione

Pipeline completa (RAG):
```bash
python main.py
```

Variante senza RAG:
```bash
python pipeline/procedural_no_rag_llm.py
```

Rigenerazione diagrammi:
```bash
python diagrams/main_diagram_run.py
```

---

## Output

Per ogni documento viene creata una directory dedicata in `./output/<doc_id>/` contenente JSON strutturati, file intermedi e artefatti per la valutazione.

---

## Modalità di valutazione

Il sistema supporta confronti tra:
- pipeline con RAG e senza RAG  
- uso o esclusione di pattern cross-document  
- modalità evaluation con congelamento delle collezioni  

---

## Dipendenze principali

- chromadb  
- openai  
- sentence-transformers  
- python-dotenv  
- numpy  
- tqdm  
- pypdf  
- pdf2image  
- pytesseract  
- processpiper  
- graphviz  
- scikit-learn  

---

## Limitazioni

- Sensibilità alla qualità dell’OCR  
- Dipendenza dai prompt LLM  
- Assenza di una ground truth completa  
- Sistema non general-purpose  

---

## Licenza

MIT License  
Copyright (c) 2026 Alessia Membrini

---

## Citazione

```
DocuProcessAI — Pipeline RAG/LLM per l’estrazione e la modellazione di procedure amministrative da documenti non strutturati.
Tesi di Laurea Magistrale, 2026.
```
