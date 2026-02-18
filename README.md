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
## Output

Per ogni documento viene creata una directory dedicata in `./output/<doc_id>/` contenente JSON strutturati, file intermedi e artefatti per la valutazione.

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
