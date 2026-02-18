# config.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

"""
Configurazione centralizzata del sistema.
- Carica chiavi API da .env
- Carica impostazioni utente da variables.txt
- Inizializza client LLM e modello di embedding
"""

# Carica variabili d'ambiente
load_dotenv()

# === Azure OpenAI ===
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT]):
    raise ValueError("Mancano variabili Azure OpenAI in .env: controlla AZURE_OPENAI_ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT")


client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

# === Embedding model ===
# Usa un modello multilingua https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
#embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

def parse_value(value: str):
    v = value.lower()
    if v in ("true", "false"):
        return v == "true"
    if v == "none":
        return None
    return value

#test_emb = embedder.encode("test")
#print(f"[DEBUG] Dimensione embedding: {len(test_emb)}")

# === Configurazione utente (da variables.txt) ===
def load_user_config(config_file="variables.txt"):
    """
    Carica le impostazioni dal file di testo
    """
    config = {
        "input_path": "./documenti",
        "output_path": "./output",
        "delete_existing_chroma_db": True,
        # Se True, elimina la singola collection per-doc (admin_proc_{doc_id}) prima di indicizzare.
        # Utile quando rilanci più volte sugli stessi documenti senza voler mantenere stati precedenti.
        "reset_doc_collections": False,
        # Se True, NON aggiorna la global pattern bank durante la run (evita leakage in evaluation).
        "freeze_global_patterns": False,
        # Se False, disabilita del tutto la retrieval di esempi cross-doc.
        "use_crossdoc_patterns": True,
        "use_ocr_for_pdf": False,
        "poppler_path": None,
        "comparison_base_path": "./comparison_base"
    }
    
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                key, sep, value = line.partition("=")
                if not sep:
                    continue

                key = key.strip()
                value = parse_value(value.strip())
                config[key] = value

        # Rendi poppler_path assoluto se è relativo
        poppler = config.get("poppler_path")
        if poppler and not os.path.isabs(poppler):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config["poppler_path"] = os.path.normpath(
                os.path.join(script_dir, poppler)
            )

    return config