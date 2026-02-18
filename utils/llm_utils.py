#llm_utils.py
from config import client, AZURE_OPENAI_DEPLOYMENT

def call_llm(prompt: str) -> str:
    """
    Invia un prompt a Azure OpenAI e restituisce la risposta.
    Usa un system prompt fisso per analisi di documenti amministrativi.
    In caso di errore, restituisce stringa vuota.
    """
    
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", 
                 "content": (
                        "Sei un assistente analista esperto di estrazione e "
                        "individuazione di procedure amministrative nei documenti. "
                        "Rispondi sempre e solo con testo chiaro e conciso."
                    ),
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            top_p=0.8,
        )

        content = None
        try:
            content = response.choices[0].message.content
        except Exception:
            content = None

        if content is None:
            return ""

        if not isinstance(content, str):
            content = str(content)

        return content.strip()
        #return response.choices[0].message.content.strip()
    except Exception as e:
        print("Errore LLM:", e)
        return ""
 