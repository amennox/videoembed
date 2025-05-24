import os
import json
import base64
import requests
from pathlib import Path

# Impostazioni
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
IMAGE_DIR = Path("./dataset_images")
OUTPUT_JSON = Path("training_dataset.json")

PROMPT_TEMPLATE = """
Descrivi la schermata software elencando solo gli elementi dell’interfaccia utente (UI) e le loro funzioni, senza riferimenti visivi o commenti generali.
Elenca le funzioni presenti nel menu principale (barra blu in alto), tipicamente contiene le sezioni di navigazione principali del gestionale, ognuna rappresentata da un’etichetta testuale o icona (es. Front-Office, Controllo Accessi, Report, Planning, Istruttore, Pianificazione, Impostazioni).
Elenca le funzioni presenti nel menu secondario (barra grigia sotto il menu principale), tipicamente mostra le opzioni contestuali della sezione selezionata, con etichette testuali e icone (es. Rubrica, Proshop, Servizi, Dashboard CRM, Agenda Utenti, Giftcard/Coupon).

Descrivi dettagliatamente il contenuto del corpo centrale, tipicamente contiene il pannello funzionale della maschera. Cerca di identificare gli elementi più importanti come: elenco utenti, impostazione documento.
Per il corpo centrale elenca tutti i principali elementi di UI trovati e il contenuto del testo dei bottoni presenti e delle caselle di testo cercando di spiegarne l'uso nel contesto della maschera.

Non includere riferimenti al nome del software. Ignora elementi in basso o sulla barra nera.
Non usare il formato markdown, non usare i tag HTML, non usare le virgolette o asterischi *, non aggiungere valutazioni o tue spiegazioni.
La descrizione deve essere oggettiva, strutturata e funzionale all’individuazione degli oggetti UI e delle loro funzioni, senza spiegazioni aggiuntive.
"""

def describe_image(image_path):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": PROMPT_TEMPLATE,
        "images": [image_b64],
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json().get("response", "").strip()

def load_existing_data():
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def create_training_data():
    dataset = load_existing_data()
    processed_images = {item["image_path"] for item in dataset}

    for image_file in IMAGE_DIR.glob("*.jpg"):
        image_path_str = str(image_file.resolve())
        if image_path_str in processed_images:
            print(f"Immagine già elaborata, saltata: {image_file.name}")
            continue

        description = describe_image(image_file)
        description = description.replace("**", "")
        dataset.append({
            "image_path": image_path_str,
            "description": description
        })

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"Descritta immagine: {image_file.name} \nDescrizione: {description}\n")

    print(f"Dataset aggiornato e salvato in {OUTPUT_JSON}")

if __name__ == "__main__":
    create_training_data()
