from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Optional
from PIL import Image
from io import BytesIO
import base64
import uvicorn
import torch
import numpy as np
import open_clip

# Mappatura modelli disponibili
AVAILABLE_MODELS = {
    "fine-tuned-openclip": "./fine_tuned_openclip_best.pth",
    "docclip": "sentence-transformers/clip-ViT-B-32-multilingual-v1",
    "clip-vit-b32": "sentence-transformers/clip-ViT-B-32",
    "clip-laion-b32": "clip-laion-b32",
}

# Cache per modelli caricati
MODEL_CACHE = {}
DEFAULT_MODEL_NAME = "fine-tuned-openclip"

# Caricamento modello OpenCLIP fine-tuned
def load_finetuned_openclip(model_path: str):
    model_name = "ViT-B-32"
    model, preprocess, tokenizer = open_clip.create_model_and_transforms(model_name)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, preprocess



def load_finetuned_openclip(model_path: str):
    model_name = "ViT-B-32"
    model, preprocess, _ = open_clip.create_model_and_transforms(model_name)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, preprocess, device


# Funzione caricamento generica
def get_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Modello '{model_name}' non supportato.")

    if model_name not in MODEL_CACHE:
        print(f"[SERVER DEBUG] Carico modello '{model_name}'...")
        if model_name == "fine-tuned-openclip":
            MODEL_CACHE[model_name] = load_finetuned_openclip(AVAILABLE_MODELS[model_name])
        else:
            from sentence_transformers import SentenceTransformer
            MODEL_CACHE[model_name] = SentenceTransformer(AVAILABLE_MODELS[model_name])

    return MODEL_CACHE[model_name]

app = FastAPI()

class EmbedRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = DEFAULT_MODEL_NAME


@app.post("/api/embed")
async def embed_image(request: EmbedRequest):
    model_name = request.model or DEFAULT_MODEL_NAME

    try:
        model_data = get_model(model_name)
    except Exception as e:
        return {"error": f"Errore caricamento modello: {e}"}

    inputs = request.input if isinstance(request.input, list) else [request.input]
    embeddings = []

    for idx, item in enumerate(inputs):
        try:
            if item.startswith("data:image"):
                item = item.split(",")[1]
            image_data = base64.b64decode(item)
            image = Image.open(BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return {"error": f"Immagine base64 non valida (posizione {idx}): {e}"}

        try:
            if model_name == "fine-tuned-openclip":
                model, preprocess, device = model_data
                image_tensor = preprocess(image).unsqueeze(0).to(device)
                print(f"[DEBUG] Tensor sum: {image_tensor.sum().item()}, min: {image_tensor.min().item()}, max: {image_tensor.max().item()}")
                with torch.no_grad():
                    image_features = model.encode_image(image_tensor)
                    emb = image_features.cpu().numpy()[0]
                print(f"[SERVER DEBUG] Embedding immagine {idx} calcolato. Primi 5 valori: {emb[:5]}")
            else:
                emb = model_data.encode([image])[0]
        except Exception as e:
            return {"error": f"Errore embedding (posizione {idx}): {e}"}

        embeddings.append(emb.tolist())

    return {"embeddings": embeddings, "model": model_name}

@app.get("/health")
def health():
    return {"status": "ok", "models": list(AVAILABLE_MODELS.keys())}

if __name__ == "__main__":
    uvicorn.run("docclip_api:app", host="0.0.0.0", port=11436, reload=False)