from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union
from PIL import Image
from io import BytesIO
import base64
import uvicorn
import torch
import numpy as np  # <-- IMPORTANTE
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

app = FastAPI()

class EmbedRequest(BaseModel):
    input: Union[str, List[str]]

@app.post("/api/embed")
async def embed_image(request: EmbedRequest):
    inputs = request.input if isinstance(request.input, list) else [request.input]
    embeddings = []

    for idx, item in enumerate(inputs):
        try:
            print(f"[SERVER DEBUG] Received base64 at idx={idx}, length: {len(item)} chars")

            if item.startswith("data:image"):
                item = item.split(",")[1]

            image_data = base64.b64decode(item)
            print(f"[SERVER DEBUG] Decoded image size: {len(image_data)} bytes")

            image = Image.open(BytesIO(image_data)).convert("RGB")

            # (opzionale) salva immagine decodificata per debug
            #with open(f"debug_img_{idx}.jpg", "wb") as f:
            #    f.write(image_data)

        except Exception as e:
            print(f"[ERROR] Image decode failed at idx={idx}: {e}")
            return {"error": f"Invalid base64 image at position {idx}: {e}"}

        try:
            image_array = np.array(image)  # FIX CRITICO PER COMPATIBILITÀ
            emb = model.encode([image_array])[0]
            print(f"[SERVER DEBUG] Embedding OK at idx={idx}, length={len(emb)}")
        except Exception as e:
            print(f"[ERROR] Encoding failed at idx={idx}: {e}")
            return {"error": f"Model encoding failed at position {idx}: {e}"}

        embeddings.append(emb.tolist())

    return {"embeddings": embeddings}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("docclip_api:app", host="0.0.0.0", port=11436, reload=False)
