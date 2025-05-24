import torch
import open_clip
from PIL import Image
import json
from torch.utils.data import Dataset, DataLoader
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

MODEL_NAME = "ViT-B-32"
LOCAL_MODEL_PATH = "./models/vit_b_32/open_clip_pytorch_model.bin"

model, preprocess_train, _ = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=LOCAL_MODEL_PATH
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

class UIDataset(Dataset):
    def __init__(self, json_path, transform):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["image_path"]
        description = item["description"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, description

dataset = UIDataset("training_dataset.json", preprocess_train)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # batch più piccolo

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)  # learning rate più basso

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

best_loss = float("inf")
for epoch in range(10):
    model.train()
    total_loss = 0.0
    for i, (imgs, texts) in enumerate(dataloader):
        print(f"[DEBUG] IMMAGINI: shape={imgs.shape}, min={imgs.min().item()}, max={imgs.max().item()}, sum={imgs.sum().item()}")
        texts_tok = tokenizer(texts).to(device)
        print(f"[DEBUG] TOKENIZER shape: {texts_tok.shape if hasattr(texts_tok, 'shape') else 'NO SHAPE'}")
        # Prova solo la forward SENZA backward
        with torch.no_grad():
            image_features, text_features, logit_scale = model(imgs.to(device), texts_tok)
            logit_scale_clamped = torch.clamp(logit_scale, 0, 4.6052)  # Clamp tra 0 e log(100)
            logits_per_image = image_features @ text_features.t() * logit_scale_clamped.exp()
            print(f"[DEBUG] logit_scale: {logit_scale.item()}, clamped: {logit_scale_clamped.item()}")
            print(f"[DEBUG] LOGITS: min={logits_per_image.min().item()}, max={logits_per_image.max().item()}, mean={logits_per_image.mean().item()}")
    

    loss_avg = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} COMPLETATO - Loss medio: {loss_avg:.6f}")

    # Salva il best checkpoint se migliora
    if loss_avg < best_loss:
        best_loss = loss_avg
        torch.save(model.state_dict(), "fine_tuned_openclip_best.pth")
        print(f"[INFO] Salvo nuovo best model con loss {loss_avg:.6f}")

print(f"Training terminato. Best model salvato come fine_tuned_openclip_best.pth")
