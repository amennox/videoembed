
# SnapshotEmbedder

SnapshotEmbedder is a project for **training a specialized embedding model (LLM)** to recognize and measure similarity between **software screenshots** (“snapshots”).
The project enables the automatic generation of descriptive captions for each snapshot and trains an image–text embedding model for use cases such as software interface recognition, duplicate screen detection, or UI analytics.

## Project Structure

* **dataset\_generator.py**
  Generates a dataset of screenshots with targeted, structured descriptions using an LLM (via Ollama). This script automates the creation of a high-quality dataset for training.
* **trainer\_videoemb.py**
  Provides training routines for fine-tuning the base embedding model using the generated dataset.
* **docclip\_api.py**
  Offers an API to generate embeddings from new snapshots and perform inference using the fine-tuned model.

## Base Model

The main embedding model used as the base is **OpenCLIP (ViT-B-32)**, a widely adopted vision-language model architecture. Fine-tuning is performed on your custom, domain-specific dataset.

## Main Dependencies

* **Python 3.10**
* [PyTorch](https://pytorch.org/)
* [OpenCLIP](https://github.com/mlfoundations/open_clip)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Uvicorn](https://www.uvicorn.org/)
* [Pillow](https://python-pillow.org/)
* [Requests](https://requests.readthedocs.io/)
* [SentenceTransformers](https://www.sbert.net/)

See [`requirements.txt`](./requirements.txt) for the full list of dependencies.

## How to Use

1. **Generate the dataset**
   Place your software screenshots in the `dataset_images` directory and run:

   ```bash
   python dataset_generator.py
   ```
2. **Train the model**
   Use the training script to fine-tune the embedding model:

   ```bash
   python trainer_videoemb.py
   ```
3. **Run the API server**
   Launch the API to generate embeddings and serve predictions:

   ```bash
   python docclip_api.py
   ```
