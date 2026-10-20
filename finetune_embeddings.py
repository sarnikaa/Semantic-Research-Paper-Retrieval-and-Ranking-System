"""
Lightweight Domain Adaptation / Fine-tuning Script:
Fine-tunes a SentenceTransformer (all-MiniLM-L6-v2) on scientific (Title -> Abstract) pairs
using MultipleNegativesRankingLoss to demonstrate in-domain representation learning.
"""

import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

from src.data_loader import load_beir_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_embeddings")


def finetune_on_scientific_corpus(
    dataset_name: str = "scifact",
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str = "models/finetuned_minilm_scifact",
    max_train_samples: int = 300,
    batch_size: int = 16,
    epochs: int = 1
):
    print("=" * 80)
    print("  DOMAIN ADAPTATION: CONTRASTIVE FINE-TUNING ON TITLE -> ABSTRACT PAIRS")
    print("=" * 80)

    # 1. Load corpus
    corpus, _, _ = load_beir_dataset(dataset_name, split="test")

    # 2. Prepare Title -> Abstract training pairs
    train_examples = []
    for doc_id, doc in list(corpus.items())[:max_train_samples]:
        title = doc.get("title", "").strip()
        abstract = doc.get("text", "").strip()
        if title and abstract and len(title) > 10 and len(abstract) > 30:
            train_examples.append(InputExample(texts=[title, abstract]))

    print(f"Loaded {len(train_examples)} valid scientific (Title, Abstract) pairs for fine-tuning.")

    # 3. Initialize model and data loader
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading base model '{base_model_name}' on device: {device}...")
    model = SentenceTransformer(base_model_name, device=device)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Train
    print(f"Fine-tuning for {epochs} epoch(s) with MultipleNegativesRankingLoss...")
    os.makedirs(output_dir, exist_ok=True)
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=10,
        show_progress_bar=True,
        output_path=output_dir
    )

    print("\n" + "=" * 80)
    print(f"✅ Domain adaptation complete! Adapted model checkpoint saved to: {output_dir}")
    print("=" * 80)
    return output_dir


if __name__ == "__main__":
    finetune_on_scientific_corpus()
