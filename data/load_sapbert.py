import torch
from transformers import AutoTokenizer, AutoModel


def load_sapbert_encoder():
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    print(f"Loading SapBERT model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"SapBERT model loaded on device: {device}")

    return {"tokenizer": tokenizer, "model": model, "device": device}
