import os
import torch
import torch.nn as nn

from src.supervised import AudioTransformer


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str):
    device = get_device()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = AudioTransformer(input_dim=41)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return device, model


def infer(model: nn.Module, device, feats):
    import numpy as np
    import torch

    x = torch.tensor(feats, dtype=torch.float32, device=device)  # [1, T, 41]
    with torch.no_grad():
        logits = model(x)  # [1, T, 1]
        logits = logits.view(-1)  # [T]
        probs = torch.sigmoid(logits).cpu().numpy()  # [T]
    return probs
