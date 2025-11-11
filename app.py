import os
import uuid
import json
import traceback
from typing import List, Tuple

import numpy as np
from flask import Flask, request, jsonify

import torch
import torch.nn as nn

from src.supervised import AudioTransformer
from src.audio_preprocessing import fbank_features_extraction
import librosa


MODEL_PATH = os.path.join("models", "AudioTransformer.pth")
# User-provided ngrok token fallback (for Colab). Prefer environment variable NGROK_AUTHTOKEN.
NGROK_HARDCODED_TOKEN = "356WEFaqem4MagvNftOEm89DX79_6xmB5VPMyCzJ2f5SSLv1u"


def load_model(model_path: str, device: torch.device) -> nn.Module:
    model = AudioTransformer(input_dim=41)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preds_to_segments(
    preds: np.ndarray,
    window_length: float = 0.025,
    window_step: float = 0.01,
) -> List[Tuple[float, float]]:
    segments: List[Tuple[float, float]] = []
    if preds.size == 0:
        return segments
    # Find contiguous spans of 1s
    in_seg = False
    start_idx = 0
    for i, v in enumerate(preds):
        if v == 1 and not in_seg:
            in_seg = True
            start_idx = i
        elif v == 0 and in_seg:
            in_seg = False
            end_idx = i - 1
            start_t = start_idx * window_step
            end_t = end_idx * window_step + window_length
            segments.append((round(start_t, 3), round(end_t, 3)))
    if in_seg:
        end_idx = len(preds) - 1
        start_t = start_idx * window_step
        end_t = end_idx * window_step + window_length
        segments.append((round(start_t, 3), round(end_t, 3)))
    return segments


def create_app():
    app = Flask(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH, device)

    # Ensure tmp upload directory exists
    upload_dir = os.path.join(os.getcwd(), "tmp")
    os.makedirs(upload_dir, exist_ok=True)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "device": str(device)})

    @app.post("/predict")
    def predict():
        try:
            # Accept threshold via query or form/json
            threshold = 0.5
            if "threshold" in request.args:
                threshold = float(request.args.get("threshold", "0.5"))
            else:
                try:
                    payload = request.get_json(silent=True) or {}
                    if isinstance(payload, dict) and "threshold" in payload:
                        threshold = float(payload["threshold"])
                except Exception:
                    pass

            # Accept file field name 'audio' or 'file'
            file = request.files.get("audio") or request.files.get("file")
            if not file:
                return (
                    jsonify({"error": "No audio file found in form-data under 'audio' or 'file'"}),
                    400,
                )

            # Persist to a temp file with original extension if possible
            ext = os.path.splitext(file.filename or "uploaded.wav")[1] or ".wav"
            tmp_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
            file.save(tmp_path)

            # Compute max length for this single file (so we don't pad)
            signal, _ = librosa.load(tmp_path)
            max_len = len(signal)

            # Extract fbank features (shape: [1, frames, 41]) with same params as training
            feats = fbank_features_extraction([tmp_path], max_length=max_len)
            # Clean up temp file
            try:
                os.remove(tmp_path)
            except Exception:
                pass

            # Run inference
            x = torch.tensor(feats, dtype=torch.float32, device=device)  # [1, T, 41]
            with torch.no_grad():
                logits = model(x)  # [1, T, 1]
                logits = logits.view(-1)  # [T]
                probs = torch.sigmoid(logits).cpu().numpy()  # [T]
            preds = (probs > threshold).astype(int)

            # Build response
            segments = preds_to_segments(preds)
            response = {
                "model": "AudioTransformer",
                "threshold": threshold,
                "num_frames": int(preds.shape[0]),
                "cry_ratio": float(np.mean(preds)) if preds.size > 0 else 0.0,
                "any_cry": bool(preds.any()),
                "segments": [
                    {"start": s, "end": e, "duration": round(e - s, 3)} for s, e in segments
                ],
                # For large files, these arrays can be big; include for completeness
                "frame_times_sec": [round(i * 0.01, 3) for i in range(preds.shape[0])],
                "frame_probabilities": [float(p) for p in probs.tolist()],
                "frame_predictions": preds.tolist(),
            }
            return jsonify(response)

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()

    # Optional: start an ngrok tunnel (useful on Google Colab)
    # Enable by setting USE_NGROK=true and add your token to NGROK_AUTHTOKEN
    use_ngrok = os.getenv("USE_NGROK", "").lower() in {"1", "true", "yes"}
    # Auto-enable on Colab if token present
    try:
        import google.colab  # type: ignore
        colab_env = True
    except Exception:
        colab_env = False

    token = os.getenv("NGROK_AUTHTOKEN") or NGROK_HARDCODED_TOKEN
    if (use_ngrok or colab_env) and token:
        try:
            from pyngrok import ngrok
            # Set auth token and open tunnel
            ngrok.set_auth_token(token)
            tunnel = ngrok.connect(addr=5000, proto="http")
            public_url = tunnel.public_url
            print(f"[ngrok] Tunnel active: {public_url} -> http://127.0.0.1:5000")
            # Optionally expose as env var
            os.environ["PUBLIC_URL"] = public_url
        except Exception as e:
            print(f"[ngrok] Failed to start tunnel: {e}")

    # Default Flask dev server; in production, use gunicorn or waitress
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
