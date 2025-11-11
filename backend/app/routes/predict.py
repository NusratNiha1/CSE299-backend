import os
import numpy as np
from flask import Blueprint, request, jsonify, current_app

from ..services.audio_service import save_temp_file, compute_features
from ..services.model_service import infer
from ..utils.segments import preds_to_segments

bp = Blueprint('predict', __name__)


@bp.post('/predict')
def predict():
    try:
        threshold = 0.5
        if 'threshold' in request.args:
            threshold = float(request.args.get('threshold', '0.5'))
        else:
            payload = request.get_json(silent=True) or {}
            if isinstance(payload, dict) and 'threshold' in payload:
                threshold = float(payload['threshold'])

        file = request.files.get('audio') or request.files.get('file')
        if not file:
            return jsonify({"error": "No audio file found in form-data under 'audio' or 'file'"}), 400

        upload_dir = os.path.join(os.getcwd(), 'tmp')
        tmp_path = save_temp_file(file, upload_dir)
        try:
            feats = compute_features(tmp_path)  # [1, T, 41]
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        device = current_app.config['DEVICE']
        model = current_app.config['MODEL']
        probs = infer(model, device, feats)
        preds = (probs > threshold).astype(int)

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
            "frame_times_sec": [round(i * 0.01, 3) for i in range(preds.shape[0])],
            "frame_probabilities": [float(p) for p in probs.tolist()],
            "frame_predictions": preds.tolist(),
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
