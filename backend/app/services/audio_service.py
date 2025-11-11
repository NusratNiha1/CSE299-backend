import os
import uuid
import librosa
from src.audio_preprocessing import fbank_features_extraction


def save_temp_file(uploaded_storage, upload_dir: str) -> str:
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(getattr(uploaded_storage, 'filename', 'uploaded.wav'))[1] or '.wav'
    tmp_path = os.path.join(upload_dir, f"{uuid.uuid4().hex}{ext}")
    uploaded_storage.save(tmp_path)
    return tmp_path


def compute_features(file_path: str):
    signal, _ = librosa.load(file_path)
    max_len = len(signal)
    feats = fbank_features_extraction([file_path], max_length=max_len)
    return feats
