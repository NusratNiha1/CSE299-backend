import os

# Optional hardcoded ngrok token fallback (per user request)
NGROK_TOKEN_FALLBACK = "356WEFaqem4MagvNftOEm89DX79_6xmB5VPMyCzJ2f5SSLv1u"


class Config:
    DEBUG = False
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", "5000"))

    # Model
    MODEL_PATH = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')), 'models', 'AudioTransformer.pth')

    # Ngrok
    USE_NGROK = os.environ.get("USE_NGROK", "").lower() in {"1", "true", "yes"}
    NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN") or NGROK_TOKEN_FALLBACK
