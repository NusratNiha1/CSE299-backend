import os
import sys

from app import create_app
from app.config import Config

# Optional ngrok
try:
    import google.colab  # type: ignore
    COLAB = True
except Exception:
    COLAB = False

# Start app
app = create_app(Config)

# Ngrok integration: auto-start on Colab or when USE_NGROK true
use_ngrok = app.config['USE_NGROK'] or COLAB
if use_ngrok and app.config['NGROK_AUTHTOKEN']:
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(app.config['NGROK_AUTHTOKEN'])
        tunnel = ngrok.connect(addr=app.config['PORT'], proto='http')
        public_url = tunnel.public_url
        print(f"[ngrok] Tunnel active: {public_url} -> http://127.0.0.1:{app.config['PORT']}")
        os.environ['PUBLIC_URL'] = public_url
    except Exception as e:
        print(f"[ngrok] Failed to start tunnel: {e}")

if __name__ == '__main__':
    app.run(host=app.config['HOST'], port=app.config['PORT'], debug=app.config['DEBUG'], use_reloader=False)
