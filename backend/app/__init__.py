import os
import sys
from flask import Flask

# Ensure project root is on sys.path so we can import from src.*
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .config import Config
from .services.model_service import load_model


def create_app(config_class: type = Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Load model once and attach to app config
    device, model = load_model(app.config['MODEL_PATH'])
    app.config['DEVICE'] = device
    app.config['MODEL'] = model

    # Register blueprints
    from .routes.health import bp as health_bp
    from .routes.predict import bp as predict_bp
    app.register_blueprint(health_bp, url_prefix='/')
    app.register_blueprint(predict_bp, url_prefix='/')

    return app
