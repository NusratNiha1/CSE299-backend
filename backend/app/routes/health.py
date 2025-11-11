from flask import Blueprint, jsonify, current_app

bp = Blueprint('health', __name__)


@bp.get('/health')
def health():
    device = current_app.config.get('DEVICE')
    return jsonify({"status": "ok", "device": str(device)})
