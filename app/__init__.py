from __future__ import annotations
import os
from flask import Flask, jsonify
from flask_cors import CORS

# Keras compat for TF 2.15/2.16+ parity
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object("app.config.Config")

    # CORS for Next.js frontend (adjust origins as needed)
    CORS(
        app,
        resources={r"/*": {"origins": app.config["CORS_ALLOW_ORIGINS"]}},
        supports_credentials=True,
    )

    # Register blueprints
    from app.routes.predict import bp as predict_bp
    app.register_blueprint(predict_bp, url_prefix="/api")
    from app.routes.location_recommender import bp as locrec_bp
    app.register_blueprint(locrec_bp, url_prefix="/api")
    from app.routes.score import bp as score_bp
    app.register_blueprint(score_bp, url_prefix="/api")
    from app.routes.investor_match import bp as investor_match_bp
    app.register_blueprint(investor_match_bp, url_prefix="/api")

    # Health + meta
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.errorhandler(422)
    @app.errorhandler(400)
    def handle_400(err):
        msg = getattr(err, "description", str(err))
        return jsonify({"error": "bad_request", "message": msg}), 400

    @app.errorhandler(404)
    def handle_404(err):
        return jsonify({"error": "not_found"}), 404

    @app.errorhandler(500)
    def handle_500(err):
        return jsonify({"error": "server_error", "message": str(err)}), 500

    return app