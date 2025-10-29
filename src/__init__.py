from flask import Blueprint, Flask, jsonify
from pymongo import MongoClient
from pymongo.database import Database
from typing import Type
import atexit

from config import Config as DefaultConfig
from .extensions import cors, jwt, swaggerui_bp

def create_app(config_object: Type[DefaultConfig] = DefaultConfig) -> Flask:
    app = Flask(__name__, static_folder="public", static_url_path="/public")
    app.config.from_object(config_object)

    cors.init_app(
        app,
        resources={r"/api/*": {"origins": ["https://sardine-app.sendoc.fr", "http://localhost:4200", "http://127.0.0.1:4200"]}},
        allow_headers=["Content-Type", "Authorization"],
        expose_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        supports_credentials=False,
        vary_header=True,
        always_send=True,
        max_age=86400,
    )

    jwt.init_app(app)
    _register_jwt_error_handlers(app)

    mongo_client = MongoClient(app.config["MONGO_URI"])
    db = mongo_client.get_database()
    app.mongo_client = mongo_client
    app.mongo_db = db

    atexit.register(mongo_client.close)

    _register_blueprints(app, db)
    return app

def _register_blueprints(app: Flask, db: Database) -> None:
    api_bp = Blueprint("api", __name__, url_prefix="/api")

    @api_bp.get("/")
    def root():
        return jsonify({"message": "'Gloup Gloup' I'm Sardine and this is my API !"}), 200
    
    from src.app.agents import create_agents_router
    api_bp.register_blueprint(create_agents_router(db), url_prefix="/agents")

    from src.app.auth import create_auth_router
    api_bp.register_blueprint(create_auth_router(db), url_prefix="/auth")

    from src.app.ai import create_ai_router
    api_bp.register_blueprint(create_ai_router(db), url_prefix="/ai")

    from src.app.configurations import create_configurations_router
    api_bp.register_blueprint(create_configurations_router(db), url_prefix="/models/configurations")

    from src.app.data import create_data_router
    api_bp.register_blueprint(create_data_router(db), url_prefix="/models/data")

    from src.app.datasets import create_datasets_router
    api_bp.register_blueprint(create_datasets_router(db), url_prefix="/datasets")

    from src.app.models import create_models_router
    api_bp.register_blueprint(create_models_router(db), url_prefix="/models")

    from src.app.users import create_users_router
    api_bp.register_blueprint(create_users_router(db), url_prefix="/users")

    from src.app.documents import create_documents_router
    api_bp.register_blueprint(create_documents_router(db), url_prefix="/documents")

    app.register_blueprint(swaggerui_bp)
    app.register_blueprint(api_bp)

def _register_jwt_error_handlers(app: Flask):
    from flask import jsonify
    from .extensions import jwt

    @jwt.unauthorized_loader
    def _unauthorized(msg):
        return jsonify({"error": "authorization_required", "message": msg}), 401

    @jwt.invalid_token_loader
    def _invalid(msg):
        return jsonify({"error": "invalid_token", "message": msg}), 422

    @jwt.expired_token_loader
    def _expired(jwt_header, jwt_payload):
        return jsonify({"error": "token_expired", "message": "Access token has expired"}), 401

    @jwt.needs_fresh_token_loader
    def _needs_fresh(jwt_header, jwt_payload):
        return jsonify({"error": "fresh_token_required"}), 401

    @jwt.revoked_token_loader
    def _revoked(jwt_header, jwt_payload):
        return jsonify({"error": "token_revoked"}), 401
