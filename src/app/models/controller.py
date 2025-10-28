from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import ModelsService

def create_models_router(db: Database) -> Blueprint:
    bp = Blueprint("models", __name__)
    service = ModelsService(db)

    @bp.get("/")
    @jwt_required()
    def find_models():
        docs = service.find_all()
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200
    
    @bp.post("/")
    @jwt_required()
    def create_model():
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        model = service.create(get_jwt_identity(), payload)
        return jsonify(model), 201
    
    @bp.post("/build/<model_id>")
    @jwt_required()
    def build_model(model_id):
        try:
            parameters = request.get_json(silent=True) or {}
            model = service.build_model(model_id, parameters, user_id=get_jwt_identity())
            return jsonify(model), 200
        except ValueError as e:
            return json_error(str(e))

    return bp
