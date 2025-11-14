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

    @bp.post("/ai")
    @jwt_required()
    def create_model_via_ai():
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        model = service.create_model_via_ai(payload)
        return jsonify(model), 201
    
    @bp.get("/status/<status>")
    @jwt_required()
    def find_models_by_status(status: str):
        try:
            docs = service.find_by_status(status)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(docs), 200

    @bp.get("/<id>")
    @jwt_required()
    def find_model_by_id(id: str):
        try:
            doc = service.get_document(id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(doc), 200

    @bp.put("/<id>")
    @jwt_required()
    def update_model(id: str):
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        try:
            updated = service.update(id=id, update_data=payload, user_id=get_jwt_identity())
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(updated), 200
    
    @bp.delete("/<id>")
    @jwt_required()
    def delete_model(id: str):
        try:
            service.delete(id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify({"message": "Model deleted"}), 200
    
    @bp.post("/build/<id>")
    @jwt_required()
    def build_model(id: str):
        parameters = request.get_json(silent=True) or {}
        try:
            result = service.build_model(id, parameters, user_id=get_jwt_identity())
        except ValueError as err:
            return json_error(str(err), 404)
        return jsonify(result), 200

    return bp
