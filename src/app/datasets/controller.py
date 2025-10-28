from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import DatasetsService

def create_datasets_router(db: Database) -> Blueprint:
    bp = Blueprint("datasets", __name__)
    service = DatasetsService(db)

    @bp.get("/")
    @jwt_required()
    def find_datasets():
        docs = service.find_all()
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200
    
    @bp.get("/<id>/examples")
    @jwt_required()
    def find_dataset_examples(id: str):
        size = request.args.get("size", type=int)
        docs = service.find_examples(id, size)
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200
    
    @bp.post("/train/<id>")
    @jwt_required()
    def train_dataset(id: str):
        parameters = request.get_json(silent=True) or {}
        result = service.train_dataset(id, get_jwt_identity(), parameters)
        if not result:
            return json_error("Not found", 404)
        return jsonify(result), 200

    return bp
