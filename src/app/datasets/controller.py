from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import DatasetsService

def create_datasets_router(db: Database) -> Blueprint:
    bp = Blueprint("datasets", __name__)
    service = DatasetsService(db)

    @bp.get("")
    @jwt_required()
    def find_datasets():
        docs = service.find_all()
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200

    @bp.delete("/<id>")
    @jwt_required()
    def delete_dataset(id: str):
        try:
            service.delete(id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify({"message": "Dataset deleted"}), 200

    @bp.get("/<id>/examples")
    @jwt_required()
    def find_dataset_examples(id: str):
        size = request.args.get("size", type=int)
        try:
            docs = service.find_examples(id, size)
        except ValueError:
            return json_error("Not found", 404)
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200
    
    @bp.post("/train/<id>")
    @jwt_required()
    def train_dataset(id: str):
        try:
            result = service.train_dataset(id, get_jwt_identity())
        except ValueError as err:
            return json_error(str(err), 404)
        return jsonify(result), 200
    
    @bp.get("/status/<status>")
    @jwt_required()
    def find_datasets_by_status(status: str):
        try:
            docs = service.find_by_status(status)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(docs), 200

    return bp
