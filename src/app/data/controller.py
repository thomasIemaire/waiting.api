from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import DataService

def create_data_router(db: Database) -> Blueprint:
    bp = Blueprint("models/data", __name__)
    service = DataService(db)

    @bp.get("/")
    @jwt_required()
    def find_data():
        docs = service.find_all()
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200

    @bp.post("/")
    @jwt_required()
    def create_data():
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        created = service.create(payload, user_id=get_jwt_identity())
        return jsonify(created), 201

    @bp.put("/<id>")
    @jwt_required()
    def update_data(id):
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        try:
            updated = service.update(data_id=id, data=payload)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(updated), 200

    @bp.get("/<id>")
    @jwt_required()
    def get_data(id):
        try:
            doc = service.get_data(data_id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(doc), 200
    
    @bp.delete("/<id>")
    @jwt_required()
    def delete_data(id):
        try:
            service.delete_data(data_id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify({"status": "deleted"}), 200

    return bp
