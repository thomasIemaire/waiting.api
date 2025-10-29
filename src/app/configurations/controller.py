from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import ConfigurationsService

def create_configurations_router(db: Database) -> Blueprint:
    bp = Blueprint("models/configurations", __name__)
    service = ConfigurationsService(db)

    @bp.get("/")
    @jwt_required()
    def find_configurations():
        docs = service.find_all()
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200

    @bp.post("/")
    @jwt_required()
    def create_configuration():
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        configuration = service.create(payload, user_id=get_jwt_identity())
        return jsonify(configuration), 201
    
    @bp.get("/<id>")
    @jwt_required()
    def get_configuration(id):
        try:
            doc = service.get_configuration(config_id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(doc), 200

    return bp
