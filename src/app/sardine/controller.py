from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import SardineService

def create_sardine_router(db: Database) -> Blueprint:
    bp = Blueprint("sardine", __name__)
    service = SardineService(db)

    # ... (route /magnetic-zone existante) ...
    @bp.post("/magnetic-zone")
    @jwt_required()
    def create_magnetic_zone():
        # ... (code existant inchangé) ...
        try:
            payload = request.get_json(silent=False) or {}
            zone = service.create_magnetic_zone(
                b64_string=payload.get("base64", ""),
                zone_data=payload.get("zone_data", {}),
                user_id=get_jwt_identity()
            )
            return jsonify(zone), 201
        except Exception as e:
            return json_error(str(e), 400)

    # === NOUVELLES ROUTES ===

    @bp.get("/tree")
    @jwt_required()
    def get_tree():
        tree = service.get_documents_tree(get_jwt_identity())
        return jsonify(tree), 200

    @bp.post("/document")
    @jwt_required()
    def save_document():
        try:
            payload = request.get_json()
            result = service.save_document(payload, get_jwt_identity())
            return jsonify(result), 201
        except Exception as e:
            return json_error(str(e), 500)

    @bp.get("/document")
    @jwt_required()
    def get_document():
        filename = request.args.get("filename")
        classification = request.args.get("classification")
        doc = service.get_document_content(filename, classification)
        if doc:
            return jsonify(doc), 200
        return json_error("Document not found", 404)

    @bp.post("/classification")
    @jwt_required()
    def create_classification():
        try:
            payload = request.get_json()
            res = service.create_classification(payload.get("name"))
            return jsonify(res), 201
        except Exception as e:
            return json_error(str(e), 400)

    return bp