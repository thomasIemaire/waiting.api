from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import DocumentsService

def create_documents_router(db: Database) -> Blueprint:
    bp = Blueprint("documents", __name__)
    service = DocumentsService(db)

    @bp.get("/")
    @jwt_required()
    def find_documents():
        docs = service.find_documents_by_user_id(user_id=get_jwt_identity())
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200
    
    @bp.post("/")
    @jwt_required()
    def create_document():
        try:
            payload = request.get_json(silent=False) or {}
            document = service.create_document(document_data=payload, user_id=get_jwt_identity())
            return jsonify(document), 201
        except ValueError as ve:
            return json_error(str(ve), 400)
        except RuntimeError as err:
            return json_error(str(err), 500)
    
    @bp.get("/<string:document_id>")
    @jwt_required()
    def get_document(document_id: str):
        try:
            document = service.find_document(document_id=document_id, user_id=get_jwt_identity())
        except ValueError as err:
            return json_error(str(err), 404)
        except RuntimeError as err:
            return json_error(str(err), 500)
        return jsonify(document), 200

    return bp
