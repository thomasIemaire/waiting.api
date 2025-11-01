from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import AiService

def create_ai_router(db: Database) -> Blueprint:
    bp = Blueprint("ai", __name__)
    service = AiService(db)

    @bp.post("/document")
    @jwt_required()
    def ai_document():
        data = request.get_json() or {}
        try:
            created = service.analyze_document(data, user_id=get_jwt_identity())
        except ValueError as e:
            return json_error(str(e))
        return jsonify(created), 201

    return bp
