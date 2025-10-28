from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import AgentsService

def create_agents_router(db: Database) -> Blueprint:
    bp = Blueprint("agents", __name__)
    service = AgentsService(db)

    @bp.get("/")
    @jwt_required()
    def find_agents():
        docs = service.find_all()
        if not docs:
            return json_error("Not found", 404)
        return jsonify(docs), 200

    return bp
