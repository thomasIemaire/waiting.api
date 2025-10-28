from flask import Blueprint, jsonify
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import UsersService

def create_users_router(db: Database) -> Blueprint:
    bp = Blueprint("users", __name__)
    service = UsersService(db)

    @bp.get("/")
    @jwt_required()
    def find_users():
        return jsonify(service.find_users()), 200

    @bp.get("/me")
    @jwt_required()
    def find_me():
        user = service.find_user_by_id(get_jwt_identity())
        if not user:
            return json_error("Not found", 404)
        return jsonify(user), 200
    
    @bp.put("/me/avatar")
    @jwt_required()
    def update_my_avatar():
        service.update_avatar(get_jwt_identity())
        return jsonify({"message": "Avatar mis à jour"}), 200

    return bp
