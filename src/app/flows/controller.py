from flask import Blueprint, jsonify, request
from pymongo.database import Database
from flask_jwt_extended import get_jwt_identity, jwt_required

from src.helpers.utils import json_error
from .service import FlowService

def create_flows_router(db: Database) -> Blueprint:
    bp = Blueprint("flows", __name__)
    service = FlowService(db)

    @bp.get("/")
    @jwt_required()
    def find_flows():
        """Retourne la liste formatée avec le mock user"""
        print( "---- FLOW CONTROLLER: GET ALL FLOWS ----" )
        docs = service.find_all()
        # Pas de 404 ici si liste vide, on retourne juste un tableau vide [], c'est plus standard
        return jsonify(docs), 200

    @bp.post("/")
    @jwt_required()
    def create_flow():
        print( "---- FLOW CONTROLLER: CREATE FLOW ----" )
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        
        # On passe le payload complet (name, description, data)
        created = service.create(payload, user_id=get_jwt_identity())
        return jsonify(created), 201

    @bp.put("/<id>")
    @jwt_required()
    def update_flow(id):
        print( "---- FLOW CONTROLLER: UPDATE FLOW ----" )
        payload = request.get_json(silent=True)
        if not payload:
            return json_error("Bad request")
        try:
            updated = service.update(flow_id=id, payload=payload)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(updated), 200
    
    @bp.post("/<id>/default")
    @jwt_required()
    def set_default_flow(id):
        print( "---- FLOW CONTROLLER: SET DEFAULT FLOW ----" )
        try:
            service.set_default(flow_id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify({"status": "updated"}), 200

    @bp.get("/<id>")
    @jwt_required()
    def get_flow(id):
        print( "---- FLOW CONTROLLER: GET FLOW ----" )
        try:
            doc = service.get_flow(flow_id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify(doc), 200
    
    @bp.delete("/<id>")
    @jwt_required()
    def delete_flow(id):
        print( "---- FLOW CONTROLLER: DELETE FLOW ----" )
        try:
            service.delete_flow(flow_id=id)
        except ValueError:
            return json_error("Not found", 404)
        return jsonify({"status": "deleted"}), 200

    return bp