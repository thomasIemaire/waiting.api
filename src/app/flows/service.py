from bson.objectid import ObjectId
from pymongo.database import Database

from src.helpers import utils
from src.helpers.base_service import BaseService

from .dao import FlowDao


class FlowService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = FlowDao(self.db)

    # -- Queries ---------------------------------------------------------
    def find_all(self) -> list[dict]:
        """
        Récupère tous les flux et formate la sortie selon le besoin spécifique
        (avec mock de l'utilisateur).
        """
        docs = self.dao.find()
        results = []

        for doc in docs:
            # Transformation pour correspondre au format demandé
            results.append({
                "id": str(doc["_id"]),
                "name": doc.get("name"),
                "description": doc.get("description", ""),
                "created_at": doc.get("created_at"),
                # MOCK: Données utilisateur en dur comme demandé
                "created_by": {
                    "id": str(doc.get("created_by", "6900ca440de85ad6173e53f7")), 
                    "firstname": "Thomas",
                    "lastname": "Lemaire"
                }
            })
        
        return results

    def get_flow(self, *, flow_id: str) -> dict:
        return self.get_document(id=flow_id)

    # -- Commands --------------------------------------------------------
    def create(
        self,
        payload: dict,
        *,
        user_id: str | None = None,
    ) -> dict:
        doc = {
            "name": payload.get("name"),
            "description": payload.get("description", ""),
            "data": payload.get("data", {}), # Le JSON du graph (nodes/links)
            "created_at": utils.get_current_time(),
        }

        if user_id:
            doc["created_by"] = ObjectId(user_id)

        return self.dao.insert_one(doc)
    
    def update(self, *, flow_id: str, payload: dict) -> dict:
        if not self.document_exists(id=flow_id):
            raise ValueError("Document not found")

        update_fields = {
            "name": payload.get("name"),
            "description": payload.get("description"),
            "data": payload.get("data"),
            "updated_at": utils.get_current_time(),
        }
        
        # Nettoyage des clés None si on ne veut pas écraser avec du vide (optionnel)
        # update_fields = {k: v for k, v in update_fields.items() if v is not None}

        self.dao.update_one({"_id": ObjectId(flow_id)}, update_fields)
        return self.get_flow(flow_id=flow_id)

    def delete_flow(self, *, flow_id: str) -> None:
        if not self.document_exists(id=flow_id):
            raise ValueError("Document not found")
        
        self.dao.delete_one({"_id": ObjectId(flow_id)})