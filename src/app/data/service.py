from bson.objectid import ObjectId
from pymongo.database import Database

from src.helpers import utils
from src.helpers.base_service import BaseService

from .dao import DataDao


class DataService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = DataDao(self.db)

    # -- Queries ---------------------------------------------------------
    def find_all(self) -> list[dict]:
        return self.dao.find()

    def get_data(self, *, data_id: str) -> dict:
        return self.get_document(id=data_id)

    # -- Commands --------------------------------------------------------
    def create(
        self,
        data: dict,
        *,
        user_id: str | None = None,
    ) -> dict:
        doc = {
            "name": data.get("name"),
            "data": data.get("data"),
            "created_at": utils.get_current_time(),
        }

        if user_id:
            doc["created_by"] = ObjectId(user_id)

        return self.dao.insert_one(doc)
    
    def update(self, *, data_id: str, data: dict) -> dict:
        if not self.document_exists(id=data_id):
            raise ValueError("Document not found")

        update_fields = {
            "name": data.get("name"),
            "data": data.get("data"), # Le tableau JSON
            "updated_at": utils.get_current_time(),
        }

        self.dao.update_one({"_id": ObjectId(data_id)}, update_fields)
        return self.get_data(data_id=data_id)

    def delete_data(self, *, data_id: str) -> None:
        if not self.document_exists(id=data_id):
            raise ValueError("Document not found")
        
        self.dao.delete_one({"_id": ObjectId(data_id)})