from .dao import DataDao
from pymongo.database import Database
from bson.objectid import ObjectId
from src.helpers.base_service import BaseService
from src.helpers import utils

class DataService(BaseService):
    
    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = DataDao(self.db)

    def create(
        self,
        data: dict,
        *,
        user_id: str = None
    ) -> dict:
        doc = {
            "name": data.get("name"),
            "data": data.get("data"),
            "created_at": utils.get_current_time(),
        }

        if user_id: doc["created_by"] = ObjectId(user_id)

        return self.dao.insert_one(doc)
