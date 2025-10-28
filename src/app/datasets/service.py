from src.app.users.service import UsersService
from src.helpers.base_service import BaseService
from .dao import DatasetsDao
from bson import ObjectId
from pymongo.database import Database
import random

class DatasetsService(BaseService):
    
    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = DatasetsDao(db)
        self.user_service = UsersService(db)

    def find_all(self):
        datasets = self.dao.find({"status": {"$ne": "completed"}}, projection={"parameters": 0, "last_log": 0})
        return self.dao.serialize(datasets)
    
    def find_examples(self, dataset_id: str, size: int = 10):
        dataset = self.get_document(id=dataset_id)
        model = self.dao.db["models"].find_one({"_id": ObjectId(dataset.get("model"))})

        dataset_data = list(self.dao.db["datasets_data"].find({"dataset": ObjectId(dataset_id)}))
    
        ddselected = random.choices(dataset_data, k=size)
        entities = model.get("entities", []) if model else []

        examples = []
        for d in ddselected:
            data = d.get("data", {})
            text = data.get("text", "")

            example_entities = []
            for s, e, k in data.get("entities", []):
                key = entities[k]
                example_entities.append({
                    "start": s,
                    "end": e,
                    "key": key
                })
            
            examples.append({
                "text": text,
                "entities": example_entities
            })
        
        return examples

    def update_status(self, dataset_id: str, status: str):
        return self.dao.update_one(
            {"_id": ObjectId(dataset_id)},
            {"status": status}
        )

    def train_dataset(self, dataset_id: str, user_id: str, parameters: dict):
        user = self.user_service.get_document(id=user_id, projection={
            "_id": 1,
            "firstname": 1,
            "lastname": 1,
            "email": 1
        })

        self.dao.update_one(
            {"_id": ObjectId(dataset_id)},
            {"parameters": parameters, "trained_by": user, "status": "ready_to_train"}
        )

        return {"message": "Datatset is ready to be trained."}