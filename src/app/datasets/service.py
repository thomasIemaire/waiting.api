import random

from bson import ObjectId
from pymongo.database import Database

from src.app.users.service import UsersService
from src.helpers.base_service import BaseService

from .dao import DatasetsDao

class DatasetsService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = DatasetsDao(db)
        self.user_service = UsersService(db)

    # -- Queries ---------------------------------------------------------
    def find_all(self):
        return self.dao.find(
            {"status": {"$ne": "completed"}},
            projection={"parameters": 0, "last_log": 0, "model_snapshot": 0},
        )
    
    def find_examples(self, dataset_id: str, size: int = 10):
        dataset = self.get_document(id=dataset_id)
        model = dataset.get("model_snapshot")
        if not model:
            model_id = dataset.get("model")
            if isinstance(model_id, dict):
                model = model_id
            elif model_id:
                model = self.dao.db["models"].find_one({"_id": ObjectId(str(model_id))})

        dataset_data = list(self.dao.db["datasets_data"].find({"dataset": ObjectId(dataset_id)}))
        if not dataset_data:
            return []

        sample_size = max(int(size or 10), 1)
        ddselected = random.choices(dataset_data, k=sample_size)
        entities = model.get("entities", []) if model else []

        examples = []
        for d in ddselected:
            data = d.get("data", {})
            text = data.get("text", "")

            example_entities = []
            for s, e, k in data.get("entities", []):
                if 0 <= k < len(entities):
                    key = entities[k]
                else:
                    continue
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

    # -- Commands --------------------------------------------------------
    def create_dataset(self, payload: dict) -> dict:
        return self.dao.insert_one(payload)

    def train_dataset(self, dataset_id: str, user_id: str, parameters: dict):
        self.get_document(id=dataset_id)
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

        return {"message": "Dataset is ready to be trained."}
