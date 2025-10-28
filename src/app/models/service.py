from src.app.data.service import DataService
from src.app.datasets.service import DatasetsService
from src.app.users.service import UsersService
from src.app.models.dao import ModelsDao
from pymongo.database import Database
from bson.objectid import ObjectId
from src.app.configurations.service import ConfigurationsService
from src.helpers.base_service import BaseService
from src.helpers import utils

class ModelsService(BaseService):
    
    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = ModelsDao(db)

        self.configurations_service = ConfigurationsService(db)
        self.data_service = DataService(db)
        self.datasets_service = DatasetsService(db)
        self.user_service = UsersService(db)

    def find_all(self):
        models = self.dao.find_all()
        return self.dao.serialize(models)

    def create(self, user_id: str, model_data: dict) -> ObjectId:
        model_name = model_data.get("name", None)
        if not model_name:
            raise ValueError("Le nom du modèle est requis")

        model_reference = model_data.get("reference", None)
        if not model_reference:
            raise ValueError("La référence du modèle est requise")
        
        model_version = self.document_exists(query={
            "reference": model_reference
        })
        if model_version:
            raise ValueError("La référence du modèle existe déjà")

        user = self.user_service.get_document(id=user_id, projection={
            "_id": 1,
            "firstname": 1,
            "lastname": 1,
            "email": 1
        })

        default_version = "1.0"
        doc = {
            "name": model_name,
            "description": model_data.get("description", ""),
            "reference": model_reference,
            "version": default_version,
            "configuration": ObjectId(model_data.get("configuration", None)),
            "mapper": model_data.get("mapper", {}),
            "created_by": user,
            "created_at": utils.get_current_time(),
            "updated_at": utils.get_current_time()
        }

        self.dao.insert_one(doc)

        return self.dao.serialize(doc)

    def build_model(self, model_id: str, size: str, *, user_id: str = None) -> dict:
        model = self.get_document(id=model_id, projection={
            "_id": 1,
            "name": 1,
            "version": 1,
            "reference": 1,
            "description": 1,
            "configuration": 1,
        })
        user = self.user_service.get_document(id=user_id, projection={
            "_id": 1,
            "firstname": 1,
            "lastname": 1,
            "email": 1
        })

        mcid = model.get("configuration", None)
        if not mcid:
            raise ValueError("Model configuration is missing")

        docdt = {
            "model": model,
            "size": size,
            "status": "ready_to_generate",
            "created_by": user,
            "created_at": utils.get_current_time(),
        }

        self.datasets_service.dao.insert_one(docdt)

        return self.datasets_service.dao.serialize(docdt)

