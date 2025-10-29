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
        model_name = (model_data.get("name") or "").strip()
        if not model_name:
            raise ValueError("Le nom du modèle est requis")

        model_reference = (model_data.get("reference") or "").strip()
        if not model_reference:
            raise ValueError("La référence du modèle est requise")

        if self.document_exists(query={"reference": model_reference}):
            raise ValueError("La référence du modèle existe déjà")

        configuration_id = model_data.get("configuration")
        if not configuration_id:
            raise ValueError("La configuration du modèle est requise")

        user = self.user_service.get_document(
            id=user_id,
            projection={
                "_id": 1,
                "firstname": 1,
                "lastname": 1,
                "email": 1,
            },
        )

        default_version = "1.0"
        doc = {
            "name": model_name,
            "description": model_data.get("description", ""),
            "reference": model_reference,
            "version": default_version,
            "configuration": ObjectId(str(configuration_id)),
            "mapper": model_data.get("mapper", {}),
            "created_by": user,
            "created_at": utils.get_current_time(),
            "updated_at": utils.get_current_time(),
        }

        created = self.dao.insert_one(doc)

        return created

    def build_model(self, model_id: str, parameters: dict | None, *, user_id: str | None = None) -> dict:
        model = self.get_document(
            id=model_id,
            projection={
                "_id": 1,
                "name": 1,
                "version": 1,
                "reference": 1,
                "description": 1,
                "configuration": 1,
            },
        )

        if not user_id:
            raise ValueError("User identifier is required to build a model")

        user = self.user_service.get_document(
            id=user_id,
            projection={
                "_id": 1,
                "firstname": 1,
                "lastname": 1,
                "email": 1,
            },
        )

        configuration_id = model.get("configuration")
        if not configuration_id:
            raise ValueError("Model configuration is missing")

        parameters = parameters or {}
        model_id = ObjectId(str(model["_id"]))

        dataset_payload = {
            "model": model_id,
            "model_snapshot": model,
            "configuration": ObjectId(str(configuration_id)),
            "status": "ready_to_generate",
            "created_by": user,
            "created_at": utils.get_current_time(),
            "parameters": parameters,
        }

        size = parameters.get("size")
        if size is not None:
            try:
                dataset_payload["size"] = int(size)
            except (TypeError, ValueError):
                dataset_payload["size"] = size

        created = self.datasets_service.create_dataset(dataset_payload)
        return created

