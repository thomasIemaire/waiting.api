from flask import json
from src.app.data.service import DataService
from src.app.datasets.service import DatasetsService
from src.app.users.service import UsersService
from src.app.models.dao import ModelsDao
from pymongo.database import Database
from bson.objectid import ObjectId
from src.app.configurations.service import ConfigurationsService
from src.helpers.base_service import BaseService
from src.helpers import utils

import os
from dotenv import load_dotenv
from openai import OpenAI

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

        configuration_id = model_data.get("configuration", None)
        config_ref = ObjectId(str(configuration_id)) if configuration_id else None

        user = self.user_service.find_user_by_id_basic(user_id)

        default_status = "ready"
        default_version = "1.0"
        doc = {
            "name": model_name,
            "description": model_data.get("description", ""),
            "reference": model_reference,
            "version": default_version,
            "configuration": config_ref,
            "mapper": model_data.get("mapper", {}),
            "status": default_status,
            "created_by": user,
            "created_at": utils.get_current_time(),
            "updated_at": utils.get_current_time(),
        }

        created = self.dao.insert_one(doc)
        return created

    def update(self, *, id: str, update_data: dict, user_id: str) -> dict:
        model = self.get_document(id=id)
        update_fields = {}

        if "name" in update_data:
            update_fields["name"] = update_data["name"].strip()
        if "description" in update_data:
            update_fields["description"] = update_data["description"].strip()
        if "mapper" in update_data:
            update_fields["mapper"] = update_data["mapper"]
        
        if "configuration" in update_data:
            config_id = update_data["configuration"]
            update_fields["configuration"] = ObjectId(str(config_id)) if config_id else None

        if "reference" in update_data:
            new_reference = update_data["reference"].strip()
            if self.document_exists(query={"reference": new_reference, "_id": {"$ne": ObjectId(id)}}):
                raise ValueError("La référence du modèle existe déjà")
            update_fields["reference"] = new_reference

        if model.get("reference") != update_fields.get("reference", model.get("reference")) or \
            model.get("mapper") != update_fields.get("mapper", model.get("mapper")):
            update_fields["version"] = utils.increment_version(model.get("version", "1.0"), "major")

        update_fields["updated_at"] = utils.get_current_time()

        updated = self.dao.update_one(
            {"_id": ObjectId(id)},
            update_fields,
        )
        return updated

    def create_model_via_ai(self, payload: dict) -> dict:
        generate_model = payload.get("generate_model", False)
        generate_configuration = payload.get("generate_configuration", False)
        if not generate_model and not generate_configuration:
            raise ValueError("At least one of 'generate_model' or 'generate_configuration' must be true")
        
        user_prompt = payload.get("prompt", "")
        if not user_prompt:
            raise ValueError("Prompt is required to generate model via AI")

        system_prompt = ""

        model = payload.get("model", {})
        if generate_model:
            with open(os.path.join("src", "static", "openai", "model_prompt.txt"), "r", encoding="utf-8") as f:
                system_prompt = f.read()

            model_content = self.call_openai_api(system_prompt, user_prompt)
            model = json.loads(model_content)

            model_mapper = model.get("mapper", {})
            model["mapper"] = self.set_leaf_mapper(model.get("reference", ""), model_mapper)

        configuration = payload.get("configuration", {})
        if generate_configuration and model:
            with open(os.path.join("src", "static", "openai", "configuration_prompt.txt"), "r", encoding="utf-8") as f:
                system_prompt = f.read()

            existing_configuration = self.configurations_service.dao.find(query={}, projection={"_id": 1, "name": 1, "description": 1})

            user_prompt += f"\nGénérer la configuration pour le modèle suivant:\n{json.dumps(model, indent=2)}"
            user_prompt += f"\nNe pas oublier de bien respecter les références du mapper:\n{json.dumps(model.get('mapper', {}), indent=2)}"
            user_prompt += f"\nVoici les configurations existantes dans le système:\n{json.dumps(self.configurations_service.dao.serialize(existing_configuration), indent=2)}\Tu peux réutiliser des configurations existantes si nécessaire avec leur `_id`."

            configuration_content = self.call_openai_api(system_prompt, user_prompt)
            configuration = json.loads(configuration_content)

        return { "model": model, "configuration": configuration}
    
    def call_openai_api(self, system_prompt: str, user_prompt: str) -> str:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is not configured")
        
        client = OpenAI(api_key=openai_api_key)

        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return completion.choices[0].message.content

    def set_leaf_mapper(self, reference: str, mapper: dict) -> dict:
        for key, value in mapper.items():
            if isinstance(value, dict):
                reference += f"_{key}"
                mapper[key] = self.set_leaf_mapper(reference.upper(), value)
            else:
                mapper[key] = f"{reference}_{key}".upper()

        return mapper

    def delete(self, *, id: str) -> None:
        if not self.document_exists(id=id):
            raise ValueError("Document not found")
        
        self.dao.delete_one({"_id": ObjectId(id)})

    def find_by_status(self, status: str):
        models = self.dao.find(query={"status": status}, projection={"mapper": 0})
        return self.dao.serialize(models)

    def update_status(self, model_id: str, status: str, *, user_id: str | None = None) -> dict:
        return self.dao.update_one(
            {"_id": ObjectId(model_id)},
            {
                "status": status,
                "build_at": utils.get_current_time(),
                "build_by": self.user_service.find_user_by_id_basic(user_id)
            },
        )

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

        user = self.user_service.find_user_by_id_basic(user_id)

        configuration_id = model.get("configuration")
        if not configuration_id:
            raise ValueError("Model configuration is missing")

        model["model_id"] = model.pop("_id")
        parameters = parameters or {}

        self.dao.update_one(
            {"_id": ObjectId(model_id)},
            {"version": utils.increment_version(model.get("version", "1.0"), "minor")}
        )

        dataset_payload = {
            **model,
            "status": "to-build",
            "created_by": user,
            "created_at": utils.get_current_time(),
            "parameters": parameters,
        }

        return self.datasets_service.create(dataset_payload)

