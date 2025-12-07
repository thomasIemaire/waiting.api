from __future__ import annotations

from typing import Any, Dict

from bson.objectid import ObjectId
from pymongo.database import Database

from src.helpers import utils
from src.helpers.base_service import BaseService

from .dao import ConfigurationsDao


class ConfigurationsService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = ConfigurationsDao(self.db)

    # -- Queries ---------------------------------------------------------
    def find_all(self) -> list[Dict[str, Any]]:
        return self.dao.serialize(self.dao.find_all())

    def get_configuration(self, *, config_id: str) -> Dict[str, Any]:
        config = self.get_document(id=config_id)
        
        # --- NOUVEAU : Conversion des ObjectIds en strings pour le frontend ---
        if "negative_configurations" in config:
            config["negative_configurations"] = [
                str(nc_id) for nc_id in config["negative_configurations"]
            ]
        # ----------------------------------------------------------------------

        # On parcourt les attributs pour enrichir les références
        for attr in config.get("attributes", []):
            value = attr.get("value", {})
            rule = value.get("rule")
            params = value.get("parameters", {})
            
            # Si l'attribut pointe vers une Data ou une autre Config
            if rule in ["data", "configuration"] and "object_id" in params:
                ref_id = params["object_id"]
                try:
                    if rule == "data":
                        # Recherche rapide du nom dans la collection data
                        ref_doc = self.db["models_data"].find_one(
                            {"_id": ObjectId(ref_id)}, 
                            {"name": 1} # On ne récupère que le nom, pas tout le JSON
                        )
                    elif rule == "configuration":
                        # Recherche du nom dans la collection configurations
                        ref_doc = self.dao.find_one(
                            {"_id": ObjectId(ref_id)}, 
                            {"name": 1}
                        )
                    
                    if ref_doc:
                        # On injecte le nom pour le frontend
                        params["object_name"] = ref_doc.get("name")
                except Exception:
                    pass # Si l'ID est invalide ou supprimé, on ignore

        return config

    # -- Commands --------------------------------------------------------
    def create(
        self,
        data: dict,
        *,
        user_id: str | None = None,
    ) -> dict:
        doc: Dict[str, Any] = {
            "name": data.get("name"),
            "description": data.get("description", ""),
            "constants": data.get("constants", {}),
            "attributes": data.get("attributes", []),
            "formats": data.get("formats", []),
            "randomizers": data.get("randomizers", []),
            "negative_configurations": [
                ObjectId(i) for i in data.get("negative_configurations", []) if i
            ],
            "created_at": utils.get_current_time(),
            "possibilities": self.calculate_max_configuration_possibilities(data),
        }

        if user_id:
            doc["created_by"] = ObjectId(user_id)

        return self.dao.insert_one(doc)
    
    def update(self, *, config_id: str, data: dict) -> dict:
        if not self.document_exists(id=config_id):
             raise ValueError("Configuration not found")

        update_fields = {
            "name": data.get("name"),
            "description": data.get("description", ""),
            "constants": data.get("constants", {}),
            "attributes": data.get("attributes", []),
            "formats": data.get("formats", []),
            "randomizers": data.get("randomizers", []),
            "negative_configurations": [
                ObjectId(i) for i in data.get("negative_configurations", []) if i
            ],
            "possibilities": self.calculate_max_configuration_possibilities(data),
        }
        
        self.dao.update_one({"_id": ObjectId(config_id)}, update_fields)
        
        return self.get_configuration(config_id=config_id)
    
    def delete_configuration(self, *, config_id: str) -> None:
        if not self.document_exists(id=config_id):
            raise ValueError("Configuration not found")
        
        self.dao.delete_one({"_id": ObjectId(config_id)})

    # -- Helpers ---------------------------------------------------------
    def calculate_max_configuration_possibilities(self, configuration: dict) -> int:
        possibilities = 1

        for attr in configuration.get("attributes", []):
            value = attr.get("value") or {}
            attr_size = self._calculate_attribute_size(
                value.get("rule", ""),
                value.get("parameters", {}),
            )
            possibilities *= max(attr_size, 1) * max(int(attr.get("frequency", 1)), 1)

        formats = configuration.get("formats", [])
        return possibilities * max(len(formats), 1)

    def _calculate_attribute_size(self, rule: str, parameters: dict) -> int:
        match rule:
            case "randint":
                vmin = int(parameters.get("min", 0))
                vmax = int(parameters.get("max", 0))
                if vmin > vmax:
                    vmin, vmax = vmax, vmin
                # randint is inclusive on both bounds
                return (vmax - vmin) + 1
            case "data":
                data_id = parameters.get("object_id")
                if not data_id:
                    return 1
                data = self.db["models_data"].find_one({"_id": ObjectId(data_id)}) or {}
                return len(data.get("data", [])) or 1
            case "configuration":
                config_id = parameters.get("object_id")
                if not config_id:
                    return 1
                configuration = self.dao.find_one({"_id": ObjectId(config_id)})
                if not configuration:
                    return 1
                return self.calculate_max_configuration_possibilities(configuration)
            case _:
                return 1