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
        return self.dao.find_all()

    def get_configuration(self, *, config_id: str) -> Dict[str, Any]:
        return self.get_document(id=config_id)

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
            "attributes": data.get("attributes", []),
            "formats": data.get("formats", []),
            "randomizers": data.get("randomizers", []),
            "created_at": utils.get_current_time(),
            "possibilities": self.calculate_max_configuration_possibilities(data),
        }

        if user_id:
            doc["created_by"] = ObjectId(user_id)

        return self.dao.insert_one(doc)

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

