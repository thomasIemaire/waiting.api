from pymongo.database import Database
from bson.objectid import ObjectId
from src.helpers.base_service import BaseService
from src.helpers import utils

from .dao import ConfigurationsDao

class ConfigurationsService(BaseService):
    
    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = ConfigurationsDao(self.db)

    def create(
        self,
        data: dict,
        *,
        user_id: str = None
    ) -> dict:
        doc = {
            "name": data.get("name"),
            "description": data.get("description", ""),
            "attributes": data.get("attributes", []),
            "formats": data.get("formats", []),
            "randomizers": data.get("randomizers", []),
            "created_at": utils.get_current_time(),
            "possibilities": self.calculate_max_configuration_possibilities(data)
        }

        if user_id: doc["created_by"] = ObjectId(user_id)

        return self.dao.insert_one(doc)
    
    def calculate_max_configuration_possibilities(
            self,
            configuration: dict
        ) -> int:
        possibilities = 1

        for attr in configuration.get("attributes", []):
            vattr = attr.get("value")
            possibilities *= (self._calculate_attribute_size(
                vattr.get("rule", ""),
                vattr.get("parameters", {})
            ) * attr.get("frequency", 1))

        return possibilities * len(configuration.get("formats", []))
    
    def _calculate_attribute_size(
        self,
        rule: str,
        parameters: dict
    ):
        match rule:
            case "randint":
                vmin = int(parameters.get("min", 0))
                vmax = int(parameters.get("max", 0))
                if vmin > vmax: vmin, vmax = vmax, vmin
                return abs(vmin) + vmax
            case "data":
                data_id = parameters.get("object_id")
                data = None
                if data_id:
                    data = self.db["models_data"].find_one({"_id": ObjectId(data_id)})
                return len(data.get("data", [])) if data else 1
            case "configuration":
                config_id = parameters.get("object_id")
                if config_id:
                    configuration = self.find_one({"_id": ObjectId(config_id)})
                    size = self.calculate_max_configuration_possibilities(configuration)
                return size
            case _:
                return 1
