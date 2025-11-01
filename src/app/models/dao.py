from src.helpers.base_dao import BaseDao


class ModelsDao(BaseDao):
    collection_name = "models"

    def find_all(self, *, sort: str = "updated_at") -> list[dict]:
        return self.find(
            sort=[(sort, -1)],
            projection={"mapper": 0, "configuration": 0, "entities": 0, "labels": 0, "randomizers": 0},
        )
