from src.helpers.base_dao import BaseDao


class ConfigurationsDao(BaseDao):
    collection_name = "models_configurations"

    def find_all(self, *, sort: str = "created_at") -> list[dict]:
        return self.find(
            sort=[(sort, -1)],
            # Exclure le nouveau champ de la liste globale pour alléger
            projection={
                "attributes": 0, 
                "formats": 0, 
                "randomizers": 0,
                "negative_configurations": 0
            },
        )