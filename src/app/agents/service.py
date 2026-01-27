from pymongo.database import Database

from src.app.agents.dao import AgentsDao
from src.helpers.base_service import BaseService


class AgentsService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = AgentsDao(self.db)

    def find_all(self):
        return self.dao.find(projection={"path": 0})
