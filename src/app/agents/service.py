from src.app.agents.dao import AgentsDao
from src.helpers.base_service import BaseService
from pymongo.database import Database

class AgentsService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = AgentsDao(self.db)

    def find_all(self):
        agents = self.dao.find(projection={"path": 0})
        return self.dao.serialize(agents)