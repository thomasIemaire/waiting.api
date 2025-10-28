from dataclasses import dataclass
from bson import ObjectId
from pymongo.database import Database

from .base_dao import BaseDao

@dataclass(slots=True)
class BaseService:
    """Simple container for the Mongo database instance."""

    db: Database
    dao: BaseDao = None

    def query_or_id(self, *, query: dict = {}, id: str = None) -> dict:
        return query if not id else {"_id": ObjectId(id)}

    def get_document(self, *, query: dict = {}, id: str = None, projection = {}) -> bool:
        document = self.dao.find_one(self.query_or_id(query=query, id=id), projection=projection)

        if not document:
            raise ValueError("Document not found")
        
        return document

    def document_exists(self, *, query: dict = {}, id: str = None, projection = {}) -> bool:
        return self.dao.find_one(self.query_or_id(query=query, id=id), projection=projection) is not None