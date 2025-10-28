from bson import ObjectId

from src.helpers.base_dao import BaseDao

class DocumentsDao(BaseDao):
    collection_name = "documents"