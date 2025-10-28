from bson import ObjectId

from src.helpers.base_dao import BaseDao

class UsersDao(BaseDao):
    collection_name = "users"