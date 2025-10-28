from pymongo.database import Database
import os, random

from src.helpers.avatar import generate_avatar, save_avatar

from src.helpers.base_service import BaseService

from src.app.users.dao import UsersDao

class UsersService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = UsersDao(self.db)

    def find_user_by_id(self, user_id: str):
        return self.get_document(id=user_id, projection={ "password": 0 })
    
    def find_users(self):
        return self.dao.find(projection={"password": 0})

    def update_avatar(self, user_id: str) -> None:
        email = self.get_document(id=user_id).get("email")
        save_avatar(
            generate_avatar(email, 800, variant=random.randint(1, 10e16)),
            os.path.join("src", "public", "avatars"),
            f"{user_id}.png"
        )