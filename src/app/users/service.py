import os
import random
import shutil

from pymongo.database import Database

from src.app.users.dao import UsersDao
from src.helpers.avatar import generate_avatar, save_avatar
from src.helpers.base_service import BaseService


class UsersService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = UsersDao(self.db)

    def find_user_by_id(self, user_id: str):
        return self.get_document(id=user_id, projection={"password": 0})

    def find_users(self):
        return self.dao.find(projection={"password": 0})

    def update_avatar(self, user_id: str) -> None:
        email = self.get_document(id=user_id).get("email")
        if not email:
            raise ValueError("Adresse e-mail introuvable pour l'utilisateur")
        save_avatar(
            generate_avatar(email, 800, variant=random.randint(1, int(10e16))),
            os.path.join("src", "public", "avatars"),
            f"{user_id}.png",
        )

    def restore_user(self, user_id: str) -> None:
        self.db['documents'].delete_many({"created_by._id": user_id})
        dir = os.path.join("..", "sardine.documents", user_id)
        if os.path.exists(dir):
            shutil.rmtree(dir)