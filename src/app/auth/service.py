from typing import Any, Dict
from src.app.users.dao import UsersDao
from pymongo.database import Database
from src.helpers.base_service import BaseService
from src.helpers.avatar import save_avatar, generate_avatar
from flask_jwt_extended import create_access_token, create_refresh_token
import os

from src.helpers import utils

class AuthService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = UsersDao(self.db)

    def register(self, data: Dict[str, Any]) -> Dict[str, Any]:
        email = (data.get("email") or "").strip().lower()

        if not email or "@" not in email:
            raise ValueError("Email invalide")

        if self.document_exists(query={"email": email}):
            raise ValueError("Email déjà utilisé")

        if not data.get("firstname") or not data.get("lastname"):
            raise ValueError("Nom et prénom requis")
        
        if not data.get("password"):
            raise ValueError("Mot de passe requis")

        apikey = utils.generate_apikey()

        role = "user"
        user = {
            "email": email,
            "firstname": data["firstname"],
            "lastname": data["lastname"],
            "apikey": apikey,
            "password": utils.hash_password(data["password"], apikey),
            "role": role,
        }

        user = self.dao.serialize(self.dao.insert_one(user))

        save_avatar(
            generate_avatar(email, 800),
            os.path.join("src", "public", "avatars"),
            f"{user['_id']}.png"
        )

        token, refresh = self.token(user=user)
        user.pop("password", None)

        return {"token": token, "refresh_token": refresh, "user": user}

    def signin(self, data: Dict[str, Any]) -> Dict[str, Any]:
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if not email or not password:
            raise ValueError("Email et mot de passe requis")

        user = self.dao.find_one({"email": email})
        if not user or not utils.verify_password(password, user["password"], user["apikey"]):
            raise ValueError("Email ou mot de passe invalide")

        user.pop("password", None)

        token, refresh = self.token(user=user)
        return {"token": token, "refresh_token": refresh, "user": user}
    
    def token(self, *, user_id: str = None, user: Dict[str, Any]) -> tuple:
        if not user:
            user = self.get_document(id=user_id)
        
        token = create_access_token(identity=str(user["_id"]), additional_claims={"role": user.get("role", "user")})
        refresh = create_refresh_token(identity=str(user["_id"]))

        return (token, refresh)
    
    def signin_token(self, user_id: str) -> Dict[str, Any]:
        user = self.get_document(id=user_id, projection={"password": 0})
        token, refresh = self.token(user=user)
        return {"token": token, "refresh_token": refresh, "user": user}

    def email_exists(self, email: str) -> bool:
        email = email.strip().lower()
        return self.document_exists(query={"email": email})
