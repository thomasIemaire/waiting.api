from pyparsing import Optional
from pymongo.database import Database
import os, base64

from src.app.users.service import UsersService
from src.helpers.avatar import generate_avatar, save_avatar
from src.helpers import utils

from src.helpers.base_service import BaseService

from src.app.documents.dao import DocumentsDao
from src.helpers import documents as doc_utils

class DocumentsService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.dao = DocumentsDao(self.db)
        self.user_service = UsersService(db)

    def find_documents_by_user_id(self, user_id: str):
        return self.dao.find({"created_by._id": user_id})
    
    def find_document(
        self,
        document_id: str,
        user_id: str
    ):
        document = self.get_document(
            id=document_id,
            projection={
                "_id": 1,
                "filename": 1,
                "content_type": 1,
                "storage": 1,
                "created_by": 1,
                "created_at": 1,
            },
        )
        if not document:
            raise ValueError("Document introuvable")
        
        if not user_id == str(document.get("created_by", {}).get("_id")):
            raise ValueError("Accès refusé au document")

        storage = document.get("storage") or {}
        file_path: Optional[str] = storage.get("path")
        if not file_path or not os.path.exists(file_path):
            raise RuntimeError("Fichier introuvable sur le disque")

        with open(file_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")

        response = { **self.dao.serialize(document), "data": b64 }

        return response

    def create_document(self, document_data: dict, user_id: str):
        filename = document_data.get("filename", None)
        if not filename:
            raise ValueError("Le nom du fichier est requis")

        content_type = document_data.get("contentType", None)
        if not content_type:
            raise ValueError("Le type de contenu est requis")

        data = document_data.get("data", None)
        if not data:
            raise ValueError("Le contenu du document est requis")
        
        user = self.user_service.get_document(id=user_id, projection={
            "_id": 1,
            "firstname": 1,
            "lastname": 1,
            "email": 1
        })
        
        raw, _ = doc_utils._decode_base64_any(data)

        doc = {
            "filename": filename,
            "content_type": content_type,
            "created_at": utils.get_current_time(),
            "created_by": user,
            "storage": {"type": "disk", "path": None},
        }

        insert_result = self.dao.insert_one(doc)
        doc_id = getattr(insert_result, "inserted_id", None) or doc.get("_id")
        if not doc_id:
            raise RuntimeError("Impossible de récupérer l'id du document inséré")

        base_dir = os.path.join("..", "sardine.documents", str(user_id))
        os.makedirs(base_dir, exist_ok=True)

        ext = doc_utils._ext_from(content_type, filename)
        safe_name = f"{str(doc_id)}{ext}" if ext else f"{str(doc_id)}"
        file_path = os.path.join(base_dir, safe_name)

        try:
            with open(file_path, "wb") as f:
                f.write(raw)
        except Exception as e:
            try:
                self.dao.delete_one({"_id": doc_id})
            finally:
                raise RuntimeError(f"Échec d’écriture du fichier: {e}")

        self.dao.update_one(
            {"_id": doc_id},
            {"storage": {"type": "disk", "path": file_path}}
        )

        saved = self.get_document(id=str(doc_id))
        return self.dao.serialize(saved)
