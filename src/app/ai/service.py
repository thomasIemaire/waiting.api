from src.helpers.base_service import BaseService

from bson.objectid import ObjectId
from pymongo.database import Database
from src.app.users.service import UsersService
from src.app.documents.service import DocumentsService

from src.helpers import documents as doc_utils

class AiService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.users_service = UsersService(db)
        self.documents_service = DocumentsService(db)

    def analyze_document(self, data: dict, *, user_id: str = None) -> dict:
        document_id = data.get("document_id", None)
        if not document_id:
            raise ValueError("document_id is required")

        document = self.documents_service.get_document(id=document_id)
        document_path = document.get("storage", "").get("path", "")
        if document.get("analysis", None) is not None:
            return document
        
        print(f"[AiService] Analyzing document at path: {document_path}")

        document_data = doc_utils.file_to_base64(document_path)
        from src.helpers import flows

        result = flows.run(flows.flow, base64=document_data, debug=True)
        type = result.get("type", "unknown")
        analysis = result.get(type, {})

        document["type"] = type
        document["analysis"] = analysis

        self.documents_service.dao.update_one(
            {"_id": ObjectId(document_id)},
            {
                "type": type,
                "analysis": analysis,
            }
        )

        return document
