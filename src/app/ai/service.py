from src.helpers.base_service import BaseService

from bson.objectid import ObjectId
from pymongo.database import Database
from src.app.users.service import UsersService
from src.app.documents.service import DocumentsService
from src.app.flows.service import FlowService # Import ajouté

from src.helpers import documents as doc_utils
from src.helpers import flows # Import du module flows modifié

class AiService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.users_service = UsersService(db)
        self.documents_service = DocumentsService(db)
        self.flow_service = FlowService(db) # Service ajouté

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
        
        # --- RÉCUPÉRATION DYNAMIQUE DU FLUX ---
        
        # 1. On cherche un flow_id dans la requête, sinon on prend le premier disponible
        flow_id = data.get("flow_id", None)
        flow_doc = None
        print(f"[AiService] Requested flow_id: {flow_id}")

        if not flow_id:
            flow_doc = self.flow_service.get_default_flow()
        elif flow_id:
            flow_doc = self.flow_service.get_flow(flow_id=flow_id)
        else:
            # Fallback : on prend le premier flux trouvé
            all_flows = list(self.flow_service.dao.find({}, limit=1))
            if all_flows:
                flow_doc = self.flow_service.dao.serialize(all_flows[0])
        
        if not flow_doc:
            raise ValueError("Aucun flux actif trouvé pour traiter le document.")

        print(f"[AiService] Using flow: {flow_doc.get('name')} ({flow_doc.get('_id')})")

        # 2. Transformation du graphe JSON (Frontend) vers Graphe Exécutable (Engine)
        engine_flow = flows.transform_graph(flow_doc.get("data", {}))

        # 3. Exécution
        result = flows.run(engine_flow, base64=document_data, debug=True)
        
        # --- FIN MODIFICATIONS ---

        type = result.get("type", "unknown")
        analysis = result.get("analysis", {})

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