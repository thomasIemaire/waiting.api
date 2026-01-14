import base64
import numpy as np
import cv2
import datetime
from src.helpers.base_service import BaseService
from pymongo.database import Database

class SardineService(BaseService):

    def __init__(self, db: Database) -> None:
        super().__init__(db)
        self.documents = db.get_collection("documents")
        self.classifications = db.get_collection("classifications")

    def create_magnetic_zone(self, b64_string: str, zone_data: dict, user_id: str) -> dict:
        if not b64_string:
            raise ValueError("Base64 data is required")
        if not zone_data:
            raise ValueError("Zone data is required")

        try:
            # 1. Décodage
            img = self._base64_to_cv2(b64_string)
            if img is None:
                return zone_data
            
            height, width = img.shape[:2]

            # === NOUVELLE LOGIQUE DE DÉTECTION ===
            
            # A. Détection par Saturation (Pour les couleurs claires comme le jaune)
            # On convertit en HSV (Teinte, Saturation, Valeur)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # On récupère le canal Saturation (S). Le blanc a une saturation de 0. Le jaune a une saturation élevée.
            s_channel = hsv[:, :, 1]
            # Tout ce qui a un peu de couleur (saturation > 25) devient blanc (détecté)
            _, mask_color = cv2.threshold(s_channel, 25, 255, cv2.THRESH_BINARY)

            # B. Détection par Luminosité (Pour le texte noir/gris)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Tout ce qui est plus foncé que 250 (sur 255) devient blanc (détecté)
            # On utilise un seuil haut (250) pour capter même le gris très clair
            _, mask_dark = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # C. Combinaison des deux masques
            # Le contenu = (C'est coloré) OU (C'est sombre)
            binary = cv2.bitwise_or(mask_color, mask_dark)

            # === FIN NOUVELLE LOGIQUE ===

            # 3. Conversion des coordonnées (Ratios 0-1 -> Pixels)
            x_ratio = zone_data.get('x', 0)
            y_ratio = zone_data.get('y', 0)
            w_ratio = zone_data.get('w', 0)
            h_ratio = zone_data.get('h', 0)

            x_px = int(x_ratio * width)
            y_px = int(y_ratio * height)
            w_px = int(w_ratio * width)
            h_px = int(h_ratio * height)

            # Sécurité bords
            x_px = max(0, x_px)
            y_px = max(0, y_px)
            w_px = min(width - x_px, w_px)
            h_px = min(height - y_px, h_px)

            # 4. Extraction ROI
            roi = binary[y_px:y_px+h_px, x_px:x_px+w_px]

            # 5. Magnétisme
            points = cv2.findNonZero(roi)

            if points is not None:
                rx, ry, rw, rh = cv2.boundingRect(points)

                # Petit padding pour ne pas coller au pixel près
                padding = 1 
                rx = max(0, rx - padding)
                ry = max(0, ry - padding)
                rw = min(w_px - rx, rw + (padding * 2))
                rh = min(h_px - ry, rh + (padding * 2))

                final_x = (x_px + rx) / width
                final_y = (y_px + ry) / height
                final_w = rw / width
                final_h = rh / height

                zone_data['x'] = final_x
                zone_data['y'] = final_y
                zone_data['w'] = final_w
                zone_data['h'] = final_h
                zone_data['is_magnetic'] = True

            return zone_data

        except Exception as e:
            print(f"Error processing magnetic zone: {e}")
            return zone_data

    def _base64_to_cv2(self, b64_string: str):
        """Helper pour convertir base64 string en image OpenCV"""
        # Supprimer le header type 'data:image/png;base64,' si présent
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        img_data = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    
    def save_document(self, data: dict, user_id: str) -> dict:
        """Sauvegarde une page spécifique d'un document"""
        # Construction du nom de fichier avec le suffixe de page si nécessaire
        filename = data.get("filename")
        
        doc = {
            "filename": filename,
            "classification": data.get("classification"),
            "base64": data.get("base64"), # On stocke l'image de la page
            "zones": data.get("zones", []),
            "width": data.get("width"),
            "height": data.get("height"),
            "user_id": user_id,
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }

        # On écrase si le fichier existe déjà pour cette classification (update)
        # ou on crée un nouveau
        self.documents.update_one(
            {"filename": filename, "classification": data.get("classification")},
            {"$set": doc},
            upsert=True
        )
        return {"success": True, "filename": filename}

    def get_documents_tree(self, user_id: str) -> dict:
        """Récupère l'arborescence : Classifications -> Fichiers"""
        
        # 1. Récupérer toutes les classifications OFFICIELLES
        class_cursor = self.classifications.find({})
        tree = {c["name"]: [] for c in class_cursor}

        # 2. Récupérer les documents
        docs_cursor = self.documents.find({}, {"base64": 0}) 

        for doc in docs_cursor:
            classification = doc.get("classification")
            
            # MODIFICATION ICI :
            # On vérifie que 'classification' existe ET qu'elle fait partie des clés de l'arbre (donc qu'elle existe officiellement)
            # Si le document n'a pas de classification ou si elle est inconnue, on l'ignore.
            if classification and classification in tree:
                tree[classification].append({
                    "name": doc.get("filename"),
                    "zones_count": len(doc.get("zones", [])),
                    "_id": str(doc.get("_id"))
                })
            
        return tree

    def get_document_content(self, filename: str, classification: str) -> dict:
        """Récupère un document complet avec son image base64"""
        doc = self.documents.find_one({"filename": filename, "classification": classification})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
        return None

    def create_classification(self, name: str) -> dict:
        if not name:
            raise ValueError("Name required")
        
        if self.classifications.find_one({"name": name}):
             raise ValueError("Classification already exists")

        self.classifications.insert_one({"name": name, "created_at": datetime.datetime.utcnow()})
        return {"name": name}