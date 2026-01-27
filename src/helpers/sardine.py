from pathlib import Path
import sys, os, re
import threading
from typing import List, Optional

from ultralytics import YOLO
import cv2, numpy as np, pytesseract
from PIL import Image, ImageFilter, ImageOps
from concurrent.futures import ThreadPoolExecutor

import base64, io

from dotenv import load_dotenv
load_dotenv()

DEFAULT_TESSERACT_CMD = os.getenv("TESSERACT_CMD")

# ===================== CACHE MODEL YOLO =====================
_YOLO_CACHE: dict[str, YOLO] = {}
_YOLO_LOCK = threading.Lock()

def get_cached_yolo_model(path: str) -> YOLO:
    abs_path = os.path.abspath(path)
    if abs_path in _YOLO_CACHE:
        return _YOLO_CACHE[abs_path]

    with _YOLO_LOCK:
        if abs_path not in _YOLO_CACHE:
            print(f"[INFO] Loading YOLO model into memory: {abs_path}")
            try:
                _YOLO_CACHE[abs_path] = YOLO(path)
            except Exception as e:
                print(f"[ERROR] Failed to load model {path}: {e}")
                raise e
        return _YOLO_CACHE[abs_path]

# ===================== IMAGES UTILS =====================
def pdf_bytes_to_pil_images(pdf_bytes: bytes, dpi: int = 350) -> List[Image.Image]:
    try:
        import fitz  # PyMuPDF
    except Exception:
        print("[ERROR] PyMuPDF (pymupdf) requis pour lire les PDF.", file=sys.stderr)
        return []

    images: List[Image.Image] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images

def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    ink = 255 - np.max(arr, axis=2).astype(np.uint8)
    gray = (255 - ink).astype(np.uint8)
    gray = cv2.medianBlur(gray, 3)
    gpil = Image.fromarray(gray)
    gpil = ImageOps.autocontrast(gpil, cutoff=1)
    gpil = gpil.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=2))
    
    W, H = gpil.size
    if H < 120:
        scale = max(2.0, 120 / max(1, H))
        new_w = int(W * scale)
        new_h = int(H * scale)
        gpil = gpil.resize((new_w, new_h), resample=Image.BICUBIC)
    return gpil

def _deskew_and_orient(pil_img: Image.Image) -> Image.Image:
    try:
        osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0) or 0
        if rotation and rotation % 360 != 0:
            pil_img = pil_img.rotate(360 - rotation, expand=True)
    except Exception:
        pass
    return pil_img

def _ocr_pil_image(pil_img: Image.Image, *, lang: str = "fra+eng") -> str:
    pil_img = pil_img.convert("RGB")
    pil_img = _deskew_and_orient(pil_img)
    pil_img = _preprocess_for_ocr(pil_img)
    text = pytesseract.image_to_string(pil_img, lang=lang, config="--oem 3 --psm 6")
    return (text or "").strip()

def _ocr_many_pil(crops: list[Image.Image], max_workers: int = 4) -> list[str]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_ocr_pil_image, c) for c in crops]
        return [f.result() for f in futures]

# ===================== BASE64 UTILS =====================
_DATAURL_RE = re.compile(r'^data:(?P<mime>[\w/+.\-]+);base64,(?P<data>.+)$', re.IGNORECASE)

def _try_decode_base64(s: str) -> tuple[Optional[bytes], Optional[str]]:
    if not isinstance(s, str) or not s.strip(): return None, None
    m = _DATAURL_RE.match(s.strip())
    if m:
        try:
            return base64.b64decode(m.group("data"), validate=True), (m.group("mime") or "").lower()
        except: return None, None
    try:
        return base64.b64decode(s), None
    except: return None, None

def _load_pil_from_bytes(img_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def classify_pages(model_cls: YOLO, sources: list, device: str = "cpu"):
    if not sources: return None
    results = model_cls.predict(source=sources, device=device, save=False, verbose=False)
    counts = {}
    for r in results:
        if hasattr(r, 'probs') and r.probs is not None:
            top1 = r.probs.top1
            name = r.names[top1]
            counts[name] = counts.get(name, 0) + 1
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    return "unknown"

# ===================== NOUVELLES FONCTIONS =====================

def load_images(img_b64: str, pdf_dpi: int = 200) -> List[Image.Image]:
    raw_bytes, mime = _try_decode_base64(img_b64)
    if raw_bytes is None:
        return []
    try:
        is_pdf = (mime == "application/pdf") or (mime is None and raw_bytes[:4] == b"%PDF")
        if is_pdf:
            return pdf_bytes_to_pil_images(raw_bytes, dpi=pdf_dpi)
        else:
            return [_load_pil_from_bytes(raw_bytes)]
    except Exception as e:
        print(f"[SARDINE] Erreur chargement images: {e}")
        return []

def predict_classification(images: List[Image.Image], model_path: str, device: str = "cpu") -> str:
    if not images: return "unknown"
    try:
        model_cls = get_cached_yolo_model(model_path)
        print(f"[INFO] CLASSIFY on {len(images)} page(s)")
        return classify_pages(model_cls, images, device=device) or "unknown"
    except Exception as e:
        print(f"[SARDINE] Erreur classification: {e}")
        return "error"

def predict_detection(images: List[Image.Image], model_path: str, device: str = "cpu", conf: float = 0.25) -> List[List[str]]:
    if not images: return []
    try:
        model_det = get_cached_yolo_model(model_path)
        print(f"[INFO] DETECT on {len(images)} page(s)")
        det_results = model_det.predict(source=images, save=False, conf=conf, device=device, verbose=False)
        
        pages_data: List[List[str]] = []
        pad = 8

        for im, r in zip(images, det_results):
            page_texts = []
            W, H = im.size
            if r.boxes and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, "cpu") else r.boxes.xyxy.numpy()
                xyxy = xyxy.astype(int)
                # Tri vertical puis horizontal
                order = np.lexsort((xyxy[:, 0], xyxy[:, 1]))
                xyxy = xyxy[order]
                
                crops = []
                for (x1, y1, x2, y2) in xyxy:
                    x1p, y1p = max(0, x1 - pad), max(0, y1 - pad)
                    x2p, y2p = min(W, x2 + pad), min(H, y2 + pad)
                    if x2p > x1p and y2p > y1p:
                        crops.append(im.crop((x1p, y1p, x2p, y2p)))
                
                if crops:
                    texts = _ocr_many_pil(crops)
                    page_texts = [t.strip() for t in texts]

            # Fallback
            if not page_texts:
                full_text = _ocr_pil_image(im)
                if full_text: page_texts.append(full_text)
                
            pages_data.append(page_texts)
            
        return pages_data

    except Exception as e:
        print(f"[SARDINE] Erreur détection: {e}")
        return []