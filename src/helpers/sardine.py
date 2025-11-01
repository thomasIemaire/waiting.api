from pathlib import Path
import shutil
import sys, os, re
from typing import List, Tuple, Optional

from ultralytics import YOLO
import cv2, numpy as np, pandas as pd, pytesseract
from pytesseract import Output

import pytesseract
from PIL import Image, ImageFilter, ImageOps
from concurrent.futures import ThreadPoolExecutor

# === AJOUTS POUR BASE64 ===
import base64, io

# ===================== PDF -> IMAGES =====================
def pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 350) -> List[Path]:
    try:
        import fitz
    except Exception:
        print("[ERROR] PyMuPDF (pymupdf) requis pour lire les PDF. Faites: pip install pymupdf", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    with fitz.open(str(pdf_path)) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_img = out_dir / f"{pdf_path.stem}_p{i:03d}.png"
            pix.save(str(out_img))
            paths.append(out_img)

    return paths

def pdf_to_pil_images(pdf_path: Path, dpi: int = 350) -> List[Image.Image]:
    """Rend les pages d'un PDF en PIL.Image, sans sauvegarde disque (à partir d'un chemin)."""
    try:
        import fitz
    except Exception:
        print("[ERROR] PyMuPDF (pymupdf) requis pour lire les PDF. Faites: pip install pymupdf", file=sys.stderr)
        sys.exit(1)

    images: List[Image.Image] = []
    with fitz.open(str(pdf_path)) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    return images

# ===================== ZONE TABLE -> TABLE STRUCTURE =====================
def _detect_columns_from_table_region(
    pil_img: Image.Image,
    model_tbl: YOLO,
    conf: float = 0.25,
    pad: int = 6,
) -> list[tuple[int, int, int, int]]:
    """
    Détecte les colonnes (classe == 1) dans une zone de tableau.
    Retourne une liste de boxes (x1,y1,x2,y2) triées de gauche à droite.
    """
    if model_tbl is None:
        return []

    results = model_tbl.predict(
        source=[pil_img],
        save=False,
        conf=conf,
        verbose=False,
    )
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy
    cls = r.boxes.cls

    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
    else:
        xyxy = np.array(xyxy)
    if hasattr(cls, "cpu"):
        cls = cls.cpu().numpy()
    else:
        cls = np.array(cls)

    # Ne garder que les colonnes (classe 1)
    mask = (cls.astype(int).reshape(-1) == 1)
    cols = xyxy[mask]
    if cols.size == 0:
        return []

    # Trier gauche -> droite
    cols = cols.astype(int)
    order = np.argsort(cols[:, 0])
    cols = cols[order]

    # Padding et clipping image
    W, H = pil_img.size
    boxes: list[tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in cols:
        x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad); y2p = min(H, y2 + pad)
        if x2p > x1p and y2p > y1p:
            boxes.append((x1p, y1p, x2p, y2p))
    return boxes

def _ocr_lines_with_y_single_pass(pil_img: Image.Image, *, lang: str, config: str):
    """
    Un SEUL appel à image_to_data par colonne.
    Retourne (lines, y_centers) pour cette colonne.
    """
    try:
        df = pytesseract.image_to_data(pil_img, lang=lang, config=config, output_type=Output.DATAFRAME)
    except Exception:
        txt = pytesseract.image_to_string(pil_img, lang=lang, config=config)
        lines = [re.sub(r"\s+", " ", t).strip() for t in txt.splitlines() if t.strip()]
        ycs = np.arange(len(lines))*20 + 10
        return lines, ycs

    df = df.copy()
    if 'conf' in df.columns:
        df = df[df.conf.astype(str) != '-1']
    if 'text' not in df.columns or df.empty:
        return [], np.array([])

    df = df.dropna(subset=['text'])
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].ne("")]
    if df.empty:
        return [], np.array([])

    groups = df.groupby(['block_num', 'par_num', 'line_num'], sort=False)
    ordered = sorted(groups, key=lambda kv: (kv[1]['top'].min(), kv[1]['left'].min()))

    lines, ycs = [], []
    for _, g in ordered:
        g = g.sort_values('left')
        t = " ".join(g['text'].tolist())
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            continue
        top = int(g['top'].min())
        bot = int((g['top'] + g['height']).max())
        yc  = (top + bot)//2
        lines.append(t); ycs.append(yc)

    return lines, np.array(ycs, dtype=float)

def _cluster_row_centers(all_y: list[np.ndarray]) -> np.ndarray:
    """
    Fusionne tous les y-centers de toutes les colonnes -> centres de lignes communs.
    Tolérance = 0.55 * médiane des gaps globaux (ou 12 px par défaut).
    """
    ys = np.concatenate([y for y in all_y if y.size], axis=0)
    if ys.size == 0:
        return np.array([], dtype=float)

    ys = np.sort(ys)
    diffs = np.diff(ys)
    med_gap = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 20.0
    tol = max(12.0, med_gap)

    centers = []
    cur = ys[0]
    cluster = [cur]
    for y in ys[1:]:
        if y - cur <= tol:
            cluster.append(y)
            cur = y
        else:
            centers.append(float(np.mean(cluster)))
            cluster = [y]
            cur = y
    centers.append(float(np.mean(cluster)))
    return np.array(centers, dtype=float)

def _assign_to_rows(col_lines: list[str], col_y: np.ndarray, row_centers: np.ndarray) -> list[str]:
    """
    Assigne chaque ligne de la colonne au centre de ligne le plus proche.
    Insère "" pour les lignes non présentes.
    """
    out = [""] * len(row_centers)
    if len(col_lines) == 0 or row_centers.size == 0:
        return out
    for t, y in zip(col_lines, col_y):
        idx = int(np.argmin(np.abs(row_centers - y)))
        out[idx] = t
    return out
# === NOUVEAU: PDF bytes -> PIL sans passer par un fichier ===
def pdf_bytes_to_pil_images(pdf_bytes: bytes, dpi: int = 350) -> List[Image.Image]:
    """Rend les pages d'un PDF (fourni en bytes/base64) en PIL.Image, en mémoire."""
    try:
        import fitz  # PyMuPDF
    except Exception:
        print("[ERROR] PyMuPDF (pymupdf) requis pour lire les PDF. Faites: pip install pymupdf", file=sys.stderr)
        sys.exit(1)

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
    """
    Pré-traitement rapide & conservatif pour Tesseract:
    - convertit en niveaux de gris via 'canal encre' K = 255 - max(R,G,B)
      (texte sombre devient très sombre, fonds colorés deviennent clairs)
    - légère débruitage, autocontraste doux
    - upscaling *uniquement* si la zone est vraiment petite
    """
    import numpy as np, cv2
    from PIL import ImageOps, ImageFilter

    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    # 'encre' : noir = 255, blanc = 0
    ink = 255 - np.max(arr, axis=2).astype(np.uint8)

    # On veut texte sombre sur fond clair => gris = 255 - ink
    gray = (255 - ink).astype(np.uint8)

    # Débruitage léger & contraste doux (peu coûteux)
    gray = cv2.medianBlur(gray, 3)
    gpil = Image.fromarray(gray)
    gpil = ImageOps.autocontrast(gpil, cutoff=1)
    gpil = gpil.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=2))

    # Upscaling seulement si petite hauteur
    W, H = gpil.size
    if H < 120:
        scale = max(2.0, 120 / max(1, H))
        new_w = int(W * scale)
        new_h = int(H * scale)
        gpil = gpil.resize((new_w, new_h), resample=Image.BICUBIC)

    return gpil

def _deskew_and_orient(pil_img: Image.Image) -> Image.Image:
    """
    Utilise Tesseract OSD pour détecter l’orientation et corrige la rotation.
    Si OSD échoue, on retourne l'image telle quelle.
    """
    try:
        osd = pytesseract.image_to_osd(pil_img, output_type=pytesseract.Output.DICT)
        rotation = osd.get("rotate", 0) or 0  # 0, 90, 180, 270
        if rotation and rotation % 360 != 0:
            pil_img = pil_img.rotate(360 - rotation, expand=True)
    except Exception:
        pass
    return pil_img

def _ocr_pil_image(pil_img: Image.Image, *, lang: str = "fra+eng",
                   tesseract_cmd: Optional[str] = r"C:\Users\Utilisateur\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
                   type_box: Optional[int] = None,
                   model_tbl: YOLO = None) -> str:
    """OCR robustifié sur une image PIL (orientation + prétraitement)."""
    cls_id = _to_class_id(type_box)

    try:
        if tesseract_cmd and Path(tesseract_cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    except Exception:
        pass

    pil_img = pil_img.convert("RGB")
    pil_img = _deskew_and_orient(pil_img)
    pil_img = _preprocess_for_ocr(pil_img)

    base_config = (
        "--oem 3 --psm 6 "
        "-c preserve_interword_spaces=1 "
        "-c tessedit_do_invert=1"
    )

    try:
        match cls_id:
            case 2:
                if model_tbl is None:
                    return {"type": "table", "header": [], "columns": []}

                # Détection colonnes (classe 1), gauche -> droite
                col_boxes = _detect_columns_from_table_region(pil_img, model_tbl=model_tbl, conf=0.25, pad=6)
                if not col_boxes:
                    # Fallback: une colonne unique
                    tmp = _preprocess_for_ocr(_deskew_and_orient(pil_img.convert("RGB")))
                    lines, ycs = _ocr_lines_with_y_single_pass(tmp, lang=lang, config="--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_do_invert=0")
                    header = [lines[0]] if lines else []
                    body   = [lines[1:]] if len(lines) > 1 else []
                    return {"type": "table", "header": header, "columns": body}

                # OCR une seule fois par colonne
                headers: list[str] = []
                col_texts: list[list[str]] = []
                col_ycs:   list[np.ndarray] = []

                for (x1, y1, x2, y2) in col_boxes:
                    col = pil_img.crop((x1, y1, x2, y2))
                    col = _deskew_and_orient(col)
                    col = _preprocess_for_ocr(col)
                    lines, ycs = _ocr_lines_with_y_single_pass(
                        col, lang=lang,
                        config="--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_do_invert=0"
                    )

                    if not lines:
                        headers.append("")
                        col_texts.append([])
                        col_ycs.append(np.array([]))
                        continue

                    headers.append(lines[0])           # 1re ligne = entête
                    col_texts.append(lines[1:])        # corps
                    col_ycs.append(ycs[1:] if len(ycs) > 1 else np.array([]))

                # Centres de lignes communs (alignement global)
                row_centers = _cluster_row_centers(col_ycs)

                # Colonnes alignées (insertion automatique des "")
                aligned_cols = [
                    _assign_to_rows(texts, ycs, row_centers)
                    for texts, ycs in zip(col_texts, col_ycs)
                ]

                # Harmoniser longueurs
                max_rows = max((len(c) for c in aligned_cols), default=0)
                aligned_cols = [c + ([""]*(max_rows - len(c))) for c in aligned_cols]

                return {"type": "table", "header": headers, "columns": aligned_cols}

        text = pytesseract.image_to_string(pil_img, lang=lang, config=base_config)
        text = text.replace('\r', '\n')
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        return (text or "").strip()

    except pytesseract.TesseractNotFoundError:
        print("[ERROR] Tesseract introuvable. Installez-le et/ou mettez à jour 'tesseract_cmd'.", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"[ERROR] OCR échec: {e}", file=sys.stderr)
        return ""

def _ocr_many_pil(crops: list[Image.Image],
                  max_workers: int | None = None,
                  types_box: list[int | None] | None = None,
                  model_tbl: YOLO = None) -> list[str]:
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)
    n = len(crops)
    tb = (types_box or [])
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_ocr_pil_image, crops[i],
                      type_box=(tb[i] if i < len(tb) else None),
                      model_tbl=model_tbl)
            for i in range(n)
        ]
        return [f.result() for f in futures]

# ===================== OCR WRAPPER =====================
def read_image(img_path: Path, *, lang: str = "fra+eng",
               tesseract_cmd: Optional[str] = r"C:\Users\Utilisateur\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
               ) -> str:
    with Image.open(img_path) as im:
        return _ocr_pil_image(im, lang=lang, tesseract_cmd=tesseract_cmd)

def move_label_txt_next_to_images(save_dir: Path) -> None:

    label_dir = save_dir / "labels"
    if not label_dir.exists():
        return
    txts = list(label_dir.glob("*.txt"))
    for src in txts:
        dst = save_dir / src.name
        try:
            src.replace(dst)
        except Exception:
            shutil.copy2(src, dst)
            src.unlink(missing_ok=True)
    try:
        next(label_dir.iterdir())
    except StopIteration:
        label_dir.rmdir()

def classify_pages(model_cls: YOLO, sources: list, device: str = "cpu"):
    # sources: list de PIL.Image, np.ndarray, chemins... Ultralytics gère.
    results = model_cls.predict(source=sources, device=device, save=False, verbose=False)
    counts = {}
    for r in results:
        top1 = r.probs.top1
        name = r.names[top1]
        counts[name] = counts.get(name, 0) + 1
    if counts:
        winner = max(counts.items(), key=lambda kv: kv[1])[0]
        total = sum(counts.values())
        print("[CLS] Majority vote:", winner, f"({counts[winner]}/{total})")
        print("[CLS] Breakdown:", counts)
        return winner
    print("[CLS] Aucun résultat de classification.")
    return None

def _to_class_id(v) -> Optional[int]:
    if v is None:
        return None
    try:
        if hasattr(v, "item"):
            v = v.item()
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return None

# === AJOUTS BASE64 : helpers ===
_DATAURL_RE = re.compile(r'^data:(?P<mime>[\w/+.\-]+);base64,(?P<data>.+)$', re.IGNORECASE)

def _try_decode_base64(s: str) -> tuple[Optional[bytes], Optional[str]]:
    """
    Décode soit une data URL, soit du base64 brut (standard ou URL-safe).
    Retourne (bytes, mime) ; mime peut être None si non déductible.
    """
    if not isinstance(s, str) or not s.strip():
        return None, None

    # 1) data URL ?
    m = _DATAURL_RE.match(s.strip())
    if m:
        try:
            raw = base64.b64decode(m.group("data"), validate=True)
            mime = (m.group("mime") or "").lower()
            return raw, mime or _sniff_mime(raw)
        except Exception:
            return None, None

    # 2) base64 brut (on tolère espaces / retours ligne)
    b64_clean = re.sub(r'\s+', '', s)
    if len(b64_clean) < 16:
        return None, None  # trop court pour être utile

    # Essai en alphabet standard
    try:
        raw = base64.b64decode(b64_clean, validate=True)
        return raw, _sniff_mime(raw)
    except Exception:
        pass

    # Essai en URL-safe ('-' '_' au lieu de '+' '/')
    try:
        raw = base64.urlsafe_b64decode(b64_clean + '==')  # padding permissif
        # urlsafe_b64decode n'a pas l'option validate, on fait un check simple
        if raw:
            return raw, _sniff_mime(raw)
    except Exception:
        pass

    return None, None

def _sniff_mime(data: bytes) -> str | None:
    """Détecte grossièrement le type MIME depuis la signature des bytes."""
    if not data or len(data) < 4:
        return None
    # PDF
    if data[:4] == b"%PDF":
        return "application/pdf"
    # PNG
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    # JPEG
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    # GIF
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    # WEBP (RIFF container)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    # TIFF
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return None

def _load_pil_from_bytes(img_bytes: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return im.copy()

# ===================== PIPELINE =====================
def inference(
    *,
    model_detect_path: str,
    model_class_path: str,
    model_table_path: str,
    img_b64: str,                 # <-- base64 obligatoire maintenant
    device: str = "cpu",
    conf: float = 0.25,
    pdf_dpi: int = 200
):
    """
    Traite EXCLUSIVEMENT une entrée base64 (data URL ou base64 brut).
    Retourne:
      - cls_result: label global (majority vote)
      - pages: List[List[str]] = OCR par zones détectées pour chaque page

    `img_b64` doit être :
      - une data URL base64 (image/* ou application/pdf), ex: 'data:application/pdf;base64,....'
      - ou du base64 brut (image ou PDF)
    """
    # Charger les modèles
    model_det = YOLO(model_detect_path)
    model_cls = YOLO(model_class_path)
    model_tbl = YOLO(model_table_path)

    # ---------- Décodage base64 uniquement ----------
    raw_bytes, mime = _try_decode_base64(img_b64)
    if raw_bytes is None:
        raise ValueError(
            "Entrée invalide : cette fonction n'accepte plus que du base64 "
            "(data URL 'data:...;base64,...' ou base64 brut)."
        )

    # PDF ou image ?
    is_pdf = (mime == "application/pdf") or (mime is None and raw_bytes[:4] == b"%PDF")
    if is_pdf:
        sources_pil: list[Image.Image] = pdf_bytes_to_pil_images(raw_bytes, dpi=pdf_dpi)
    else:
        sources_pil = [_load_pil_from_bytes(raw_bytes)]

    if not sources_pil:
        raise FileNotFoundError("Aucune page/image décodée depuis le base64 fourni.")

    # 1) Détection (en mémoire)
    print(f"[INFO] DETECT on {len(sources_pil)} image(s)")
    det_results = model_det.predict(
        source=sources_pil,
        save=False,
        save_txt=False,
        conf=conf,
        device=device,
        verbose=True,
    )

    # 2) Classification (en mémoire)
    print(f"[INFO] CLASSIFY on {len(sources_pil)} image(s)")
    cls_result = classify_pages(model_cls, sources_pil, device=device)

    # 3) OCR par zones
    pages: list[list[str]] = []
    pad = 8

    for im, r in zip(sources_pil, det_results):
        W, H = im.size
        page_texts: list[str] = []

        if r.boxes is None or len(r.boxes) == 0:
            page_texts.append(_ocr_pil_image(im, model_tbl=model_tbl))
        else:
            bcls = r.boxes.cls
            if hasattr(bcls, "cpu"):
                bcls = bcls.detach().cpu().numpy()
            else:
                bcls = np.asarray(bcls)

            xyxy = r.boxes.xyxy
            if hasattr(xyxy, "cpu"):
                xyxy = xyxy.cpu().numpy()
            else:
                xyxy = np.array(xyxy)

            xyxy = xyxy.astype(int)
            order = np.lexsort((xyxy[:, 0], xyxy[:, 1]))
            xyxy = xyxy[order]
            bcls = bcls.astype(float).reshape(-1)[order]
            bcls_list = [_to_class_id(v) for v in bcls]

            crops: list[Image.Image] = []
            for (x1, y1, x2, y2) in xyxy:
                x1p = max(0, x1 - pad); y1p = max(0, y1 - pad)
                x2p = min(W, x2 + pad); y2p = min(H, y2 + pad)
                if x2p <= x1p or y2p <= y1p:
                    continue
                crops.append(im.crop((x1p, y1p, x2p, y2p)))

            if crops:
                texts = _ocr_many_pil(crops, types_box=bcls_list, model_tbl=model_tbl)
                page_texts = [t.strip() if isinstance(t, str) else t for t in texts]

            if not page_texts:
                page_texts.append(_ocr_pil_image(im, model_tbl=model_tbl))

        pages.append(page_texts)

    print("[INFO] Inference complete (base64-only).")
    return cls_result, pages

def delete_tmp_dir(path: Path):
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
        print(f"[INFO] Temporary directory {path} deleted.")
