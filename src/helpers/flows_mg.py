#!/usr/bin/env python3
"""
Pipeline généraliste : base64 (PDF ou image) -> image page 1 -> MinerU -> simplification robuste -> GLiNER2 + post-traitements.

Améliorations vs version simple :
- Conversion de tables plus robuste (détection KV vs tableau multi-colonnes, choix intelligent de la ligne d’en-tête)
- Sectionnement (HEADER / ITEMS / TOTALS / FOOTER) pour réduire les confusions vendeur/acheteur et montants
- Multi-pass extraction GLiNER2 par section + fusion/choix du meilleur candidat
- Validateurs + heuristiques de cohérence (TVA base/montant, IBAN/BIC, SIREN/SIRET/TVA)
- Fallback regex sur identifiants structurés quand GLiNER2 renvoie null

Usage:
  python pipeline_base64_mineru_gliner2_generalist.py --base64 "<...>"
  python pipeline_base64_mineru_gliner2_generalist.py --base64-file input.b64.txt
  cat input.b64.txt | python pipeline_base64_mineru_gliner2_generalist.py

Sortie: JSON sur stdout
"""

import argparse
import base64
import io
import json
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient
from gliner2 import GLiNER2


def _strip_data_url_prefix(s: str) -> str:
    if s.startswith("data:"):
        comma = s.find(",")
        if comma != -1:
            return s[comma + 1:]
    return s

def decode_base64_to_bytes(b64_str: str) -> bytes:
    b64_str = _strip_data_url_prefix(b64_str.strip())
    b64_str = re.sub(r"\s+", "", b64_str)
    return base64.b64decode(b64_str, validate=False)

def bytes_to_first_page_image(data: bytes) -> Image.Image:
    if data[:4] == b"%PDF":
        if fitz is None:
            raise RuntimeError("PyMuPDF (pymupdf) n'est pas installé: impossible de rendre un PDF.")
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.page_count < 1:
            raise ValueError("PDF vide (0 page).")
        page = doc.load_page(0)
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    return Image.open(io.BytesIO(data)).convert("RGB")

def normalize_image(image: Image.Image, max_dimension: int = 1600) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) > max_dimension:
        image = image.copy()
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    return image


MINERU_MODEL_ID = "opendatalab/MinerU2.5-2509-1.2B"

def load_mineru_client() -> MinerUClient:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MINERU_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if use_cuda else None,
        )
    except TypeError:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MINERU_MODEL_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)

    model.eval()
    processor = AutoProcessor.from_pretrained(MINERU_MODEL_ID, use_fast=True)

    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    return MinerUClient(backend="transformers", model=model, processor=processor)


class HTMLTableWithSpansParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[Tuple[str, int, int]]] = []
        self._in_cell = False
        self._cell_text_parts: List[str] = []
        self._current_row: List[Tuple[str, int, int]] = []
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_text_parts = []
            attr_dict = {k.lower(): v for k, v in attrs}
            self._cell_rowspan = int(attr_dict.get("rowspan", "1") or "1")
            self._cell_colspan = int(attr_dict.get("colspan", "1") or "1")

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text_parts.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("td", "th"):
            text = "".join(self._cell_text_parts).strip()
            text = " ".join(text.split())
            self._current_row.append((text, self._cell_rowspan, self._cell_colspan))
            self._in_cell = False
        elif tag == "tr":
            if self._current_row:
                self.rows.append(self._current_row)

def _expand_rows_with_rowspan(parsed_rows: List[List[Tuple[str, int, int]]]) -> List[List[str]]:
    grid: List[List[str]] = []
    pending: List[Tuple[int, str]] = []

    def ensure_pending_len(n: int):
        while len(pending) < n:
            pending.append((0, ""))

    for row in parsed_rows:
        out_row: List[str] = []
        col = 0

        def consume_pending_at_col():
            nonlocal col
            ensure_pending_len(col + 1)
            rem, val = pending[col]
            if rem > 0:
                out_row.append(val)
                pending[col] = (rem - 1, val)
                col += 1
                return True
            return False

        for text, rowspan, colspan in row:
            while consume_pending_at_col():
                pass

            out_row.append(text)
            ensure_pending_len(col + 1)
            if rowspan > 1:
                pending[col] = (rowspan - 1, text)

            for _k in range(1, colspan):
                col += 1
                out_row.append("")
                ensure_pending_len(col + 1)
                if rowspan > 1:
                    pending[col] = (rowspan - 1, "")

            col += 1

        max_cols = max(len(out_row), len(pending))
        ensure_pending_len(max_cols)
        while col < max_cols:
            consume_pending_at_col()

        grid.append(out_row)

    max_cols = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([""] * (max_cols - len(r)))
    return grid


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def norm_text(s: str) -> str:
    s = strip_accents(s or "")
    s = s.lower()
    s = re.sub(r"[^\w%€$£]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

_money_re = re.compile(
    r"(?<!\w)(\d{1,3}(?:[ .]\d{3})*(?:[,.]\d{2})|\d+(?:[,.]\d{2}))(?:\s?(€|eur|usd|gbp|chf))?(?!\w)",
    re.IGNORECASE,
)

def parse_money(s: str) -> Optional[float]:
    if not s:
        return None
    m = _money_re.search(s.replace("\xa0", " "))
    if not m:
        return None
    num = m.group(1)
    num = num.replace(" ", "").replace(".", "")
    num = num.replace(",", ".")
    try:
        return float(num)
    except Exception:
        return None

def parse_percent(s: str) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"(\d{1,2}(?:[.,]\d+)?)\s*%", s)
    if not m:
        return None
    return float(m.group(1).replace(",", "."))

def looks_like_header_cell(cell: str) -> bool:
    if not cell:
        return False
    c = norm_text(cell)
    letters = sum(ch.isalpha() for ch in c)
    digits = sum(ch.isdigit() for ch in c)
    if letters == 0 or digits > letters:
        return False
    return len(c) <= 28

def guess_header_row(grid: List[List[str]]) -> int:
    candidates = range(min(4, len(grid)))
    best_i, best_score = 0, -1
    for i in candidates:
        row = grid[i]
        score = sum(1 for c in row if looks_like_header_cell(c))
        moneyish = sum(1 for c in row if parse_money(c) is not None)
        score = score - moneyish
        if score > best_score:
            best_score = score
            best_i = i
    return best_i

def is_kv_table(grid: List[List[str]]) -> bool:
    if not grid:
        return False
    ncols = len(grid[0])
    if ncols == 2:
        return True
    if ncols == 3:
        empties = [sum(1 for r in grid if not (r[c] or "").strip()) for c in range(3)]
        return max(empties) >= int(0.7 * len(grid))
    return False

def kv_table_to_text(grid: List[List[str]]) -> str:
    ncols = len(grid[0])
    col_scores = []
    for c in range(ncols):
        score = 0
        for r in grid:
            cell = (r[c] or "").strip()
            if parse_money(cell) is not None:
                score += 2
            if re.search(r"\d", cell):
                score += 1
            if re.search(r"\b\d{2}/\d{2}/\d{2,4}\b", cell):
                score += 2
        col_scores.append(score)
    val_col = max(range(ncols), key=lambda c: col_scores[c])
    key_col = 0 if val_col != 0 else 1

    lines = []
    for r in grid:
        key = (r[key_col] or "").strip()
        val = (r[val_col] or "").strip()
        if not key or not val:
            continue
        if len(norm_text(key)) > 60:
            continue
        lines.append(f"{key} : {val}")
    return "        ".join(lines).strip()

def matrix_table_to_row_blocks(grid: List[List[str]]) -> str:
    if len(grid) < 2:
        return ""
    header_i = guess_header_row(grid)
    headers = [h.strip() for h in grid[header_i]]
    body = [r for j, r in enumerate(grid) if j != header_i]

    headers = [h if h else f"col{idx+1}" for idx, h in enumerate(headers)]

    blocks = []
    for row in body:
        if sum(1 for c in row if (c or "").strip()) == 0:
            continue
        lines = []
        for key, val in zip(headers, row):
            key = (key or "").strip()
            val = (val or "").strip()
            if key and val:
                lines.append(f"{key} : {val}")
        if lines:
            blocks.append("    ".join(lines))
    return "        ".join(blocks).strip()

def html_table_to_text(html: str) -> str:
    parser = HTMLTableWithSpansParser()
    parser.feed(html)
    if not parser.rows:
        return html.strip()

    grid = _expand_rows_with_rowspan(parser.rows)
    if not grid or not grid[0]:
        return html.strip()

    if is_kv_table(grid):
        out = kv_table_to_text(grid)
        return out if out else html.strip()

    out = matrix_table_to_row_blocks(grid)
    return out if out else html.strip()


SECTION_SEP = "        "

def classify_table_kind(table_text: str) -> str:
    t = norm_text(table_text)
    item_hits = sum(
        1
        for k in [
            "designation",
            "désignation",
            "description",
            "article",
            "code article",
            "reference",
            "ref",
            "qty",
            "qte",
            "quantite",
            "pu",
            "prix unitaire",
            "montant ht",
            "unit price",
        ]
        if k in t
    )
    total_hits = sum(
        1
        for k in [
            "total",
            "net a payer",
            "net à payer",
            "subtotal",
            "grand total",
            "montant total",
            "total ttc",
            "total ht",
            "total tva",
            "vat",
            "taux tva",
            "base tva",
            "taxable base",
            "amount due",
        ]
        if k in t
    )
    if item_hits >= 3 and item_hits >= total_hits:
        return "items"
    if total_hits >= 2:
        return "totals"
    return "meta"

def blocks_to_sections(blocks: List[Dict[str, Any]]) -> Dict[str, str]:
    header_parts: List[str] = []
    meta_parts: List[str] = []
    items_parts: List[str] = []
    totals_parts: List[str] = []
    footer_parts: List[str] = []

    n = len(blocks)
    header_cut = max(3, int(0.25 * n))
    footer_cut = max(3, int(0.15 * n))

    for i, b in enumerate(blocks):
        t = (b.get("type") or "").lower()
        content = (b.get("content") or "").strip()
        if not content:
            continue

        if t == "table":
            content = html_table_to_text(content)
            kind = classify_table_kind(content)
            if kind == "items":
                items_parts.append(content)
            elif kind == "totals":
                totals_parts.append(content)
            else:
                meta_parts.append(content)
        else:
            content = content.replace("\r    ", "    ").strip()
            if i < header_cut:
                header_parts.append(content)
            elif i >= n - footer_cut:
                footer_parts.append(content)
            else:
                meta_parts.append(content)

    sections = {
        "header": SECTION_SEP.join(header_parts).strip(),
        "meta": SECTION_SEP.join(meta_parts).strip(),
        "items": SECTION_SEP.join(items_parts).strip(),
        "totals": SECTION_SEP.join(totals_parts).strip(),
        "footer": SECTION_SEP.join(footer_parts).strip(),
    }
    sections["all"] = (
        "        ### HEADER    "
        + sections["header"]
        + "        ### META    "
        + sections["meta"]
        + "        ### ITEMS    "
        + sections["items"]
        + "        ### TOTALS    "
        + sections["totals"]
        + "        ### FOOTER    "
        + sections["footer"]
    )
    return sections


GLINER2_MODEL_ID = "fastino/gliner2-large-2907"

def build_schema(extractor: GLiNER2):
    return (
        extractor.create_schema()
        .structure("seller")
            .field("name", dtype="str", description="Seller/issuer name (legal header or footer). Do not confuse with Ref/folder. Nom du vendeur/émetteur (en-tête ou footer légal). Ne pas confondre avec Réf/dossier.")
            .field("tax_id", dtype="str", description="Seller's intracom VAT (VAT, VAT ID). Ex FR..TVA intracom du vendeur (TVA, VAT ID). Ex FR..") 
            .field("siren", dtype="str", description="Seller's SIREN: 9 digits. SIREN vendeur: 9 chiffres (France)")
            .field("siret", dtype="str", description="Seller's SIRET: 14 digits. SIRET vendeur: 14 chiffres (France)")
            .field("email", dtype="str", description="Seller's email vendeur si présente")
            .field("phone", dtype="str", description="Seller's phone number si présent")
        .structure("address_seller")
            .field("street", dtype="str", description="Seller's address (number + street + additional information) example '10bis rue de Paris', or '464 Boulevard des Tamaris'. Adresse vendeur (numéro + voie + compléments) exemple '10bis rue de Paris', ou '464 Boulevard des Tamaris'")
            .field("city", dtype="str", description="City selling example 'Lyon' or 'Marseille Cedex 13' or 'Onet Le Château'. Ville vendeur exemple 'Lyon' ou 'Marseille Cedex 13' ou 'Onet Le Château'")
            .field("zip_code", dtype="str", description="Seller's postal code example '75008' or '13013'. Code postal vendeur exemple '75008' ou '13013'")
            .field("country", dtype="str", description="Seller country example 'France' or 'Germany' or 'FR'. Pays vendeur exemple 'France' ou 'Germany' ou 'FR'")
        .structure("buyer")
            .field("name", dtype="str", description="Nom acheteur / facturé à / bill to / client")
            .field("tax_id", dtype="str", description="TVA intracom acheteur si présente")
            .field("siren", dtype="str", description="SIREN acheteur si présent")
            .field("siret", dtype="str", description="SIRET acheteur si présent")
            .field("email", dtype="str", description="Email acheteur si présent (rare)")
        .structure("address_buyer")
            .field("street", dtype="str", description="Adresse acheteur (bill to)")
            .field("city", dtype="str", description="Ville acheteur")
            .field("zip_code", dtype="str", description="Code postal acheteur")
            .field("country", dtype="str", description="Pays acheteur")
        .structure("order")
            .field("number", dtype="str", description="Numéro de commande / PO / Purchase order")
        .structure("customer")
            .field("number", dtype="str", description="Numéro client / code client / customer ID")
        .structure("document")
            .field("number", dtype="str", description="Numéro facture / invoice # / document")
            .field("type", dtype="str", description="Type doc (Facture/Invoice/Avoir/Credit note)")
        .structure("items")
            .field("quantity", dtype="str", description="Quantité ligne")
            .field("reference", dtype="str", description="Référence/SKU/code article")
            .field("description", dtype="str", description="Désignation/description article/service")
            .field("unit_price_excl_tax", dtype="str", description="Prix unitaire HT")
            .field("tax_rate", dtype="str", description="Taux TVA/VAT rate (ex 20%)")
            .field("tax_amount", dtype="str", description="Montant TVA ligne si présent")
            .field("line_total_excl_tax", dtype="str", description="Total ligne HT si présent")
            .field("line_total_incl_tax", dtype="str", description="Total ligne TTC si présent")
            .field("discount", dtype="str", description="Remise ligne si présente")
        .structure("invoice")
            .field("date", dtype="str", description="Date de facture / invoice date")
            .field("due_date", dtype="str", description="Date d'échéance / due date")
            .field("payment_method", dtype="str", description="Mode de règlement / payment method")
            .field("payment_terms", dtype="str", description="Conditions de paiement (Net 30, 45 JFM...)")
            .field("total_excl_tax", dtype="str", description="Total HT / subtotal (excl. tax)")
            .field("tax_base", dtype="str", description="Base TVA / taxable base (assiette)")
            .field("tax_rate", dtype="str", description="Taux TVA global (si unique)")
            .field("tax_amount", dtype="str", description="Total TVA/VAT total")
            .field("total_incl_tax", dtype="str", description="Total TTC / amount due")
            .field("currency", dtype="str", description="Devise (EUR, USD...) ou symbole")
            .field("iban", dtype="str", description="IBAN si présent")
            .field("bic", dtype="str", description="BIC/SWIFT si présent")
    )


RE_SIREN = re.compile(r"\b\d{9}\b")
RE_SIRET = re.compile(r"\b\d{14}\b")
RE_VAT = re.compile(r"\b[A-Z]{2}[A-Z0-9]{2,12}\b")
RE_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b")
RE_BIC = re.compile(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b")

def normalize_iban(s: str) -> str:
    return re.sub(r"\s+", "", s).upper()

def is_valid_siren(s: str) -> bool:
    s = re.sub(r"\D", "", s or "")
    return len(s) == 9

def is_valid_siret(s: str) -> bool:
    s = re.sub(r"\D", "", s or "")
    return len(s) == 14

def is_valid_vat(s: str) -> bool:
    s = (s or "").strip().upper()
    return bool(re.match(r"^[A-Z]{2}[A-Z0-9]{2,12}$", s))

def is_valid_iban(s: str) -> bool:
    s = normalize_iban(s or "")
    return 15 <= len(s) <= 34 and bool(re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]+$", s))

def is_valid_bic(s: str) -> bool:
    s = (s or "").strip().upper()
    return bool(re.match(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$", s))

def pick_first_valid(cands: List[str], validator) -> Optional[str]:
    for c in cands:
        if c and validator(c):
            return c
    return None


SINGLE_STRUCTS = {"seller", "address_seller", "buyer", "address_buyer", "order", "customer", "document", "invoice"}

def _safe_get_first(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    arr = d.get(key) or []
    if isinstance(arr, list) and arr and isinstance(arr[0], dict):
        return arr[0]
    return {}

def _candidate_score(struct_name: str, cand: Dict[str, Any], section_name: str) -> float:
    non_null = sum(1 for v in cand.values() if v not in (None, "", []))
    bonus = 0.0
    if struct_name in ("seller", "address_seller") and section_name in ("header", "footer"):
        bonus += 1.0
    if struct_name in ("buyer", "address_buyer") and section_name in ("header", "meta"):
        bonus += 1.0
    if struct_name == "invoice" and section_name in ("totals", "meta", "footer"):
        bonus += 1.0
    if struct_name == "items" and section_name == "items":
        bonus += 2.0
    return non_null + bonus

def merge_extractions(extractions_by_section: Dict[str, Dict[str, Any]], sections_text: Dict[str, str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    # items: concat
    all_items: List[Dict[str, Any]] = []
    for sec, ext in extractions_by_section.items():
        for it in (ext.get("items") or []):
            if isinstance(it, dict) and any(v not in (None, "", []) for v in it.values()):
                all_items.append(it)
    merged["items"] = all_items

    # single structs: pick best
    for struct in SINGLE_STRUCTS:
        best, best_score = None, -1.0
        for sec, ext in extractions_by_section.items():
            cand = _safe_get_first(ext, struct)
            if not cand:
                continue
            sc = _candidate_score(struct, cand, sec)
            if sc > best_score:
                best_score = sc
                best = cand
        merged[struct] = [best] if best else []

    full = sections_text.get("all", "")

    seller = _safe_get_first(merged, "seller")
    if seller:
        if not seller.get("siret"):
            siret = pick_first_valid(RE_SIRET.findall(full), is_valid_siret)
            if siret:
                seller["siret"] = re.sub(r"\D", "", siret)
        if not seller.get("siren"):
            siren = pick_first_valid(RE_SIREN.findall(full), is_valid_siren)
            if siren:
                seller["siren"] = re.sub(r"\D", "", siren)
        if not seller.get("tax_id"):
            vats = [v for v in RE_VAT.findall(full) if is_valid_vat(v)]
            vats = sorted(vats, key=lambda x: 0 if x.startswith("FR") else 1)
            if vats:
                seller["tax_id"] = vats[0]

    inv = _safe_get_first(merged, "invoice")
    if inv:
        if not inv.get("iban") or str(inv.get("iban")).strip().upper() in ("RIB",):
            ibans = [normalize_iban(x) for x in RE_IBAN.findall(full) if is_valid_iban(x)]
            if ibans:
                inv["iban"] = ibans[0]
        if not inv.get("bic"):
            bics = [m.group(0).upper() for m in re.finditer(r"\b[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?\b", full) if is_valid_bic(m.group(0))]
            if bics:
                inv["bic"] = bics[0]

    # TVA/base coherence swap
    if inv:
        tax_amount = parse_money(str(inv.get("tax_amount") or ""))
        tax_base = parse_money(str(inv.get("tax_base") or ""))
        rate = parse_percent(str(inv.get("tax_rate") or ""))
        if rate is not None and tax_base is not None and tax_amount is not None:
            expected = round(tax_base * (rate / 100.0), 2)
            if abs(expected - tax_amount) > max(0.5, 0.03 * (tax_amount or 1)):
                expected_swapped = round(tax_amount * (rate / 100.0), 2)
                if abs(expected_swapped - tax_base) <= max(0.5, 0.03 * (tax_base or 1)):
                    inv["tax_base"], inv["tax_amount"] = inv.get("tax_amount"), inv.get("tax_base")

    # Dedup buyer if copied seller
    buyer = _safe_get_first(merged, "buyer")
    if seller and buyer:
        same = 0
        for k in ("siret", "email", "phone", "tax_id"):
            if seller.get(k) and buyer.get(k) and str(seller.get(k)).strip() == str(buyer.get(k)).strip():
                same += 1
        if same >= 2 and not buyer.get("name"):
            merged["buyer"] = []
            merged["address_buyer"] = []

    return merged


@dataclass
class PipelineResult:
    simplified_text: str
    extracted: Dict[str, Any]
    sections: Dict[str, str]
    mineru_blocks: Optional[List[Dict[str, Any]]] = None


def run_pipeline_from_base64(b64_str: str, threshold: float = 0.25, return_blocks: bool = False) -> PipelineResult:
    data = decode_base64_to_bytes(b64_str)
    image = normalize_image(bytes_to_first_page_image(data))

    mineru = load_mineru_client()
    blocks = mineru.two_step_extract(image)

    sections = blocks_to_sections(blocks)
    simplified_text = sections["all"].strip()

    extractor = GLiNER2.from_pretrained(GLINER2_MODEL_ID)
    schema = build_schema(extractor)

    extractions: Dict[str, Dict[str, Any]] = {}
    for sec in ("header", "meta", "items", "totals", "footer"):
        txt = sections.get(sec, "")
        if txt.strip():
            extractions[sec] = extractor.extract(txt, schema, threshold=threshold)
        else:
            extractions[sec] = {}

    merged = merge_extractions(extractions, sections)

    return PipelineResult(
        simplified_text=simplified_text,
        extracted=merged,
        sections=sections,
        mineru_blocks=blocks if return_blocks else None,
    )


def _read_base64_from_args_or_stdin(args) -> str:
    if args.base64:
        return args.base64
    if args.base64_file:
        return Path(args.base64_file).read_text(encoding="utf-8")
    import sys
    data = sys.stdin.read()
    if not data.strip():
        raise ValueError("Aucun base64 fourni (ni --base64, ni --base64-file, ni stdin). ")
    return data

def run(base64_str: str, threshold: float = 0.25, return_blocks: bool = False) -> Dict[str, Any]:
    res = run_pipeline_from_base64(base64_str, threshold=threshold, return_blocks=return_blocks)
    out = {"simplified_text": res.simplified_text, "extracted": res.extracted}
    if return_blocks:
        out["mineru_blocks"] = res.mineru_blocks
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base64", help="Base64 (PDF ou image). Supporte data:...;base64,...")
    ap.add_argument("--base64-file", help="Fichier texte contenant le base64 (PDF ou image)")
    ap.add_argument("--threshold", type=float, default=0.2, help="Seuil GLiNER2 (default: 0.2)")
    ap.add_argument("--with-blocks", action="store_true", help="Inclure les blocs MinerU bruts dans le JSON de sortie")
    ap.add_argument("--with-sections", action="store_true", help="Inclure le texte par section dans la sortie")
    args = ap.parse_args()

    b64 = _read_base64_from_args_or_stdin(args)
    res = run_pipeline_from_base64(b64, threshold=args.threshold, return_blocks=args.with_blocks)

    out = {"simplified_text": res.simplified_text, "extracted": res.extracted}
    if args.with_sections:
        out["sections"] = {k: v for k, v in res.sections.items() if k != "all"}
    if args.with_blocks:
        out["mineru_blocks"] = res.mineru_blocks

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()