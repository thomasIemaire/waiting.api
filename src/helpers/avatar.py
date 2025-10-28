# pip install pillow
import os
from PIL import Image, ImageChops, ImageFilter, ImageEnhance
import colorsys, hashlib, math, random
from typing import Tuple, List

RGB = Tuple[int, int, int]

# ---------- utils ----------
def _rng(seed: str) -> random.Random:
    return random.Random(hashlib.sha256(seed.encode()).digest())

def _hls_to_rgb(h, l, s) -> RGB:
    r, g, b = colorsys.hls_to_rgb(h % 1.0, max(0, min(1, l)), max(0, min(1, s)))
    return int(r * 255), int(g * 255), int(b * 255)

def _mix_rgb(a: RGB, b: RGB, t: float) -> RGB:
    return (int(a[0]*(1-t)+b[0]*t), int(a[1]*(1-t)+b[1]*t), int(a[2]*(1-t)+b[2]*t))

def _towards_white(c: RGB, amt: float) -> RGB:
    return _mix_rgb(c, (255,255,255), amt)

def _punch(img, sat=1.10, contrast=1.05, brightness=1.02):
    img = ImageEnhance.Color(img).enhance(sat)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return ImageEnhance.Brightness(img).enhance(brightness)

# ---------- palette TRIADIQUE (contrastée mais douce) ----------
def palette_from_string(s: str, variant: int = 0) -> List[RGB]:
    rnd = _rng(f"{s}|{variant}|pal-v4")
    base_h = rnd.random()
    offs = [0.0, 1/3, 2/3]
    offs = [o + rnd.uniform(-0.015, 0.015) for o in offs]  # ±5°

    # ↓ un peu plus sombre & plus saturé
    L = 0.58 + 0.05 * rnd.random()   # 0.58..0.63 (avant 0.62..0.68)
    S = 0.70 + 0.20 * rnd.random()   # 0.70..0.90 (avant 0.58..0.74)

    cols = [_hls_to_rgb(base_h + o, L, S) for o in offs]
    # ↓ moins de “pastel”
    cols = [_towards_white(c, 0.05 + 0.07 * rnd.random()) for c in cols]
    rnd.shuffle(cols)
    return cols

# ---------- dégradé diagonal (fond clair et agréable) ----------
def diagonal_gradient(size: int, c0: RGB, c1: RGB) -> Image.Image:
    w = h = size
    grad = Image.new("RGB", (w, h))
    px = grad.load()
    for y in range(h):
        for x in range(w):
            t = (x + y) / (w + h - 2)  # 0 en haut-gauche → 1 en bas-droit
            px[x, y] = _mix_rgb(c0, c1, t)
    return grad

# --- masque radial doux (centre discret, pas noir) ---
def radial_mask(size, center, radius, *, edge=2.1, center_power=0.9, strength=0.85):
    m = Image.new("L", (size, size), 0)
    px = m.load()
    cx, cy = center
    r = float(radius)
    inv = 1.0 / r
    for y in range(size):
        dy = y - cy
        for x in range(size):
            dx = x - cx
            d = math.hypot(dx, dy)
            if d >= r:
                continue
            t = d * inv                    # 0 au centre → 1 au bord
            w = (1 - t) ** edge            # fade-out bord
            w *= max(t, 0.0001) ** center_power  # léger creux au centre
            a = int(255 * strength * w)
            if a > px[x, y]:
                px[x, y] = a
    return m

# ---------- centres près des coins ----------
def corner_centers(size: int, s: str, variant: int) -> List[Tuple[int,int]]:
    rnd = _rng(f"{s}|{variant}|geom-v3")
    w = h = size
    margin = int(size * 0.15)
    jitter = int(size * 0.04)
    TL = (margin + rnd.randint(-jitter, jitter),               margin + rnd.randint(-jitter, jitter))
    TR = (w - margin + rnd.randint(-jitter, jitter),           margin + rnd.randint(-jitter, jitter))
    BL = (margin + rnd.randint(-jitter, jitter),               h - margin + rnd.randint(-jitter, jitter))
    BR = (w - margin + rnd.randint(-jitter, jitter),           h - margin + rnd.randint(-jitter, jitter))
    triads = [(TL, TR, BL), (TR, BR, TL), (BR, BL, TR), (BL, TL, BR)]
    return triads[rnd.randrange(4)]

# ---------- avatar final ----------
def generate_avatar(string: str, size: int = 800, *, variant: int = 0) -> Image.Image:
    w = h = size
    c1, c2, c3 = palette_from_string(string, variant)

    # 1) fond clair diagonal entre deux couleurs éclaircies
    bg0 = _towards_white(c1, 0.32)
    bg1 = _towards_white(c2, 0.32)
    base = diagonal_gradient(size, bg0, bg1)

    # 2) 3 cercles vers les coins (Screen + masque)
    centers = corner_centers(size, string, variant)
    radius = int(size * 0.86)

    # opacités différentes pour éviter la symétrie
    strengths = (0.97, 0.90, 0.94)
    for color, center, k in zip((c1, c2, c3), centers, strengths):
        solid = Image.new("RGB", (w, h), color)
        mask = radial_mask(size, center, radius, edge=2.0, center_power=1.0, strength=1.0)
        screen_mix = ImageChops.screen(base, solid)
        # on applique l'éclaircissement avec un alpha global k, limité au masque
        pre = Image.blend(base, screen_mix, k)
        base = Image.composite(pre, base, mask)

    # 3) fondu très léger (évite banding)
    return _punch(base.filter(ImageFilter.GaussianBlur(radius=0.3)))

def save_avatar(img: Image.Image, folder: str, filename: str) -> None:
    os.makedirs(folder, exist_ok=True)
    img.save(os.path.join(folder, filename), "PNG")
    img.close()
