# ============================================
# DOTA slicer com SAHI + geração de YOLO-OBB
# Salva em .../sliced/images e .../sliced/labels
# Visualização ao final (grid com polygons)
# Dependências: sahi pillow matplotlib
# pip install sahi pillow matplotlib
# ============================================

import os
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import sahi.slicing as sahi_slicing

# ========= CONFIGURE AQUI =========
IMAGES_DIR   = "/home/jonasvm/docker-images/dota_dataset_v15/kfold/fold_2/images/val"        # onde estão as imagens .png/.jpg
LABELS_DIR   = "/home/jonasvm/docker-images/dota_dataset_v15/kfold/fold_2/labels/val"        # onde estão os .txt (YOLO-OBB)
OUT_ROOT_DIR = "/home/jonasvm/docker-images/dota_dataset_v15/kfold/fold_2/sliced" # raiz da saída

SLICE_HEIGHT = 1024
SLICE_WIDTH  = 1024
OVERLAP_H    = 0.20
OVERLAP_W    = 0.20

EDGE_TOL_PX  = 1.0   # tolerância no centróide (px)
PREVIEW_MAX  = 12    # quantos tiles mostrar
# ===================================

OUT_IMAGES_DIR = os.path.join(OUT_ROOT_DIR, "images")
OUT_LABELS_DIR = os.path.join(OUT_ROOT_DIR, "labels")
os.makedirs(OUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUT_LABELS_DIR, exist_ok=True)

# ---------------- Utilitários ----------------
def _to_float_list(tokens):
    out = []
    for t in tokens:
        try: out.append(float(t))
        except: pass
    return out

def _clamp01(v):
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

def normalize_points(points_abs, w, h):
    return [(x/float(w), y/float(h)) for (x, y) in points_abs]

def clamp_point_to_rect(x, y, xmin, ymin, xmax, ymax):
    if x < xmin: x = xmin
    if x > xmax: x = xmax
    if y < ymin: y = ymin
    if y > ymax: y = ymax
    return x, y

def _centroid4(pts_abs):
    cx = (pts_abs[0][0] + pts_abs[1][0] + pts_abs[2][0] + pts_abs[3][0]) / 4.0
    cy = (pts_abs[0][1] + pts_abs[1][1] + pts_abs[2][1] + pts_abs[3][1]) / 4.0
    return cx, cy

def _inside(px, py, x0, y0, x1, y1, tol=0.0):
    return (x0 - tol) <= px <= (x1 + tol) and (y0 - tol) <= py <= (y1 + tol)

def read_yolo_obb_label(path_txt):
    """
    Linha: class x1 y1 x2 y2 x3 y3 x4 y4 (coords normalizadas)
    Retorna: [(cls_id, [(x1,y1)..(x4,y4)])]
    """
    anns = []
    if not os.path.exists(path_txt):
        return anns
    with open(path_txt, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            parts = line.replace(",", " ").replace("\t", " ").split()
            nums = _to_float_list(parts)
            if len(nums) < 9:
                continue
            cls_id = int(nums[0])
            coords = nums[1:9]
            poly = []
            for i in range(0, 8, 2):
                nx = _clamp01(coords[i])
                ny = _clamp01(coords[i+1])
                poly.append((nx, ny))
            if len(poly) == 4:
                anns.append((cls_id, poly))
    return anns

def _get_tile_origin_from_sahi(s):
    """
    Extrai (x0, y0, w, h) quando presente no dict do SAHI.
    """
    if isinstance(s, dict):
        if "bbox" in s and isinstance(s["bbox"], (list, tuple)) and len(s["bbox"]) >= 4:
            x0, y0, w, h = s["bbox"][:4]
            return float(x0), float(y0), int(w), int(h)
        if "starting_pixel" in s and isinstance(s["starting_pixel"], (list, tuple)) and len(s["starting_pixel"]) >= 2:
            x0, y0 = s["starting_pixel"][:2]
            im = Image.fromarray(s["image"])
            return float(x0), float(y0), im.size[0], im.size[1]
        if "origin" in s and isinstance(s["origin"], (list, tuple)) and len(s["origin"]) >= 2:
            x0, y0 = s["origin"][:2]
            im = Image.fromarray(s["image"])
            return float(x0), float(y0), im.size[0], im.size[1]
    return None

# ---------------- Pipeline ----------------
image_files = [f for f in os.listdir(IMAGES_DIR)
               if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff"))]
image_files.sort()

print(f"Imagens encontradas: {len(image_files)}")
for img_name in image_files:
    stem, _ = os.path.splitext(img_name)
    img_path = os.path.join(IMAGES_DIR, img_name)
    lbl_path = os.path.join(LABELS_DIR, f"{stem}.txt")

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    anns = read_yolo_obb_label(lbl_path)
    print(f"- {img_name}: {W}x{H}, labels={len(anns)}")

    slices = sahi_slicing.slice_image(
        image=img_path,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_H,
        overlap_width_ratio=OVERLAP_W,
        verbose=False
    )

    step_x = SLICE_WIDTH  - int(OVERLAP_W * SLICE_WIDTH)
    step_y = SLICE_HEIGHT - int(OVERLAP_H * SLICE_HEIGHT)
    num_cols = (W - 1) // step_x + 1

    for idx, s in enumerate(slices):
        tile_img = Image.fromarray(s["image"])
        tile_w, tile_h = tile_img.size

        origin = _get_tile_origin_from_sahi(s)
        if origin is not None:
            x0, y0, tile_w, tile_h = origin
            x1, y1 = x0 + tile_w, y0 + tile_h
            row = idx // max(1, num_cols)
            col = idx %  max(1, num_cols)
        else:
            row = idx // max(1, num_cols)
            col = idx %  max(1, num_cols)
            x0 = min(col * step_x, W - tile_w)
            y0 = min(row * step_y, H - tile_h)
            x1, y1 = x0 + tile_w, y0 + tile_h

        # Salva tile (IMAGEM)
        out_img_name = f"{stem}_r{row:03d}_c{col:03d}.png"
        tile_img.save(os.path.join(OUT_IMAGES_DIR, out_img_name))

        # Salva tile (LABEL)
        out_lbl_name = f"{stem}_r{row:03d}_c{col:03d}.txt"
        out_lbl_path = os.path.join(OUT_LABELS_DIR, out_lbl_name)
        lines_out = []

        for (cls_id, poly_norm) in anns:
            poly_abs = [(px * W, py * H) for (px, py) in poly_norm]

            # critério: centróide do OBB dentro do tile (tolerância)
            cx, cy = _centroid4(poly_abs)
            if not _inside(cx, cy, x0, y0, x1, y1, tol=EDGE_TOL_PX):
                continue

            # desloca para o tile + clamp bordas
            poly_tile_abs = []
            for (ax, ay) in poly_abs:
                tx, ty = ax - x0, ay - y0
                tx, ty = clamp_point_to_rect(tx, ty, 0.0, 0.0, float(tile_w), float(tile_h))
                poly_tile_abs.append((tx, ty))

            # normaliza no tile
            poly_tile_norm = normalize_points(poly_tile_abs, tile_w, tile_h)

            vals = []
            for (nx, ny) in poly_tile_norm:
                vals.append(f"{nx:.6f}")
                vals.append(f"{ny:.6f}")
            lines_out.append(f"{cls_id} " + " ".join(vals[:8]))

        with open(out_lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out))

print("✅ Fatiamento concluído.")
print("  Imagens ->", OUT_IMAGES_DIR)
print("  Labels  ->", OUT_LABELS_DIR)

# ---------------- Visualização (preview) ----------------
tile_imgs = [f for f in os.listdir(OUT_IMAGES_DIR)
             if f.lower().endswith((".png",".jpg",".jpeg"))]
tile_imgs.sort()
tile_imgs = tile_imgs[:PREVIEW_MAX]

if tile_imgs:
    cols = 4
    rows = math.ceil(len(tile_imgs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
    if rows == 1:
        axes = [axes] if cols > 1 else [[axes]]

    for i, tname in enumerate(tile_imgs):
        r = i // cols
        c = i % cols
        ax = axes[r][c] if rows > 1 else axes[c]

        tpath = os.path.join(OUT_IMAGES_DIR, tname)
        lpath = os.path.join(OUT_LABELS_DIR, os.path.splitext(tname)[0] + ".txt")

        im = Image.open(tpath).convert("RGBA")
        draw = ImageDraw.Draw(im, 'RGBA')

        if os.path.exists(lpath):
            with open(lpath, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    ps = ln.strip().split()
                    if len(ps) >= 9:
                        vals = _to_float_list(ps[:9])
                        if len(vals) == 9:
                            coords = vals[1:9]
                            pts = [(coords[0]*im.width, coords[1]*im.height),
                                   (coords[2]*im.width, coords[3]*im.height),
                                   (coords[4]*im.width, coords[5]*im.height),
                                   (coords[6]*im.width, coords[7]*im.height)]
                            draw.polygon(pts, outline="red", width=2)

        ax.imshow(im)
        ax.set_title(tname, fontsize=9)
        ax.axis("off")

    # apaga eixos extras
    total = len(tile_imgs)
    for j in range(total, rows*cols):
        r = j // cols
        c = j % cols
        if rows > 1:
            axes[r][c].axis("off")
        else:
            axes[c].axis("off")

    plt.tight_layout()
    plt.show()
else:
    print("Nenhum tile para visualizar (verifique os diretórios de saída).")
