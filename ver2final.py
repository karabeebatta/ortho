import argparse, os, io, math
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
import cv2

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def pil_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def colorfulness_rgb(img):
    # Hasler–Süsstrunk colorfulness metric (cheap proxy)
    (B, G, R) = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    rg = np.abs(R - G)
    yb = np.abs(0.5*(R + G) - B)
    std_rg, std_yb = rg.std(), yb.std()
    mean_rg, mean_yb = rg.mean(), yb.mean()
    return math.sqrt(std_rg**2 + std_yb**2) + 0.3*math.sqrt(mean_rg**2 + mean_yb**2)

def laplacian_var(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def grayscale_fraction(img_bgr):
    # fraction of pixels close to gray (R≈G≈B)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    diffs = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return (diffs <= 8).mean()

def dark_fraction(img_gray):
    return (img_gray < 40).mean()

def is_xray_like(img_bgr):
    # Quick heuristics tuned for radiographs (tweak as needed)
    h, w = img_bgr.shape[:2]
    if min(h, w) < 256:  # skip tiny icons
        return False, 0.0, "too_small"

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cf = colorfulness_rgb(img_bgr)
    gf = grayscale_fraction(img_bgr)
    df = dark_fraction(img_gray)
    lv = laplacian_var(img_gray)
    mean = img_gray.mean()
    std = img_gray.std()

    # Score: low colorfulness + grayscaley + decent edges + not overly bright
    score = (1.0 - min(cf/20.0, 1.0)) * 0.4 \
          + min(gf, 1.0) * 0.25 \
          + min(lv/200.0, 1.0) * 0.15 \
          + min(df/0.25, 1.0) * 0.10 \
          + (1.0 - min(abs(mean-100)/100.0, 1.0)) * 0.10

    flags = []
    if cf < 5: flags.append("low_color")
    if gf > 0.85: flags.append("mostly_gray")
    if df > 0.08: flags.append("dark_bg")
    if lv > 60: flags.append("edges")
    if std > 40: flags.append("high_contrast")

    return score > 0.55, float(score), ",".join(flags)

def extract_embedded_images(doc, page_idx, out_dir, rows):
    page = doc[page_idx]
    for i, xref in enumerate(page.get_images(full=True)):
        pix = fitz.Pixmap(doc, xref[0])
        if pix.alpha:  # remove alpha to keep OpenCV happy
            pix = fitz.Pixmap(pix, 0)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_cv = pil_to_cv(img)
        ok, score, flags = is_xray_like(img_cv)
        tag = "xray" if ok else "nonxray"
        fname = f"p{page_idx+1:04d}_img{i:02d}_{tag}.png"
        path = os.path.join(out_dir, fname)
        cv2.imwrite(path, img_cv)
        rows.append({
            "page": page_idx+1, "type": "embedded", "index": i,
            "saved_as": fname, "is_xray": ok, "score": score, "flags": flags,
            "w": img_cv.shape[1], "h": img_cv.shape[0]
        })

def raster_and_find_rects(doc, page_idx, out_dir, dpi, rows):
    # Render whole page; find large dark/gray rectangles (typical radiograph plates)
    page = doc[page_idx]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    cv = pil_to_cv(img)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 51, 5)
    # Remove small specks
    kernel = np.ones((5,5), np.uint8)
    clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    candidates = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        if area < 0.06*W*H:  # keep only big blocks
            continue
        aspect = max(w/h, h/w)
        if aspect > 3.5:  # X-rays are rarely ultra-panoramic in books
            continue
        crop = cv[y:y+h, x:x+w]
        ok, score, flags = is_xray_like(crop)
        if ok:
            candidates.append((score, (x,y,w,h), flags))

    candidates.sort(reverse=True, key=lambda t:t[0])
    for j, (score, (x,y,w,h), flags) in enumerate(candidates[:3]):  # save top few
        crop = cv[y:y+h, x:x+w]
        fname = f"p{page_idx+1:04d}_crop{j:02d}_xray.png"
        cv2.imwrite(os.path.join(out_dir, fname), crop)
        rows.append({
            "page": page_idx+1, "type": "raster_crop", "index": j,
            "saved_as": fname, "is_xray": True, "score": float(score), "flags": flags,
            "w": w, "h": h
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=str)
    ap.add_argument("--out", type=str, default="out_xrays")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--pages", type=str, default="")  # e.g., "1-50,120-"
    args = ap.parse_args()

    ensure_dir(args.out)
    doc = fitz.open(args.pdf)

    # Page selection
    def parse_ranges(spec, total):
        if not spec: return range(total)
        idxs = set()
        for part in spec.split(","):
            if "-" in part:
                a,b = part.split("-")
                a = int(a)-1 if a else 0
                b = int(b)-1 if b else total-1
                idxs.update(range(max(0,a), min(total, b+1)))
            else:
                idxs.add(int(part)-1)
        return sorted([i for i in idxs if 0 <= i < total])

    pages = parse_ranges(args.pages, len(doc))
    rows = []
    for pi in pages:
        try:
            extract_embedded_images(doc, pi, args.out, rows)
        except Exception:
            # Some PDFs have odd encodings; skip embedded extraction errors
            pass
        # Even if embedded images were found, also try raster crops (covers vector/flattened layouts)
        try:
            raster_and_find_rects(doc, pi, args.out, args.dpi, rows)
        except Exception:
            pass

    report = os.path.join(args.out, "report.csv")
    pd.DataFrame(rows).to_csv(report, index=False)
    print(f"Done. Saved images and {report}")

if __name__ == "__main__":
    main()
