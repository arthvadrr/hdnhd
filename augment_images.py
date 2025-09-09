"""
Ducky credits.

Bulk augment images by class folder.

Expected layout:

input/
  walmart/
  not_walmart/

output/
  walmart/
  not_walmart/

Usage:
  python augment_images.py --input data/train --output data/train --per_image 8 --size 160
  python augment_images.py --input data/val   --output data/val   --per_image 0 --size 160

Note:
  Augment train. Keep val clean.
"""
import argparse, os, random, math, hashlib
from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np

R = random.Random(1337)

def clamp01(x): return max(0.0, min(1.0, x))

def rand_perspective(img):
  w, h = img.size
  dx = R.uniform(0.00, 0.08) * w
  dy = R.uniform(0.00, 0.08) * h
  src = [(0,0),(w,0),(w,h),(0,h)]
  dst = [(R.uniform(-dx,dx), R.uniform(-dy,dy)),
         (w+R.uniform(-dx,dx), R.uniform(-dy,dy)),
         (w+R.uniform(-dx,dx), h+R.uniform(-dy,dy)),
         (R.uniform(-dx,dx), h+R.uniform(-dy,dy))]
  coeffs = _find_perspective_coeffs(src, dst)
  return img.transform((w,h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

def _find_perspective_coeffs(pa, pb):
  # pa and pb are lists of 4 tuples (x,y)
  import numpy as np
  A = []
  B = []
  for p1, p2 in zip(pa, pb):
    x, y = p1
    u, v = p2
    A.append([x, y, 1, 0, 0, 0, -u*x, -u*y])
    A.append([0, 0, 0, x, y, 1, -v*x, -v*y])
    B.append(u)
    B.append(v)
  A = np.array(A)
  B = np.array(B)
  res = np.linalg.lstsq(A, B, rcond=None)[0]
  return res

def random_aug(im: Image.Image) -> Image.Image:
  im = im.convert("RGB")

  # random crop then resize back
  w, h = im.size
  scale = R.uniform(0.85, 1.0)
  nw, nh = int(w*scale), int(h*scale)
  if scale < 1.0:
    x0 = R.randint(0, w - nw)
    y0 = R.randint(0, h - nh)
    im = im.crop((x0, y0, x0+nw, y0+nh)).resize((w, h), Image.BICUBIC)

  # rotate small angle
  angle = R.uniform(-15, 15)
  im = im.rotate(angle, resample=Image.BICUBIC, expand=False)

  # flips sometimes
  if R.random() < 0.20:
    im = ImageOps.mirror(im)
  if R.random() < 0.10:
    im = ImageOps.flip(im)

  # perspective warp sometimes
  if R.random() < 0.35:
    im = rand_perspective(im)

  # brightness, contrast, color, sharpness
  im = ImageEnhance.Brightness(im).enhance(R.uniform(0.75, 1.25))
  im = ImageEnhance.Contrast(im).enhance(R.uniform(0.75, 1.25))
  im = ImageEnhance.Color(im).enhance(R.uniform(0.7, 1.3))
  im = ImageEnhance.Sharpness(im).enhance(R.uniform(0.7, 1.4))

  # blur or noise
  if R.random() < 0.35:
    im = im.filter(ImageFilter.GaussianBlur(radius=R.uniform(0.0, 1.2)))
  if R.random() < 0.35:
    arr = np.array(im).astype(np.float32)
    noise = np.random.normal(0, R.uniform(2, 10), arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGB")

  return im

def hash_name(p: Path):
  h = hashlib.sha1(str(p).encode()).hexdigest()[:8]
  return h

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--input", required=True, help="input root with class folders")
  ap.add_argument("--output", required=True, help="output root")
  ap.add_argument("--per_image", type=int, default=10, help="augmented images per source")
  ap.add_argument("--size", type=int, default=160, help="final square size")
  args = ap.parse_args()

  in_root = Path(args.input)
  out_root = Path(args.output)
  out_root.mkdir(parents=True, exist_ok=True)

  classes = [d for d in in_root.iterdir() if d.is_dir()]
  if not classes:
    print("No class folders in", in_root)
    return

  for cls in classes:
    files = [p for p in cls.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp"}]
    if not files:
      print("Skip empty", cls)
      continue
    dst = out_root / cls.name
    dst.mkdir(parents=True, exist_ok=True)
    for p in files:
      try:
        im = Image.open(p)
      except Exception as e:
        print("Bad image", p, e)
        continue

      base = im.convert("RGB").resize((args.size, args.size), Image.BICUBIC)
      base.save(dst / f"{p.stem}_{hash_name(p)}_orig.jpg", quality=92)

      for i in range(args.per_image):
        aug = random_aug(im).resize((args.size, args.size), Image.BICUBIC)
        aug.save(dst / f"{p.stem}_{hash_name(p)}_aug{i:02d}.jpg", quality=90)

  print("Done. Wrote to", out_root)

if __name__ == "__main__":
  main()