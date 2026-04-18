#!/usr/bin/env python3
"""
Prépare un sous-ensemble CryoVirusDB / CryoVirusDB_Lite pour le benchmarking.

Fonctions:
- convertit des micrographies .mrc en .png 8 bits si nécessaire
- copie les images .jpg/.png si le dataset Lite est déjà en image standard
- convertit les coordonnées CSV des particules en annotations COCO (boîtes centrées)
- crée les splits train/val/test
- génère un dataset.yaml pour YOLO / RT-DETR (optionnel pour fine-tuning)

Exemple:
python prepare_cryovirusdb.py \
  --raw-dir data/raw/11060 \
  --output-dir data/processed/11060 \
  --diameter 516
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

try:
    import mrcfile
except ImportError:
    mrcfile = None


VALID_IMG_EXTS = {".mrc", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Préparation CryoVirusDB -> COCO")
    parser.add_argument("--raw-dir", type=str, required=True, help="Répertoire brut extrait, ex: data/raw/11060")
    parser.add_argument("--output-dir", type=str, required=True, help="Répertoire de sortie")
    parser.add_argument("--diameter", type=float, required=True, help="Diamètre moyen de particule en pixels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0, help="Limiter le nombre d'images (0 = tout)")
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    return parser.parse_args()


def normalize_to_uint8(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.percentile(arr, p_low)
    hi = np.percentile(arr, p_high)
    if hi <= lo:
        hi = lo + 1e-6
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def read_mrc_as_pil(path: Path, p_low: float, p_high: float) -> Image.Image:
    if mrcfile is None:
        raise RuntimeError("mrcfile n'est pas installé. Lancez: pip install mrcfile")
    with mrcfile.open(path, permissive=True) as mrc:
        data = mrc.data
        if data.ndim == 3:
            data = data[0]
        img = normalize_to_uint8(data, p_low, p_high)
    return Image.fromarray(img)


def load_image(path: Path, p_low: float, p_high: float) -> Tuple[Image.Image, bool]:
    if path.suffix.lower() == ".mrc":
        return read_mrc_as_pil(path, p_low, p_high), True
    img = Image.open(path).convert("L")
    return img, False


def infer_xy_columns(df: pd.DataFrame) -> Tuple[str, str]:
    candidates_x = ["x", "X", "cx", "center_x", "col", "x_coord", "coord_x"]
    candidates_y = ["y", "Y", "cy", "center_y", "row", "y_coord", "coord_y"]

    x_col = next((c for c in candidates_x if c in df.columns), None)
    y_col = next((c for c in candidates_y if c in df.columns), None)

    if x_col and y_col:
        return x_col, y_col

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    raise ValueError(f"Impossible d'inférer les colonnes x/y dans {list(df.columns)}")


def load_particle_centers(csv_path: Path) -> List[Tuple[float, float]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    x_col, y_col = infer_xy_columns(df)
    centers = []
    for _, row in df.iterrows():
        try:
            x = float(row[x_col])
            y = float(row[y_col])
            centers.append((x, y))
        except Exception:
            continue
    return centers


def center_to_coco_bbox(x: float, y: float, diameter: float, width: int, height: int) -> Tuple[float, float, float, float]:
    half = diameter / 2.0
    x1 = max(0.0, x - half)
    y1 = max(0.0, y - half)
    x2 = min(float(width - 1), x + half)
    y2 = min(float(height - 1), y + half)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return x1, y1, w, h


def split_items(items: List[Path], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Path]]:
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def empty_coco() -> Dict:
    return {
        "info": {"description": "CryoVirusDB converted to COCO"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "virus", "supercategory": "virus"}],
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    micrographs_dir = raw_dir / "micrographs"
    coords_dir = raw_dir / "ground_truth" / "particle_coordinates"

    if not micrographs_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {micrographs_dir}")
    if not coords_dir.exists():
        raise FileNotFoundError(f"Répertoire introuvable: {coords_dir}")

    output_images_root = output_dir / "images"
    output_ann_root = output_dir / "annotations"
    temp_all = output_dir / "_all_images"
    for p in [output_images_root, output_ann_root, temp_all]:
        p.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in micrographs_dir.iterdir() if p.suffix.lower() in VALID_IMG_EXTS])
    if args.limit and args.limit > 0:
        image_files = image_files[: args.limit]

    print(f"[INFO] Images détectées: {len(image_files)}")
    split_map = split_items(image_files, args.train_ratio, args.val_ratio, args.seed)

    coco_by_split = {"train": empty_coco(), "val": empty_coco(), "test": empty_coco()}
    ann_id = 1
    img_id = 1

    split_lookup = {}
    for split_name, files in split_map.items():
        for f in files:
            split_lookup[f.name] = split_name

    for src_path in image_files:
        split_name = split_lookup[src_path.name]
        dst_split_dir = output_images_root / split_name
        dst_split_dir.mkdir(parents=True, exist_ok=True)

        pil_img, converted_from_mrc = load_image(src_path, args.percentile_low, args.percentile_high)
        stem = src_path.stem
        dst_name = f"{stem}.png" if converted_from_mrc else src_path.name
        dst_path = dst_split_dir / dst_name

        if converted_from_mrc:
            pil_img.save(dst_path)
        else:
            shutil.copy2(src_path, dst_path)
            pil_img = Image.open(dst_path).convert("L")

        width, height = pil_img.size
        rel_file_name = f"images/{split_name}/{dst_name}"

        coco_by_split[split_name]["images"].append(
            {
                "id": img_id,
                "width": width,
                "height": height,
                "file_name": rel_file_name,
            }
        )

        csv_path = coords_dir / f"{stem}.csv"
        centers = load_particle_centers(csv_path)
        for x, y in centers:
            x1, y1, w, h = center_to_coco_bbox(x, y, args.diameter, width, height)
            coco_by_split[split_name]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        img_id += 1

    for split_name, coco_dict in coco_by_split.items():
        out_json = output_ann_root / f"instances_{split_name}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, indent=2)
        print(f"[OK] JSON COCO écrit: {out_json}")

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: virus",
            ]
        ),
        encoding="utf-8",
    )

    metadata = {
        "raw_dir": str(raw_dir.resolve()),
        "diameter_px": args.diameter,
        "splits": {k: len(v) for k, v in split_map.items()},
        "note": "Les boîtes sont dérivées des coordonnées centrales et d'un diamètre moyen fourni par CryoVirusDB.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] dataset.yaml écrit: {dataset_yaml}")
    print("[FINI] Préparation terminée.")


if __name__ == "__main__":
    main()
