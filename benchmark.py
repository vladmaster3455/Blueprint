#!/usr/bin/env python3
"""
Benchmarking object detection sur dataset COCO atypique.

Modèles pris en charge:
- YOLO (Ultralytics)
- RT-DETR (Ultralytics)
- DINO (MMDetection)

Mesures:
- mAP@0.50:0.95 et mAP@0.50 via pycocotools
- latence moyenne et FPS
- mémoire max / moyenne (CPU ou GPU)
- export CSV, JSON et graphiques PNG

Exemple:
python benchmark.py \
  --coco-json data/processed/11060/annotations/instances_test.json \
  --image-root data/processed/11060 \
  --device cuda:0 \
  --weights-yolo yolo11n.pt \
  --weights-rtdetr rtdetr-l.pt \
  --dino-config configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
  --weights-dino weights/dino_best.pth \
  --outdir outputs/bench_11060
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from ultralytics import RTDETR, YOLO

MMDET_IMPORT_ERROR: Optional[Exception] = None
try:
    from mmdet.apis import DetInferencer, inference_detector, init_detector
except Exception as e:
    MMDET_IMPORT_ERROR = e
    DetInferencer = None
    inference_detector = None
    init_detector = None


@dataclass
class ModelStats:
    model: str
    fps: float
    avg_latency_ms: float
    map_50_95: float
    map_50: float
    avg_memory_mb: float
    max_memory_mb: float
    num_images: int
    num_predictions: int
    pred_json: str
    status: str = "ok"
    note: str = ""


class BaseWrapper:
    name: str

    def predict(self, image_path: str, conf: float) -> List[Dict]:
        raise NotImplementedError


class UltralyticsWrapper(BaseWrapper):
    def __init__(self, name: str, weights: str, device: str, imgsz: int):
        self.name = name
        self.device = device
        self.imgsz = imgsz
        self.model = YOLO(weights) if name.lower() == "yolo" else RTDETR(weights)

    def predict(self, image_path: str, conf: float) -> List[Dict]:
        results = self.model.predict(
            source=image_path,
            imgsz=self.imgsz,
            conf=conf,
            device=self.device,
            verbose=False,
            save=False,
        )
        result = results[0]
        preds: List[Dict] = []
        if result.boxes is None or len(result.boxes) == 0:
            return preds

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy() if result.boxes.cls is not None else np.zeros(len(scores))

        for box, score, cls_id in zip(xyxy, scores, classes):
            x1, y1, x2, y2 = [float(v) for v in box]
            preds.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "score": float(score),
                    "label": int(cls_id),
                }
            )
        return preds


class DinoWrapper(BaseWrapper):
    def __init__(self, config: str, checkpoint: str, device: str, model_name: str):
        if (init_detector is None or inference_detector is None) and DetInferencer is None:
            details = f" Cause d'import: {MMDET_IMPORT_ERROR}" if MMDET_IMPORT_ERROR is not None else ""
            raise RuntimeError(
                "MMDetection n'est pas disponible. Vérifiez les versions compatibles "
                "(ex: mmcv-lite>=2.1.0,<2.2.0 avec mmdet>=3.3.0)."
                f"{details}"
            )
        self.name = "DINO"
        self.device = device
        self.use_inferencer = not bool(config)
        if self.use_inferencer:
            if DetInferencer is None:
                raise RuntimeError("DetInferencer indisponible dans cette installation MMDetection")
            self.inferencer = DetInferencer(model=model_name, weights=checkpoint or None, device=device)
            self.model = None
        else:
            self.model = init_detector(config, checkpoint or None, device=device)
            self.inferencer = None

    def predict(self, image_path: str, conf: float) -> List[Dict]:
        preds: List[Dict] = []
        if self.use_inferencer:
            outputs = self.inferencer(image_path, no_save_vis=True, no_save_pred=True)
            prediction = outputs["predictions"][0]
            bboxes = prediction.get("bboxes", [])
            scores = prediction.get("scores", [])
            labels = prediction.get("labels", [])
        else:
            result = inference_detector(self.model, image_path)
            instances = result.pred_instances
            if instances is None or len(instances) == 0:
                return preds
            bboxes = instances.bboxes.detach().cpu().numpy().tolist()
            scores = instances.scores.detach().cpu().numpy().tolist()
            labels = instances.labels.detach().cpu().numpy().tolist()

        for bbox, score, label in zip(bboxes, scores, labels):
            if float(score) < conf:
                continue
            x1, y1, x2, y2 = [float(v) for v in bbox]
            preds.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "score": float(score),
                    "label": int(label),
                }
            )
        return preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO vs RT-DETR vs DINO")
    parser.add_argument("--coco-json", type=str, required=True, help="Fichier COCO GT, ex: instances_test.json")
    parser.add_argument("--image-root", type=str, required=True, help="Racine contenant les images référencées dans le COCO JSON")
    parser.add_argument("--outdir", type=str, default="outputs/benchmark_run")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001, help="Seuil minimal gardé pour l'évaluation mAP")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=0, help="0 = toutes les images")
    parser.add_argument("--preview-images", type=int, default=6)
    parser.add_argument("--weights-yolo", type=str, default="yolo11n.pt")
    parser.add_argument("--weights-rtdetr", type=str, default="rtdetr-l.pt")
    parser.add_argument("--dino-config", type=str, default="", help="Chemin config locale DINO. Laissez vide pour utiliser le nom de modèle mmdet.")
    parser.add_argument("--weights-dino", type=str, default="", help="Checkpoint DINO custom optionnel")
    parser.add_argument("--dino-model", type=str, default="dino-4scale_r50_8xb2-12e_coco", help="Nom modèle MMDetection si --dino-config n'est pas fourni")
    parser.add_argument(
        "--class-agnostic",
        action="store_true",
        help="Force toutes les prédictions dans la catégorie virus (utile pour un benchmark de localisation sur domaine atypique)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo", "rtdetr", "dino"],
        choices=["yolo", "rtdetr", "dino"],
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def xyxy_to_xywh(box: List[float], width: Optional[int] = None, height: Optional[int] = None) -> List[float]:
    x1, y1, x2, y2 = box
    if width is not None:
        x1 = max(0.0, min(x1, width - 1))
        x2 = max(0.0, min(x2, width - 1))
    if height is not None:
        y1 = max(0.0, min(y1, height - 1))
        y2 = max(0.0, min(y2, height - 1))
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return [float(x1), float(y1), float(w), float(h)]


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_memory_mb(device: str, baseline_rss_bytes: int) -> float:
    if device.startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    rss_now = psutil.Process().memory_info().rss
    return float(max(0, rss_now - baseline_rss_bytes) / (1024 ** 2))


def evaluate_coco(gt_json: str, pred_json: str) -> Dict[str, float]:
    coco_gt = COCO(gt_json)
    with open(pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)
    if len(preds) == 0:
        return {"map_50_95": 0.0, "map_50": 0.0}

    coco_dt = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map_50_95": float(coco_eval.stats[0]),
        "map_50": float(coco_eval.stats[1]),
    }


def draw_preview(image_path: Path, preds: List[Dict], save_path: Path, max_boxes: int = 50) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    color = (7, 89, 133)
    for pred in preds[:max_boxes]:
        x1, y1, x2, y2 = pred["bbox_xyxy"]
        score = pred["score"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 4, y1 + 4), f"{score:.2f}", fill=(255, 255, 255))
    img.save(save_path)


def create_charts(df: pd.DataFrame, charts_dir: Path) -> None:
    ensure_dir(charts_dir)
    plt.style.use("seaborn-v0_8-whitegrid")

    chart_specs = [
        ("map_50_95", "mAP@0.50:0.95", "map_50_95.png", "#0b5fa5"),
        ("fps", "FPS", "fps.png", "#00a6a6"),
        ("max_memory_mb", "Mémoire max (MB)", "memory.png", "#c27c0e"),
        ("avg_latency_ms", "Latence moyenne (ms)", "latency.png", "#7b2cbf"),
    ]

    for column, title, file_name, color in chart_specs:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(df["model"], df[column], color=color)
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_ylabel(title)
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(df[column].tolist()):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
        fig.tight_layout()
        fig.savefig(charts_dir / file_name, dpi=180)
        plt.close(fig)


def benchmark_model(
    wrapper: BaseWrapper,
    coco: COCO,
    image_root: Path,
    outdir: Path,
    device: str,
    conf: float,
    warmup: int,
    max_images: int,
    preview_images: int,
    class_agnostic: bool,
    gt_json: str,
) -> ModelStats:
    images = list(coco.imgs.values())
    images = sorted(images, key=lambda x: x["id"])
    if max_images and max_images > 0:
        images = images[:max_images]

    pred_dir = outdir / "predictions"
    preview_dir = outdir / "previews" / wrapper.name.lower()
    ensure_dir(pred_dir)
    ensure_dir(preview_dir)

    detections = []
    latencies = []
    memories = []
    preview_count = 0
    total_measured = 0

    warmup_imgs = images[: min(warmup, len(images))]
    for img_info in warmup_imgs:
        img_path = image_root / img_info["file_name"]
        _ = wrapper.predict(str(img_path), conf)
        synchronize(device)

    for img_info in tqdm(images, desc=f"Benchmark {wrapper.name}"):
        img_path = image_root / img_info["file_name"]
        baseline_rss = psutil.Process().memory_info().rss

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        synchronize(device)
        t0 = time.perf_counter()
        preds = wrapper.predict(str(img_path), conf)
        synchronize(device)
        elapsed = time.perf_counter() - t0

        mem_mb = measure_memory_mb(device, baseline_rss)
        latencies.append(elapsed)
        memories.append(mem_mb)
        total_measured += 1

        if preview_count < preview_images:
            draw_preview(img_path, preds, preview_dir / f"preview_{preview_count + 1}.png")
            preview_count += 1

        width = img_info.get("width")
        height = img_info.get("height")
        for pred in preds:
            category_id = 1 if class_agnostic else int(pred["label"]) + 1
            x, y, w, h = xyxy_to_xywh(pred["bbox_xyxy"], width, height)
            detections.append(
                {
                    "image_id": int(img_info["id"]),
                    "category_id": int(category_id),
                    "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                    "score": round(float(pred["score"]), 6),
                }
            )

    pred_json = pred_dir / f"{wrapper.name.lower()}_detections.json"
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)

    metrics = evaluate_coco(gt_json, str(pred_json))

    avg_latency = float(np.mean(latencies) * 1000.0) if latencies else 0.0
    fps = float((1.0 / np.mean(latencies))) if latencies else 0.0
    avg_memory = float(np.mean(memories)) if memories else 0.0
    max_memory = float(np.max(memories)) if memories else 0.0

    return ModelStats(
        model=wrapper.name,
        fps=fps,
        avg_latency_ms=avg_latency,
        map_50_95=metrics["map_50_95"],
        map_50=metrics["map_50"],
        avg_memory_mb=avg_memory,
        max_memory_mb=max_memory,
        num_images=total_measured,
        num_predictions=len(detections),
        pred_json=str(pred_json),
    )


def save_summary(stats: List[ModelStats], outdir: Path) -> Path:
    df = pd.DataFrame([s.__dict__ for s in stats])
    summary_csv = outdir / "summary.csv"
    summary_json = outdir / "summary.json"
    df.to_csv(summary_csv, index=False)
    summary_json.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    create_charts(df, outdir / "charts")
    return summary_csv


def build_wrappers(args: argparse.Namespace) -> tuple[List[BaseWrapper], List[dict]]:
    wrappers: List[BaseWrapper] = []
    skipped: List[dict] = []
    selected = [m.lower() for m in args.models]

    if "yolo" in selected:
        wrappers.append(UltralyticsWrapper("YOLO", args.weights_yolo, args.device, args.imgsz))
    if "rtdetr" in selected:
        wrappers.append(UltralyticsWrapper("RTDETR", args.weights_rtdetr, args.device, args.imgsz))
    if "dino" in selected:
        try:
            wrappers.append(DinoWrapper(args.dino_config, args.weights_dino, args.device, args.dino_model))
        except RuntimeError as e:
            if len(selected) == 1:
                raise
            print(f"[WARN] DINO ignoré: {e}")
            skipped.append({"model": "DINO", "reason": str(e)})

    if not wrappers:
        raise RuntimeError("Aucun modèle exploitable. Vérifiez vos dépendances et vos arguments --models.")
    return wrappers, skipped


def print_recap(df: pd.DataFrame) -> None:
    print("\n===== RÉCAPITULATIF =====")
    cols = ["model", "status", "map_50_95", "map_50", "fps", "avg_latency_ms", "max_memory_mb"]
    existing = [c for c in cols if c in df.columns]
    print(df[existing].to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    coco = COCO(args.coco_json)
    image_root = Path(args.image_root)
    wrappers, skipped_models = build_wrappers(args)

    all_stats: List[ModelStats] = []
    for wrapper in wrappers:
        stats = benchmark_model(
            wrapper=wrapper,
            coco=coco,
            image_root=image_root,
            outdir=outdir,
            device=args.device,
            conf=args.conf,
            warmup=args.warmup,
            max_images=args.max_images,
            preview_images=args.preview_images,
            class_agnostic=args.class_agnostic,
            gt_json=args.coco_json,
        )
        all_stats.append(stats)

    for skipped in skipped_models:
        all_stats.append(
            ModelStats(
                model=skipped["model"],
                fps=0.0,
                avg_latency_ms=0.0,
                map_50_95=0.0,
                map_50=0.0,
                avg_memory_mb=0.0,
                max_memory_mb=0.0,
                num_images=0,
                num_predictions=0,
                pred_json="",
                status="skipped",
                note=skipped["reason"],
            )
        )

    summary_csv = save_summary(all_stats, outdir)
    df = pd.read_csv(summary_csv)
    print_recap(df)
    print(f"\n[OK] Résultats enregistrés dans: {outdir.resolve()}")
