# Benchmark YOLO / RT-DETR / DINO

Petit projet de test pour comparer YOLO, RT-DETR et DINO sur un dataset cryo-EM.

## Lancer l'interface web

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python app.py
```

Ouvrir:

```text
http://127.0.0.1:8000
```

## Lance le benchmark

### Version complète (YOLO + RT-DETR + DINO)

```bash
python benchmark.py \
  --coco-json data/processed/11060/annotations/instances_test.json \
  --image-root data/processed/11060 \
  --outdir outputs/bench_test \
  --device cpu \
  --imgsz 1024 \
  --weights-yolo yolo11n.pt \
  --weights-rtdetr rtdetr-l.pt \
  --dino-model dino-4scale_r50_8xb2-12e_coco \
  --class-agnostic \
  --models yolo rtdetr dino
```


### Version rapide (sans DINO)

```bash
python benchmark.py \
  --coco-json data/processed/11060/annotations/instances_test.json \
  --image-root data/processed/11060 \
  --outdir outputs/bench_test \
  --device cpu \
  --imgsz 1024 \
  --weights-yolo yolo11n.pt \
  --weights-rtdetr rtdetr-l.pt \
  --class-agnostic \
  --models yolo rtdetr
```

## Générer les livrables
#nb :Si vous voulez aussi avoir document pdf et powerpoint de mon analyse vus pouvez taper ces commandes:

```bash
python generate_report.py \
  --summary-csv outputs/bench_test/summary.csv \
  --charts-dir outputs/bench_test/charts \
  --output outputs/bench_test/benchmark_report.pdf \
  --dataset-name "CryoVirusDB / EMPIAR 11060"

python generate_pptx.py \
  --summary-csv outputs/bench_test/summary.csv \
  --charts-dir outputs/bench_test/charts \
  --output outputs/bench_test/benchmark_presentation.pptx \
  --dataset-name "CryoVirusDB / EMPIAR 11060"
```

## Si DINO est vide dans les graphes

Si DINO apparait avec des zeros, c'est qu'il n'a pas ete execute (environnement MMDetection incompatible).

Utilisez un environnement conda Python 3.10:

1. source /home/srg/.miniforge3/etc/profile.d/conda.sh
2. conda create -n bench-dino python=3.10 -y
3. conda activate bench-dino
4. python -m pip install --upgrade pip setuptools wheel
5. pip install -r requirements.txt
6. python benchmark.py --coco-json data/processed/11060/annotations/instances_test.json --image-root data/processed/11060 --outdir outputs/bench_test --device cpu --imgsz 1024 --weights-yolo yolo11n.pt --weights-rtdetr rtdetr-l.pt --dino-model dino-4scale_r50_8xb2-12e_coco --class-agnostic --models yolo rtdetr dino

Puis regenerez les livrables:

1. python generate_report.py --summary-csv outputs/bench_test/summary.csv --charts-dir outputs/bench_test/charts --output outputs/bench_test/benchmark_report.pdf --dataset-name "CryoVirusDB / EMPIAR 11060"
2. python generate_pptx.py --summary-csv outputs/bench_test/summary.csv --charts-dir outputs/bench_test/charts --output outputs/bench_test/benchmark_presentation.pptx --dataset-name "CryoVirusDB / EMPIAR 11060"

