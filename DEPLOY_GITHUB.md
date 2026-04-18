# GitHub et Render

Guide court pour publier le projet sur GitHub puis le déployer sur Render.

## 1. Fichiers à ajouter à Git

Ajoute au dépôt les fichiers source et de configuration:
- `app.py`
- `benchmark.py`
- `generate_report.py`
- `generate_pptx.py`
- `prepare_cryovirusdb.py`
- `render.yaml`
- `requirements.txt`
- `README.md`
- `DEPLOY_GITHUB.md`
- `RENDER_IMAGES.md`

Ne fais pas `git add` sur:
- `.venv/`
- `__pycache__/`
- `web_uploads/`
- `web_runs/`
- `outputs/`
- les gros fichiers temporaires générés localement

## 2. Commandes Git

```bash
cd "/home/srg/Documents/IA/bench_project (2)"
git init
git add app.py benchmark.py generate_report.py generate_pptx.py prepare_cryovirusdb.py render.yaml requirements.txt README.md DEPLOY_GITHUB.md RENDER_IMAGES.md
git commit -m "Initial project setup"
git branch -M main
git remote add origin https://github.com/VOTRE_COMPTE/VOTRE_REPO.git
git push -u origin main
```

Si le dépôt Git existe déjà, remplace seulement le `git add` / `git commit` / `git push`.

## 3. Vérifier sur GitHub

Après le push, vérifie que le dépôt contient bien:
- le code Python
- `render.yaml`
- le `README.md`

## 4. Déployer sur Render

Dans Render:
1. clique sur **New**
2. choisis **Blueprint**
3. connecte ton compte GitHub
4. sélectionne le repository
5. Render lit automatiquement `render.yaml`

## 5. Variables utiles sur Render

Les variables principales sont déjà dans `render.yaml`:
- `DEVICE=cpu`
- `ENABLE_DINO=1`
- `YOLO_WEIGHTS=yolo11n.pt`
- `RTDETR_WEIGHTS=rtdetr-l.pt`
- `DINO_MODEL=dino-4scale_r50_8xb2-12e_coco`

Si tu changes les poids, modifie les variables dans Render.

## 6. Utilisation sur Render

L’application web est faite pour:
- uploader une image
- choisir un modèle
- afficher l’image annotée

Pour Render, le mode le plus fiable est l’upload d’image depuis le navigateur.

## 7. Test local rapide

```bash
python app.py
```

Puis ouvrir:

```text
http://127.0.0.1:8000
```
