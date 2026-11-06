# TrafficVision : comptage d'objets en temps réel (YOLO11 + ByteTrack)

Détection, suivi et comptage de véhicules / personnes à travers une ligne virtuelle, sur vidéo ou webcam.

> 🚧 Projet en cours — démo GIF et benchmarks à venir.

## Fonctionnalités
- Détection YOLO11 (COCO) filtrée par classe
- Tracking multi-objets (ByteTrack / BoT-SORT) avec identifiants persistants
- Comptage directionnel (IN / OUT) par classe, sans double comptage
- Trajectoires, HUD (compteurs + FPS), vidéo annotée et journal CSV des passages

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Utilisation
```bash
# Vidéo (placer le fichier dans data/)
python src/counter.py --source data/traffic.mp4 --show

# Webcam, comptage de personnes
python src/counter.py --source 0 --classes person --show

# Ligne personnalisée (coordonnées relatives x1 y1 x2 y2)
python src/counter.py --source data/traffic.mp4 --line 0.1 0.6 0.9 0.6
```
Sorties : `outputs/result.mp4` et `outputs/events.csv`.

Pour les vidéos 4K, `--imgsz 1280` améliore la détection des véhicules lointains et
`--out-width` (1280 par défaut) limite la taille de la vidéo annotée.

## Vidéo de test
Les vidéos ne sont pas versionnées. Exemple utilisé pendant le développement
(autoroute, 4K, 22 s), issue des exemples de la bibliothèque Roboflow `supervision` :
```bash
curl -L -o data/highway.mp4 https://media.roboflow.com/supervision/video-examples/vehicles.mp4
python src/counter.py --source data/highway.mp4 --imgsz 1280 --line 0 0.6 1 0.6
```
Les vues de dessus (parking filmé à la verticale) donnent de mauvais résultats avec les poids COCO :
préférer une vue en hauteur oblique ou de côté.

## Feuille de route
- [ ] Démo GIF + benchmark (yolo11n vs s vs m : FPS / précision du comptage)
- [ ] Zones polygonales et temps de présence
- [ ] Interface Streamlit
- [ ] Fine-tuning sur dataset custom
- [ ] Export ONNX et déploiement (Docker / Hugging Face Spaces)
