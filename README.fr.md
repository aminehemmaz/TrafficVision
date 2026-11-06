# TrafficVision

[English](README.md) | **Français**

Détection, suivi et comptage de véhicules et de personnes à travers une ligne virtuelle, sur
fichier vidéo, avec YOLO11 et ByteTrack.

![Démonstration du comptage sur une autoroute](docs/demo.gif)

*`yolo11s` et ByteTrack sur une vidéo d'autoroute (17 premières secondes) : les véhicules sont
suivis et comptés dans chaque sens au franchissement de la ligne rouge.*

## Fonctionnalités
- Détection YOLO11 (poids COCO) filtrée par classe
- Suivi multi-objets (ByteTrack ou BoT-SORT) avec identifiants persistants
- Comptage directionnel (IN / OUT) par classe, chaque objet compté une seule fois
- Trajectoires, compteurs et FPS à l'écran, vidéo annotée et journal CSV des passages
- Interface web pour analyser une vidéo sans ligne de commande

## Installation
Testé avec Python 3.13 sous Windows, processeur uniquement.
```bash
python -m venv .venv
.venv\Scripts\activate        # Linux / macOS : source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation en ligne de commande
```bash
# Compter les véhicules d'une vidéo (placer le fichier dans data/)
python src/counter.py --source data/traffic.mp4 --show

# Compter des personnes
python src/counter.py --source data/street.mp4 --classes person --show

# Ligne personnalisée (coordonnées relatives x1 y1 x2 y2)
python src/counter.py --source data/traffic.mp4 --line 0.1 0.6 0.9 0.6
```
Sorties : `outputs/result.mp4` (vidéo annotée) et `outputs/events.csv` (journal des passages).

Pour les vidéos 4K, `--imgsz 1280` améliore la détection des véhicules lointains et
`--out-width` (1280 par défaut) limite la taille de la vidéo annotée.

## Application web
Envoyez une vidéo, placez la ligne de comptage sur un aperçu, choisissez le modèle et les classes,
suivez l'analyse en direct, puis récupérez les résultats : compteurs, passages par classe et dans
le temps, vidéo annotée et journal CSV à télécharger.
```bash
streamlit run src/app.py
```
Le calcul s'exécute sur le processeur : la durée analysée est donc limitée (30 s par défaut,
120 s au plus) pour garder un temps de réponse raisonnable.

## Structure du dépôt
| Chemin | Rôle |
|---|---|
| `src/engine.py` | Moteur : détection, suivi, franchissement de ligne, annotation |
| `src/counter.py` | Interface en ligne de commande |
| `src/app.py` | Application web (Streamlit) |
| `src/benchmark.py` | Comparaison de modèles et de trackers face à une vérité terrain |
| `benchmark/ground_truth/` | Vérités terrain (comptages manuels) |
| `docs/` | Validation, benchmark et graphiques |

## Vidéo de test
Les vidéos ne sont pas versionnées. Exemple utilisé pendant le développement (autoroute, 4K,
22 s), issu des exemples de la bibliothèque Roboflow `supervision` :
```bash
curl -L -o data/highway.mp4 https://media.roboflow.com/supervision/video-examples/vehicles.mp4
python src/counter.py --source data/highway.mp4 --imgsz 1280 --line 0 0.6 1 0.6
```
Les vues de dessus (parking filmé à la verticale) donnent de mauvais résultats avec les poids COCO :
préférer une vue en hauteur oblique ou de côté.

## Résultats
Sur une vidéo d'autoroute avec vérité terrain manuelle (9 passages), `yolo11s` + ByteTrack compte
exactement les 9 passages, avec le bon sens, à 7,6 FPS sur CPU ; `yolo11n` est le plus rapide
(13,9 FPS) mais ajoute un passage en trop. Sur une vidéo de trafic dense, le total compté varie
de 280 à 395 selon la taille du modèle (pas de vérité terrain). Détails et limites :
[docs/fr/benchmark.md](docs/fr/benchmark.md) et [docs/fr/validation.md](docs/fr/validation.md).

![Benchmark](docs/benchmark.fr.png)

## Licence
MIT, voir [LICENSE](LICENSE).
