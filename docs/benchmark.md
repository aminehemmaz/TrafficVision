# Benchmark modèles x trackers (phase 3)

Reproduire : `python src/benchmark.py --source data/highway_720p.mp4 --gt benchmark/ground_truth/highway.json`
Matériel : CPU uniquement (pas de GPU), `imgsz 640`, `conf 0,3`, classes voiture / moto / bus / camion.

![Benchmark](benchmark.png)

## 1. Autoroute (538 images, 9 passages en vérité terrain)
Copie 720p de `highway.mp4`. Événements appariés à la vérité terrain (même sens, ± 25 images).

| Modèle | Tracker | FPS | IN / OUT | TP | FP | FN | Pistes |
|---|---|---|---|---|---|---|---|
| yolo11n | ByteTrack | **13,9** | 7 / 3 | 9 | 1 | 0 | 23 |
| yolo11n | BoT-SORT | 10,7 | 7 / 3 | 9 | 1 | 0 | 25 |
| yolo11s | ByteTrack | 7,6 | 6 / 3 | 9 | **0** | 0 | 24 |
| yolo11s | BoT-SORT | 5,9 | 7 / 3 | 9 | 1 | 0 | 24 |
| yolo11m | ByteTrack | 2,4 | 6 / 3 | 9 | **0** | 0 | 22 |
| yolo11m | BoT-SORT | 2,5 | 6 / 3 | 9 | **0** | 0 | 22 |

Aucun passage n'est manqué (FN = 0). Le seul défaut est un passage IN en trop avec `yolo11n`
(et `yolo11s` + BoT-SORT), dans les 8 premières images, là où le camion de tête est à cheval sur la
ligne : double comptage d'un même véhicule ou comptage du camion. Je n'ai pas isolé lequel des deux.

## 2. Trafic dense (1275 images, 1080p, pas de vérité terrain)
Compter à la main plus de cent véhicules serrés n'est pas fiable. Cette vidéo sert à comparer
les configurations entre elles, pas à mesurer une précision.

| Modèle | Tracker | FPS | IN / OUT | Total | Pistes |
|---|---|---|---|---|---|
| yolo11n | ByteTrack | 12,0 | 116 / 164 | 280 | 1029 |
| yolo11n | BoT-SORT | 8,4 | 116 / 166 | 282 | 857 |
| yolo11s | ByteTrack | 6,5 | 171 / 184 | 355 | 1057 |
| yolo11m | ByteTrack | 2,6 | 227 / 168 | 395 | 1097 |

## Ce qu'on peut conclure
- **Dans le trafic dense, la taille du modèle change fortement le total** : de 280 à 395 passages
  (+41 %) de `n` à `m`. Les petits modèles manquent probablement des véhicules serrés ou
  partiellement masqués. Sans vérité terrain, c'est une hypothèse, pas une mesure.
- **BoT-SORT n'apporte rien ici** : plus lent avec `n` et `s` (réidentification désactivée par
  défaut), même précision. Il produit moins de pistes (857 contre 1029 avec `n`), ce qui suggère
  moins de fragmentation, mais le total de passages est le même (282 contre 280).
- **Compromis vitesse / fiabilité** : `yolo11n` + ByteTrack est le plus rapide (~14 FPS sur CPU),
  `yolo11s` + ByteTrack est la meilleure précision mesurée pour un coût modéré (7,6 FPS), `yolo11m`
  est trop lent sur CPU pour un gain non démontré sur l'autoroute.

## Limites
- Vérité terrain : une seule vidéo, 9 passages, comptage manuel sur images échantillonnées.
- Pas de répétition des mesures de vitesse : les FPS dépendent de la charge de la machine.
- Autoroute testée sur une copie 720p (réencodée en mp4v). Le même `yolo11n` + ByteTrack lu depuis la
  4K d'origine avait donné 0 faux positif (9 TP) : la copie réencodée n'est donc pas strictement
  équivalente, et ce faux positif est sensible au prétraitement.
- Les poids COCO ne couvrent pas les vues de dessus ni la nuit.
