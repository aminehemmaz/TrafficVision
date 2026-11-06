# Validation du comptage (phase 2)

Vidéo : `highway.mp4` (autoroute, 4K, 25 fps, 538 images, 22 s), exemples de la bibliothèque
Roboflow `supervision`. Ligne de comptage horizontale à 60 % de la hauteur (`--line 0 0.6 1 0.6`).
Point de référence d'un objet : milieu du bord bas de sa boîte (contact avec la route).

## Vérité terrain (comptage manuel)
Obtenue en regardant 54 images (une toutes les 10) avec la ligne tracée. Les numéros d'image
sont approximatifs (± 10 images).

| Sens | Véhicule | Image approx. |
|---|---|---|
| IN (vers la caméra) | voiture noire | 10 |
| IN | voiture blanche | 21 |
| IN | voiture sombre | 210 |
| IN | Skoda grise | 350 |
| IN | camion DAF | 410 |
| IN | voiture bleue | 480 |
| OUT (s'éloigne) | voiture grise | 10 |
| OUT | voiture grise | 70 |
| OUT | voiture blanche | 415 |

Cas limite : le camion blanc présent dès la première image est déjà sur la ligne. Il n'est
compté ni dans la vérité terrain ni par le script.

## Résultats du script (yolo11n + ByteTrack, conf 0,3)

| `imgsz` | Durée | FPS (CPU) | IN | OUT |
|---|---|---|---|---|
| 640 | 118 s | 4,6 | 6 | 3 |
| 960 | 146 s | 3,7 | 6 | 3 |
| 1280 | 183 s | 2,9 | 6 | 3 |
| **Manuel** | | | **6** | **3** |

Le comptage est identique à la vérité terrain pour les trois tailles d'inférence, et les
événements du CSV tombent dans la même fenêtre d'images que la vérité terrain.

## Limites
- Une seule vidéo et 9 passages : trop peu pour une précision statistique.
- Vérité terrain faite à partir d'images échantillonnées, pas image par image.
- Véhicules grands et bien séparés : pas de test sur scène dense, de nuit ou avec occlusions.
- Vues de dessus (parking filmé à la verticale) : mauvaises détections avec les poids COCO.
