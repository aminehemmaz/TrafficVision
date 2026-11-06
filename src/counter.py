"""Comptage d'objets en ligne de commande : YOLO11 + ByteTrack + ligne virtuelle.

Exemples :
    python src/counter.py --source data/traffic.mp4 --show
    python src/counter.py --source data/traffic.mp4 --line 0.1 0.6 0.9 0.6
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

from engine import COCO_CLASSES, DEFAULT_CLASSES, Settings, run, video_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comptage + suivi (YOLO11 + ByteTrack)")
    p.add_argument("--source", required=True, help="chemin du fichier vidéo")
    p.add_argument("--model", default="yolo11n.pt", help="poids YOLO (yolo11n/s/m...)")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   choices=list(COCO_CLASSES), help="classes à compter")
    p.add_argument("--line", nargs=4, type=float, default=[0.0, 0.6, 1.0, 0.6],
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="ligne de comptage en coordonnées relatives (0-1)")
    p.add_argument("--conf", type=float, default=0.3, help="seuil de confiance")
    p.add_argument("--imgsz", type=int, default=640, help="taille d'inférence")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml ou botsort.yaml")
    p.add_argument("--output", default="outputs/result.mp4", help="vidéo annotée en sortie")
    p.add_argument("--csv", default="outputs/events.csv", help="journal des passages")
    p.add_argument("--out-width", type=int, default=1280,
                   help="largeur de la vidéo annotée (réduit les sources 4K)")
    p.add_argument("--show", action="store_true", help="afficher la fenêtre en direct (q pour quitter)")
    p.add_argument("--no-save", action="store_true", help="ne pas écrire la vidéo")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    if not Path(source).exists():
        raise SystemExit(f"Source introuvable : {source}")
    try:
        info = video_info(source)
    except ValueError as err:
        raise SystemExit(str(err))

    settings = Settings(model=args.model, classes=args.classes, line=tuple(args.line),
                        conf=args.conf, imgsz=args.imgsz, tracker=args.tracker,
                        out_width=args.out_width)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames, counter, t0 = 0, None, time.time()
    with open(args.csv, "w", newline="", encoding="utf-8") as csv_file:
        log = csv.writer(csv_file)
        log.writerow(["frame", "time_s", "track_id", "class", "direction"])
        for step in run(source, settings):
            frames, counter = step.index + 1, step.counter
            for e in step.events:
                log.writerow([e.frame, f"{e.time_s:.2f}", e.track_id, e.label, e.direction])
            if not args.no_save:
                if writer is None:
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    h, w = step.frame.shape[:2]
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                             info.fps, (w, h))
                writer.write(step.frame)
            if args.show:
                cv2.imshow("Comptage", step.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"{frames} images en {elapsed:.1f} s ({frames / max(elapsed, 1e-6):.1f} FPS moyen)")
    if counter:
        print(f"Total -> IN : {counter.counts['in']}  OUT : {counter.counts['out']}")
        for name, v in sorted(counter.per_class.items()):
            print(f"  {name} : in {v['in']} / out {v['out']}")


if __name__ == "__main__":
    main()
