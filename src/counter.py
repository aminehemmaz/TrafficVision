"""Comptage d'objets avec YOLO11 + ByteTrack + ligne virtuelle.

Exemples :
    python src/counter.py --source data/traffic.mp4 --show
    python src/counter.py --source 0 --show                      # webcam
    python src/counter.py --source data/traffic.mp4 --line 0.1 0.6 0.9 0.6
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Classes COCO : personne et vehicules
COCO_CLASSES = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7,
}
DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]


def side(p: tuple[float, float], a: tuple[int, int], b: tuple[int, int]) -> float:
    """Signe indiquant de quel cote de la droite (a->b) se trouve le point p."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Comptage + tracking (YOLO11 + ByteTrack)")
    p.add_argument("--source", default="0", help="chemin video ou index webcam (defaut: 0)")
    p.add_argument("--model", default="yolo11n.pt", help="poids YOLO (yolo11n/s/m...)")
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   choices=list(COCO_CLASSES), help="classes a compter")
    p.add_argument("--line", nargs=4, type=float, default=[0.0, 0.6, 1.0, 0.6],
                   metavar=("X1", "Y1", "X2", "Y2"),
                   help="ligne de comptage en coordonnees relatives (0-1)")
    p.add_argument("--conf", type=float, default=0.3, help="seuil de confiance")
    p.add_argument("--imgsz", type=int, default=640, help="taille d'inference")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml ou botsort.yaml")
    p.add_argument("--output", default="outputs/result.mp4", help="video annotee en sortie")
    p.add_argument("--csv", default="outputs/events.csv", help="journal des passages")
    p.add_argument("--out-width", type=int, default=1280,
                   help="largeur de la video annotee (reduit les sources 4K)")
    p.add_argument("--show", action="store_true", help="afficher la fenetre en direct (q pour quitter)")
    p.add_argument("--no-save", action="store_true", help="ne pas ecrire la video")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        raise SystemExit(f"Source introuvable : {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit("Impossible d'ouvrir la source video.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    out_w = min(args.out_width, width)
    out_h = int(height * out_w / width)
    k = max(width / 1280, 1.0)  # facteur d'echelle des annotations pour les grandes resolutions
    th = max(1, int(2 * k))

    x1, y1, x2, y2 = args.line
    a = (int(x1 * width), int(y1 * height))
    b = (int(x2 * width), int(y2 * height))

    model = YOLO(args.model)
    class_ids = [COCO_CLASSES[c] for c in args.classes]

    writer = None
    if not args.no_save:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(args.csv, "w", newline="", encoding="utf-8")
    log = csv.writer(csv_file)
    log.writerow(["frame", "time_s", "track_id", "class", "direction"])

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
    last_side: dict[int, float] = {}
    counted: set[int] = set()
    counts = {"in": 0, "out": 0}
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})

    frame_idx, t0, fps_smooth = 0, time.time(), 0.0
    t_prev = t0
    stream = model.track(
        source=source, stream=True, persist=True, tracker=args.tracker,
        classes=class_ids, conf=args.conf, imgsz=args.imgsz, verbose=False,
    )

    for result in stream:
        frame = result.orig_img.copy()
        boxes = result.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            clss = boxes.cls.int().cpu().tolist()
            for (bx1, by1, bx2, by2), tid, cid in zip(xyxy, ids, clss):
                name = result.names[cid]
                cx, cy = (bx1 + bx2) / 2, by2  # point bas-centre : contact au sol
                trails[tid].append((int(cx), int(cy)))

                s = side((cx, cy), a, b)
                prev = last_side.get(tid)
                if prev is not None and tid not in counted and prev * s < 0:
                    direction = "in" if prev < 0 < s else "out"
                    counts[direction] += 1
                    per_class[name][direction] += 1
                    counted.add(tid)
                    log.writerow([frame_idx, f"{frame_idx / fps:.2f}", tid, name, direction])
                last_side[tid] = s

                color = (0, 255, 0) if tid in counted else (255, 160, 0)
                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)), color, th)
                cv2.putText(frame, f"{name} #{tid}", (int(bx1), int(by1) - int(6 * k)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * k, color, max(1, int(k)), cv2.LINE_AA)
                pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, color, th)

        cv2.line(frame, a, b, (0, 0, 255), int(3 * k))
        now = time.time()
        dt = max(now - t_prev, 1e-6)  # temps total par frame (inference incluse)
        t_prev = now
        fps_smooth = 1 / dt if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * (1 / dt)

        lines = [f"IN: {counts['in']}   OUT: {counts['out']}   FPS: {fps_smooth:.1f}"]
        lines += [f"{n}: in {v['in']} / out {v['out']}" for n, v in sorted(per_class.items())]
        cv2.rectangle(frame, (int(8 * k), int(8 * k)),
                      (int(330 * k), int((14 + 24 * len(lines)) * k)), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (int(14 * k), int((28 + 24 * i) * k)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * k, (255, 255, 255),
                        max(1, int(k)), cv2.LINE_AA)

        if out_w != width:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if writer:
            writer.write(frame)
        if args.show:
            cv2.imshow("Comptage", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1

    if writer:
        writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    elapsed = time.time() - t0
    print(f"{frame_idx} frames en {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} FPS moyen)")
    print(f"Total -> IN: {counts['in']}  OUT: {counts['out']}")
    for n, v in sorted(per_class.items()):
        print(f"  {n}: in {v['in']} / out {v['out']}")


if __name__ == "__main__":
    main()
