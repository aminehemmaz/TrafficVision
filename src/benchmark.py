"""Benchmark modeles x trackers contre une verite terrain.

Exemple :
    python src/benchmark.py --source data/highway.mp4 --gt benchmark/ground_truth/highway.json
    python src/benchmark.py --check outputs/highway.csv   # evalue un CSV existant
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from engine import DEFAULT_CLASSES, Settings, run

TOLERANCE = 25  # images (1 s a 25 fps) pour apparier un evenement a la verite terrain


def match_events(pred: list[dict], gt: list[dict], tol: int = TOLERANCE) -> dict:
    """Appariement glouton (meme sens, ecart d'images <= tol). Retourne TP/FP/FN et scores."""
    free = list(range(len(gt)))
    tp = 0
    for p in sorted(pred, key=lambda e: e["frame"]):
        best, best_d = None, tol + 1
        for i in free:
            g = gt[i]
            d = abs(g["frame"] - p["frame"])
            if g["direction"] == p["direction"] and d < best_d:
                best, best_d = i, d
        if best is not None:
            free.remove(best)
            tp += 1
    fp, fn = len(pred) - tp, len(gt) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def read_csv_events(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [{"frame": int(r["frame"]), "direction": r["direction"]} for r in csv.DictReader(f)]


def run_config(model_name: str, tracker: str, source: str, line: list[float],
               imgsz: int, conf: float, classes: list[str]) -> dict:
    settings = Settings(model=model_name, classes=classes, line=tuple(line), conf=conf,
                        imgsz=imgsz, tracker=tracker, annotate=False)
    events: list[dict] = []
    n, seen = 0, 0
    t0 = time.time()
    for step in run(source, settings):
        n = step.index + 1
        seen = len(step.counter.seen)
        events += [{"frame": e.frame, "direction": e.direction} for e in step.events]
    elapsed = time.time() - t0
    return {"model": model_name, "tracker": tracker, "imgsz": imgsz, "conf": conf,
            "frames": n, "seconds": round(elapsed, 1), "fps": round(n / elapsed, 2),
            "unique_tracks": seen, "events": events,
            "in": sum(e["direction"] == "in" for e in events),
            "out": sum(e["direction"] == "out" for e in events)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/highway.mp4")
    p.add_argument("--gt", default="benchmark/ground_truth/highway.json")
    p.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"])
    p.add_argument("--trackers", nargs="+", default=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--out", default="outputs/benchmark.json")
    p.add_argument("--check", help="evalue un CSV de passages existant et quitte")
    p.add_argument("--only", nargs="*", help="limiter a des paires modele:tracker (ex: yolo11n.pt:bytetrack.yaml)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    gt = json.loads(Path(args.gt).read_text(encoding="utf-8"))
    if args.check:
        m = match_events(read_csv_events(args.check), gt["events"])
        print(json.dumps(m, indent=2))
        return

    combos = [(m, t) for m in args.models for t in args.trackers]
    if args.only:
        combos = [tuple(x.split(":")) for x in args.only]
    results = []
    for model_name, tracker in combos:
        print(f"-> {model_name} + {tracker}", flush=True)
        res = run_config(model_name, tracker, args.source, gt["line"], args.imgsz,
                         args.conf, DEFAULT_CLASSES)
        if gt.get("events"):
            res.update(match_events(res["events"], gt["events"]))
            quality = f"TP {res['tp']} FP {res['fp']} FN {res['fn']}"
        else:
            quality = "sans verite terrain"
        results.append(res)
        print(f"   {res['seconds']}s | {res['fps']} FPS | IN {res['in']} OUT {res['out']} | "
              f"{quality} | tracks {res['unique_tracks']}", flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
