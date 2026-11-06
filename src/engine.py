"""Counting engine: YOLO detection, multi-object tracking and line-crossing counts.

Used by the command-line tool (counter.py), the benchmark (benchmark.py) and the
web application (app.py).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterator

import cv2
import numpy as np

# COCO classes available for counting
COCO_CLASSES = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7,
}
DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]

Point = tuple[float, float]


def side(p: Point, a: Point, b: Point) -> float:
    """Sign telling on which side of the line (a -> b) the point p lies."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


@dataclass
class Event:
    """A tracked object crossing the counting line."""
    frame: int
    time_s: float
    track_id: int
    label: str
    direction: str  # "in": towards the positive side of the line, "out": the opposite


class LineCounter:
    """Counts objects crossing a line, once per track identifier."""

    def __init__(self, start: Point, end: Point) -> None:
        self.start, self.end = start, end
        self.counts = {"in": 0, "out": 0}
        self.per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})
        self.seen: set[int] = set()
        self.counted: set[int] = set()
        self._last_side: dict[int, float] = {}

    def update(self, track_id: int, point: Point, label: str, frame: int, fps: float) -> Event | None:
        self.seen.add(track_id)
        current = side(point, self.start, self.end)
        previous = self._last_side.get(track_id)
        self._last_side[track_id] = current
        if previous is None or track_id in self.counted or previous * current >= 0:
            return None
        direction = "in" if previous < 0 < current else "out"
        self.counted.add(track_id)
        self.counts[direction] += 1
        self.per_class[label][direction] += 1
        return Event(frame, frame / fps, track_id, label, direction)


@dataclass
class Settings:
    model: str = "yolo11n.pt"
    classes: list[str] = field(default_factory=lambda: list(DEFAULT_CLASSES))
    line: tuple[float, float, float, float] = (0.0, 0.6, 1.0, 0.6)  # relative coordinates
    conf: float = 0.3
    imgsz: int = 640
    tracker: str = "bytetrack.yaml"
    out_width: int = 1280
    annotate: bool = True


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int  # 0 if unknown


@dataclass
class Step:
    index: int
    frame: np.ndarray | None  # annotated frame (None when annotate=False)
    events: list[Event]
    counter: LineCounter
    fps: float


def video_info(source: str | int) -> VideoInfo:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 30.0,
        frames=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0),
    )
    cap.release()
    return info


def draw_line(frame: np.ndarray, line: tuple[float, float, float, float],
              thickness: int = 3) -> np.ndarray:
    """Draw the counting line (relative coordinates) on a copy of the frame."""
    h, w = frame.shape[:2]
    out = frame.copy()
    cv2.line(out, (int(line[0] * w), int(line[1] * h)), (int(line[2] * w), int(line[3] * h)),
             (0, 0, 255), thickness)
    return out


def _draw(frame, detections, counter, trails, fps, scale):
    thickness = max(1, int(2 * scale))
    font_small, font_hud = 0.5 * scale, 0.6 * scale
    text_thickness = max(1, int(scale))
    for tid, label, (x1, y1, x2, y2) in detections:
        color = (0, 255, 0) if tid in counter.counted else (255, 160, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        cv2.putText(frame, f"{label} #{tid}", (int(x1), int(y1) - int(6 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_small, color, text_thickness, cv2.LINE_AA)
        pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, color, thickness)

    cv2.line(frame, tuple(map(int, counter.start)), tuple(map(int, counter.end)),
             (0, 0, 255), int(3 * scale))

    lines = [f"IN: {counter.counts['in']}   OUT: {counter.counts['out']}   FPS: {fps:.1f}"]
    lines += [f"{name}: in {v['in']} / out {v['out']}" for name, v in sorted(counter.per_class.items())]
    cv2.rectangle(frame, (int(8 * scale), int(8 * scale)),
                  (int(330 * scale), int((14 + 24 * len(lines)) * scale)), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (int(14 * scale), int((28 + 24 * i) * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_hud, (255, 255, 255), text_thickness, cv2.LINE_AA)
    return frame


def run(source: str | int, settings: Settings, model=None,
        max_frames: int | None = None) -> Iterator[Step]:
    """Analyse a video source and yield one `Step` per frame.

    `model` lets the caller reuse already loaded weights. Tracking is reset on every
    call so that one analysis never depends on the previous one.
    """
    from ultralytics import YOLO

    info = video_info(source)
    scale = max(info.width / 1280, 1.0)  # annotation scale for high resolutions
    start = (settings.line[0] * info.width, settings.line[1] * info.height)
    end = (settings.line[2] * info.width, settings.line[3] * info.height)
    out_w = min(settings.out_width, info.width)
    out_h = int(info.height * out_w / info.width)

    model = model or YOLO(settings.model)
    model.predictor = None  # forces a fresh tracker (track ids restart from zero)
    class_ids = [COCO_CLASSES[c] for c in settings.classes]

    counter = LineCounter(start, end)
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
    fps_smooth, t_prev = 0.0, time.time()

    stream = model.track(source=source, stream=True, persist=True, tracker=settings.tracker,
                         classes=class_ids, conf=settings.conf, imgsz=settings.imgsz, verbose=False)
    try:
        for index, result in enumerate(stream):
            if max_frames is not None and index >= max_frames:
                break
            detections, events = [], []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for xyxy, tid, cid in zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().tolist(),
                                          boxes.cls.int().cpu().tolist()):
                    label = result.names[cid]
                    point = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])  # bottom centre: ground contact point
                    trails[tid].append((int(point[0]), int(point[1])))
                    detections.append((tid, label, xyxy))
                    event = counter.update(tid, point, label, index, info.fps)
                    if event:
                        events.append(event)

            now = time.time()
            instant = 1 / max(now - t_prev, 1e-6)  # total time per frame, inference included
            t_prev = now
            fps_smooth = instant if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * instant

            frame = None
            if settings.annotate:
                frame = _draw(result.orig_img.copy(), detections, counter, trails, fps_smooth, scale)
                if out_w != info.width:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            yield Step(index, frame, events, counter, fps_smooth)
    finally:
        model.predictor = None
