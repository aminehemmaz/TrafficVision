# Counting validation

[Version française](fr/validation.md)

Video: `highway.mp4` (highway, 4K, 25 fps, 538 frames, 22 s), from the Roboflow `supervision`
video examples. The counting line is horizontal at 60% of the image height (`--line 0 0.6 1 0.6`).
The reference point of an object is the middle of the bottom edge of its box (contact with the road).

## Ground truth (manual count)
Built by looking at 54 frames (one every 10) with the line drawn on them. Frame numbers are
approximate (+/- 10 frames).

| Direction | Vehicle | Approx. frame |
|---|---|---|
| IN (towards the camera) | black car | 10 |
| IN | white car | 21 |
| IN | dark car | 210 |
| IN | grey Skoda | 350 |
| IN | DAF truck | 410 |
| IN | blue car | 480 |
| OUT (moving away) | grey car | 10 |
| OUT | grey car | 70 |
| OUT | white car | 415 |

Edge case: the white truck present in the very first frame is already on the line. It is
counted neither in the ground truth nor by the script.

## Script results (yolo11n + ByteTrack, conf 0.3)

| `imgsz` | Duration | FPS (CPU) | IN | OUT |
|---|---|---|---|---|
| 640 | 118 s | 4.6 | 6 | 3 |
| 960 | 146 s | 3.7 | 6 | 3 |
| 1280 | 183 s | 2.9 | 6 | 3 |
| **Manual** | | | **6** | **3** |

The count matches the ground truth for the three inference sizes, and the crossings in the
CSV log fall in the same frame windows as the ground truth.

## Limitations
- A single video and 9 crossings: too few for a statistically meaningful accuracy.
- The ground truth comes from sampled frames, not from a frame-by-frame review.
- Large, well separated vehicles: no test on dense scenes, at night or with heavy occlusion.
- Top-down views (a car park filmed from straight above) give poor detections with COCO weights.
