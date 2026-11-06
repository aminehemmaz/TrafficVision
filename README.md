# TrafficVision

**English** | [Français](README.fr.md)

Detection, tracking and line-crossing counting of vehicles and people in video files, built on
YOLO11 and ByteTrack.

![Counting demo on a highway](docs/demo.gif)

*`yolo11s` with ByteTrack on a highway video (first 17 seconds): vehicles are tracked and counted
in each direction as they cross the red line.*

## Features
- YOLO11 detection (COCO weights), filtered by class
- Multi-object tracking (ByteTrack or BoT-SORT) with persistent identifiers
- Directional counting (IN / OUT) per class, each object counted once
- Trails, on-screen counters and FPS, annotated video and CSV crossing log
- Web interface to analyse a video without the command line

## Installation
Tested with Python 3.13 on Windows, CPU only.
```bash
python -m venv .venv
.venv\Scripts\activate        # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Command-line usage
```bash
# Count vehicles in a video (put the file in data/)
python src/counter.py --source data/traffic.mp4 --show

# Count people
python src/counter.py --source data/street.mp4 --classes person --show

# Custom line (relative coordinates x1 y1 x2 y2)
python src/counter.py --source data/traffic.mp4 --line 0.1 0.6 0.9 0.6
```
Outputs: `outputs/result.mp4` (annotated video) and `outputs/events.csv` (crossing log).

For 4K videos, `--imgsz 1280` improves detection of distant vehicles, and `--out-width`
(1280 by default) limits the size of the annotated video.

## Web application
Upload a video, place the counting line on a preview, choose the model and classes, follow the
analysis live, then get the results: counters, crossings by class and over time, annotated video
and CSV log to download.
```bash
streamlit run src/app.py
```
Computation runs on the CPU, so the analysed duration is limited (30 s by default, 120 s at most)
to keep response times reasonable.

## Repository layout
| Path | Role |
|---|---|
| `src/engine.py` | Engine: detection, tracking, line crossing, annotation |
| `src/counter.py` | Command-line interface |
| `src/app.py` | Web application (Streamlit) |
| `src/benchmark.py` | Models and trackers compared against a ground truth |
| `benchmark/ground_truth/` | Ground truths (manual counts) |
| `docs/` | Validation, benchmark and charts |

## Test video
Videos are not versioned. Example used during development (highway, 4K, 22 s), from the
Roboflow `supervision` video examples:
```bash
curl -L -o data/highway.mp4 https://media.roboflow.com/supervision/video-examples/vehicles.mp4
python src/counter.py --source data/highway.mp4 --imgsz 1280 --line 0 0.6 1 0.6
```
Top-down views (a car park filmed from straight above) give poor results with COCO weights:
prefer an elevated oblique view or a side view.

## Results
On a highway video with a manual ground truth (9 crossings), `yolo11s` + ByteTrack counts exactly
the 9 crossings, in the right direction, at 7.6 FPS on CPU; `yolo11n` is the fastest (13.9 FPS)
but adds one extra crossing. On a dense-traffic video, the total count varies from 280 to 395
depending on model size (no ground truth). Details and limitations:
[docs/benchmark.md](docs/benchmark.md) and [docs/validation.md](docs/validation.md).

![Benchmark](docs/benchmark.png)

## License
MIT, see [LICENSE](LICENSE).
