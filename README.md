# TrafficVision

**English** | [Français](README.fr.md)

Detection, tracking and line-crossing counting of vehicles and people in video files, built on
YOLO11 and ByteTrack.

> **This is an MVP**: a first working version that demonstrates the whole pipeline (detection,
> tracking, counting, validation and benchmark). See
> [Limitations and next steps](#limitations-and-next-steps) and
> [Licensing and commercial use](#licensing-and-commercial-use).

![Counting demo on a highway](docs/demo.gif)

*`yolo11s` with ByteTrack on a highway video (first 17 seconds): vehicles are tracked and counted
in each direction as they cross the red line.*

![Counting demo on dense two-way traffic](docs/demo_dense.gif)

*Dense two-way traffic, 15-second extract (`yolo11s` with ByteTrack). This scene also shows the
current limits of the MVP: vehicles at medium and long distance are partly missed at the default
inference size, and a fixed road sign is detected as a car.*

*Car park filmed from above, full 54 seconds (`yolo11s` with ByteTrack): a pedestrian and two cars
cross the line and are counted (3 crossings), while a cyclist is tracked but never crosses it.*

https://github.com/user-attachments/assets/19f67883-d225-46dc-a157-65886acb328a

## Features
- YOLO11 detection (COCO weights), filtered by class
- Multi-object tracking (ByteTrack or BoT-SORT) with persistent identifiers
- Directional counting (IN / OUT) per class, each object counted once
- Trails, on-screen counters and FPS, annotated video and CSV crossing log

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

## Repository layout
| Path | Role |
|---|---|
| `src/engine.py` | Engine: detection, tracking, line crossing, annotation |
| `src/counter.py` | Command-line interface |
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
Top-down views (a car park filmed from straight above) are harder for COCO weights: on one such
clip `yolo11n` found very few vehicles, while `yolo11s` tracked the cars and the pedestrian of
another one well. An elevated oblique view or a side view is the safest choice.

## Results
On a highway video with a manual ground truth (9 crossings), `yolo11s` + ByteTrack counts exactly
the 9 crossings, in the right direction, at 7.6 FPS on CPU; `yolo11n` is the fastest (13.9 FPS)
but adds one extra crossing. On a dense-traffic video, the total count varies from 280 to 395
depending on model size (no ground truth). Details and limitations:
[docs/benchmark.md](docs/benchmark.md) and [docs/validation.md](docs/validation.md).

![Benchmark](docs/benchmark.png)

## Limitations and next steps
Known limitations of this MVP:
- On dense scenes, detection is incomplete for vehicles at medium and long distance at the default
  inference size, so the counts are underestimated (see [docs/benchmark.md](docs/benchmark.md)).
- A fixed object can be detected permanently as a vehicle (for example a road sign on a wall).
- Only the 80 COCO classes are available, and the weights are not adapted to a specific scene.
- Validation rests on a single manually counted video with 9 crossings.
- Computation runs on the CPU, which limits the processing speed.

Improvements needed:
- **Detection quality**: fine-tune the detector on an annotated dataset of the target scenes
  (viewpoint, lighting, vehicle types), and use a larger inference size or image tiling for
  distant vehicles.
- **Advanced counting**: polygonal zones (entry, exit, waiting area), dwell time per object,
  speed estimation (requires camera calibration) and exported statistics (CSV, traffic per minute).
- **Validation**: more videos with a reliable ground truth, in varied conditions (dense traffic,
  night, rain).

## Licensing and commercial use
This MVP relies on Ultralytics YOLO11, released under the **AGPL-3.0** license (code, weights and
the trackers integrated in the package). The AGPL does allow commercial use, but only if its terms
are met, in particular making the complete source code of the application available to its users,
including when it is offered as a network service. To use YOLO in a closed-source commercial
product, Ultralytics sells an Enterprise License; its price is given on request and is not
published.

For a commercial project, the alternative is to use models under permissive licenses, such as
**RT-DETR** (a real-time version of DETR, the DEtection TRansformer), whose checkpoints published
on Hugging Face are under Apache-2.0 (check the model card of each checkpoint), fine-tuned on your
own data. This is general information, not legal advice.

I am already working on a 100% sovereign version of this project.

## License
The code of this repository is released under the MIT license, see [LICENSE](LICENSE).
Third-party components, such as Ultralytics, keep their own licenses (see above).
