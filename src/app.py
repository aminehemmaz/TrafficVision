"""Demo web application: object counting on a video.

Run with:
    streamlit run src/app.py
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from engine import Settings, draw_line, run, video_info

st.set_page_config(page_title="TrafficVision", layout="wide")

CLASS_LABELS = {
    "car": "Car", "motorcycle": "Motorcycle", "bus": "Bus", "truck": "Truck",
    "person": "Person", "bicycle": "Bicycle",
}
MODELS = {
    "yolo11n: fast": "yolo11n.pt",
    "yolo11s: more reliable, about twice as slow": "yolo11s.pt",
}
ALLOWED_TYPES = ["mp4", "avi", "mov", "mkv"]
MAX_ANALYSIS_SECONDS = 120


@st.cache_resource(show_spinner="Loading model...")
def load_model(name: str):
    from ultralytics import YOLO
    return YOLO(name)


def workdir() -> Path:
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="trafficvision_")
    return Path(st.session_state.workdir)


def save_upload(upload) -> Path:
    """Write the uploaded video to disk (once per file)."""
    path = workdir() / f"input_{upload.file_id}{Path(upload.name).suffix}"
    if not path.exists():
        path.write_bytes(upload.getbuffer())
    return path


def first_frame(path: Path):
    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so the browser can play the result."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return False
    cmd = [exe, "-y", "-i", str(src), "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


def analyse(path: Path, settings: Settings, max_frames: int, info) -> dict:
    model = load_model(settings.model)
    total = min(info.frames, max_frames) if info.frames else max_frames
    raw = workdir() / "annotated_raw.mp4"
    writer = None
    events = []
    progress = st.progress(0.0, text="Analysis in progress...")
    live = st.empty()
    metrics_box = st.empty()
    t0 = time.time()
    last = None

    for step in run(str(path), settings, model=model, max_frames=max_frames):
        last = step
        events += step.events
        if writer is None:
            h, w = step.frame.shape[:2]
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (w, h))
        writer.write(step.frame)
        if step.index % 4 == 0:
            live.image(step.frame, channels="BGR", width="stretch")
            with metrics_box.container():
                cols = st.columns(3)
                cols[0].metric("IN", step.counter.counts["in"])
                cols[1].metric("OUT", step.counter.counts["out"])
                cols[2].metric("Speed", f"{step.fps:.1f} frames/s")
            progress.progress(min((step.index + 1) / total, 1.0),
                              text=f"Frame {step.index + 1} of {total}")
    if writer:
        writer.release()
    progress.empty()
    live.empty()
    metrics_box.empty()

    video = workdir() / "annotated.mp4"
    playable = bool(writer) and to_h264(raw, video)
    if writer and not playable:
        shutil.copy(raw, video)

    return {
        "events": events,
        "per_class": {k: dict(v) for k, v in last.counter.per_class.items()} if last else {},
        "counts": dict(last.counter.counts) if last else {"in": 0, "out": 0},
        "frames": last.index + 1 if last else 0,
        "seconds": time.time() - t0,
        "fps": info.fps,
        "video": str(video) if writer else None,
        "playable": playable,
        "settings": settings,
    }


def show_results(res: dict) -> None:
    st.subheader("Results")
    c_in, c_out, c_total, c_speed = st.columns(4)
    c_in.metric("Entries (IN)", res["counts"]["in"])
    c_out.metric("Exits (OUT)", res["counts"]["out"])
    c_total.metric("Total", res["counts"]["in"] + res["counts"]["out"])
    c_speed.metric("Average speed", f"{res['frames'] / max(res['seconds'], 1e-6):.1f} frames/s")
    st.caption(f"Computation: {res['seconds']:.0f} s for {res['frames']} frames.")

    if not res["events"]:
        st.info("No crossing detected. Check the line position, the selected classes "
                "and the confidence threshold.")
        return

    df = pd.DataFrame([{
        "Frame": e.frame, "Time (s)": round(e.time_s, 2), "Track": e.track_id,
        "Class": CLASS_LABELS.get(e.label, e.label),
        "Direction": "IN" if e.direction == "in" else "OUT",
    } for e in res["events"]])

    left, right = st.columns(2)
    with left:
        st.markdown("**Crossings by class**")
        by_class = pd.DataFrame(res["per_class"]).T.rename(columns={"in": "IN", "out": "OUT"})
        by_class.index = [CLASS_LABELS.get(i, i) for i in by_class.index]
        st.dataframe(by_class, width="stretch")
    with right:
        st.markdown("**Crossings over time**")
        bucket = 5 if df["Time (s)"].max() > 20 else 2
        timeline = (df.assign(Interval=(df["Time (s)"] // bucket * bucket).astype(int))
                      .pivot_table(index="Interval", columns="Direction", values="Track",
                                   aggfunc="count", fill_value=0)
                      .reindex(columns=["IN", "OUT"], fill_value=0))
        st.bar_chart(timeline, color=["#2a78d6", "#eb6834"],
                     x_label=f"Time (s), {bucket} s bins", y_label="Crossings", width="stretch")

    if res["video"]:
        st.markdown("**Annotated video**")
        if res["playable"]:
            st.video(res["video"])
        else:
            st.caption("In-browser playback is unavailable on this machine: download the video instead.")
        st.download_button("Download annotated video", Path(res["video"]).read_bytes(),
                           file_name="trafficvision_annotated.mp4", mime="video/mp4")

    st.markdown("**Crossing log**")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download log (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="trafficvision_crossings.csv", mime="text/csv")


def main() -> None:
    st.title("TrafficVision")
    st.caption("Detection, tracking and counting of objects crossing a virtual line "
               "(YOLO11 and ByteTrack).")

    with st.sidebar:
        st.header("Settings")
        model_label = st.selectbox("Model", list(MODELS))
        classes = st.multiselect("Objects to count", list(CLASS_LABELS),
                                 default=["car", "motorcycle", "bus", "truck"],
                                 format_func=CLASS_LABELS.get)
        conf = st.slider("Confidence threshold", 0.10, 0.90, 0.30, 0.05,
                         help="Lower values detect more objects, at the risk of false detections.")
        with st.expander("Advanced options"):
            tracker = st.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"],
                                   format_func=lambda t: "ByteTrack" if t.startswith("byte") else "BoT-SORT")
            imgsz = st.select_slider("Inference size", [480, 640, 960, 1280], value=640,
                                     help="Larger sizes help with distant objects but are slower.")

    upload = st.file_uploader("Video to analyse", type=ALLOWED_TYPES)
    if upload is None:
        st.info("Upload a video (mp4, avi, mov or mkv) to get started. An elevated or side view "
                "where vehicles cross the image works best.")
        return
    if not classes:
        st.warning("Select at least one type of object to count.")
        return

    if st.session_state.get("upload_id") != upload.file_id:
        st.session_state.upload_id = upload.file_id
        st.session_state.pop("result", None)

    path = save_upload(upload)
    try:
        info = video_info(str(path))
    except ValueError:
        st.error("This video cannot be read. Try another file or format.")
        return
    frame = first_frame(path)
    if frame is None:
        st.error("The first frame of the video cannot be read.")
        return
    duration = info.frames / info.fps if info.frames else 0

    st.subheader("1. Counting line")
    controls, preview = st.columns([1, 2])
    with controls:
        st.caption("Line position as a percentage of the image.")
        x1 = st.slider("Start, horizontal", 0, 100, 0)
        y1 = st.slider("Start, vertical", 0, 100, 60)
        x2 = st.slider("End, horizontal", 0, 100, 100)
        y2 = st.slider("End, vertical", 0, 100, 60)
        st.caption("IN direction: crossing towards the right-hand side of the line, "
                   "going from start to end.")
    line = (x1 / 100, y1 / 100, x2 / 100, y2 / 100)
    with preview:
        scale = max(frame.shape[1] / 1280, 1.0)
        st.image(draw_line(frame, line, int(4 * scale)), channels="BGR", width="stretch")

    st.subheader("2. Analysis")
    st.caption(f"Video: {info.width} x {info.height}, {info.fps:.0f} frames/s, {duration:.0f} s.")
    limit = int(min(duration, MAX_ANALYSIS_SECONDS)) if duration else MAX_ANALYSIS_SECONDS
    if limit > 5:
        seconds = st.slider("Analysed duration (seconds)", 5, limit, min(limit, 30),
                            help="Computation runs on the CPU: a short duration gives a quick result.")
    else:
        seconds = max(limit, 1)
    if st.button("Run analysis", type="primary"):
        settings = Settings(model=MODELS[model_label], classes=classes, line=line, conf=conf,
                            imgsz=imgsz, tracker=tracker)
        st.session_state.result = analyse(path, settings, int(seconds * info.fps), info)

    if "result" in st.session_state:
        show_results(st.session_state.result)


main()
