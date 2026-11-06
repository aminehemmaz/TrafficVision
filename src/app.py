"""Application web de démonstration : comptage d'objets sur une vidéo.

Lancement :
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
    "car": "Voiture", "motorcycle": "Moto", "bus": "Bus", "truck": "Camion",
    "person": "Personne", "bicycle": "Vélo",
}
MODELS = {
    "yolo11n : rapide": "yolo11n.pt",
    "yolo11s : plus fiable, deux fois plus lent": "yolo11s.pt",
}
ALLOWED_TYPES = ["mp4", "avi", "mov", "mkv"]
MAX_ANALYSIS_SECONDS = 120


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_model(name: str):
    from ultralytics import YOLO
    return YOLO(name)


def workdir() -> Path:
    if "workdir" not in st.session_state:
        st.session_state.workdir = tempfile.mkdtemp(prefix="trafficvision_")
    return Path(st.session_state.workdir)


def save_upload(upload) -> Path:
    """Écrit la vidéo envoyée sur disque (une seule fois par fichier)."""
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
    """Réencode en H.264 pour une lecture dans le navigateur."""
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
    progress = st.progress(0.0, text="Analyse en cours...")
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
                cols[2].metric("Vitesse", f"{step.fps:.1f} images/s")
            progress.progress(min((step.index + 1) / total, 1.0),
                              text=f"Image {step.index + 1} sur {total}")
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
    st.subheader("Résultats")
    c_in, c_out, c_total, c_time = st.columns(4)
    c_in.metric("Entrées (IN)", res["counts"]["in"])
    c_out.metric("Sorties (OUT)", res["counts"]["out"])
    c_total.metric("Total", res["counts"]["in"] + res["counts"]["out"])
    c_time.metric("Vitesse moyenne", f"{res['frames'] / max(res['seconds'], 1e-6):.1f} images/s")
    st.caption(f"Calcul : {res['seconds']:.0f} s pour {res['frames']} images.")

    if not res["events"]:
        st.info("Aucun passage détecté. Vérifiez la position de la ligne, les classes choisies "
                "et le seuil de confiance.")
        return

    df = pd.DataFrame([{
        "Image": e.frame, "Temps (s)": round(e.time_s, 2), "Piste": e.track_id,
        "Classe": CLASS_LABELS.get(e.label, e.label),
        "Sens": "IN" if e.direction == "in" else "OUT",
    } for e in res["events"]])

    left, right = st.columns(2)
    with left:
        st.markdown("**Passages par classe**")
        by_class = pd.DataFrame(res["per_class"]).T.rename(columns={"in": "IN", "out": "OUT"})
        by_class.index = [CLASS_LABELS.get(i, i) for i in by_class.index]
        st.dataframe(by_class, width="stretch")
    with right:
        st.markdown("**Passages dans le temps**")
        bucket = 5 if df["Temps (s)"].max() > 20 else 2
        timeline = (df.assign(Intervalle=(df["Temps (s)"] // bucket * bucket).astype(int))
                      .pivot_table(index="Intervalle", columns="Sens", values="Piste",
                                   aggfunc="count", fill_value=0)
                      .reindex(columns=["IN", "OUT"], fill_value=0))
        st.bar_chart(timeline, color=["#2a78d6", "#eb6834"], x_label=f"Temps (s), par tranche de {bucket} s",
                     y_label="Passages", width="stretch")

    if res["video"]:
        st.markdown("**Vidéo annotée**")
        if res["playable"]:
            st.video(res["video"])
        else:
            st.caption("Lecture en ligne indisponible sur cet ordinateur : téléchargez la vidéo.")
        st.download_button("Télécharger la vidéo annotée", Path(res["video"]).read_bytes(),
                           file_name="trafficvision_annotee.mp4", mime="video/mp4")

    st.markdown("**Journal des passages**")
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Télécharger le journal (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="trafficvision_passages.csv", mime="text/csv")


def main() -> None:
    st.title("TrafficVision")
    st.caption("Détection, suivi et comptage d'objets à travers une ligne virtuelle "
               "(YOLO11 et ByteTrack).")

    with st.sidebar:
        st.header("Paramètres")
        model_label = st.selectbox("Modèle", list(MODELS))
        classes = st.multiselect("Objets à compter", list(CLASS_LABELS),
                                 default=["car", "motorcycle", "bus", "truck"],
                                 format_func=CLASS_LABELS.get)
        conf = st.slider("Seuil de confiance", 0.10, 0.90, 0.30, 0.05,
                         help="Plus il est bas, plus on détecte d'objets, au risque de fausses détections.")
        with st.expander("Options avancées"):
            tracker = st.selectbox("Suivi", ["bytetrack.yaml", "botsort.yaml"],
                                   format_func=lambda t: "ByteTrack" if t.startswith("byte") else "BoT-SORT")
            imgsz = st.select_slider("Taille d'inférence", [480, 640, 960, 1280], value=640,
                                     help="Plus elle est grande, mieux on voit les objets lointains, mais c'est plus lent.")

    upload = st.file_uploader("Vidéo à analyser", type=ALLOWED_TYPES)
    if upload is None:
        st.info("Envoyez une vidéo (mp4, avi, mov ou mkv) pour commencer. Une vue en hauteur ou "
                "de côté, où les véhicules traversent l'image, donne les meilleurs résultats.")
        return
    if not classes:
        st.warning("Choisissez au moins un type d'objet à compter.")
        return

    if st.session_state.get("upload_id") != upload.file_id:
        st.session_state.upload_id = upload.file_id
        st.session_state.pop("result", None)

    path = save_upload(upload)
    try:
        info = video_info(str(path))
    except ValueError:
        st.error("Cette vidéo ne peut pas être lue. Essayez un autre fichier ou un autre format.")
        return
    frame = first_frame(path)
    if frame is None:
        st.error("Impossible de lire la première image de la vidéo.")
        return
    duration = info.frames / info.fps if info.frames else 0

    st.subheader("1. Ligne de comptage")
    controls, preview = st.columns([1, 2])
    with controls:
        st.caption("Position de la ligne en pourcentage de l'image.")
        x1 = st.slider("Début, horizontal", 0, 100, 0)
        y1 = st.slider("Début, vertical", 0, 100, 60)
        x2 = st.slider("Fin, horizontal", 0, 100, 100)
        y2 = st.slider("Fin, vertical", 0, 100, 60)
        st.caption("Sens IN : franchissement vers le côté droit de la ligne, "
                   "en allant du début vers la fin.")
    line = (x1 / 100, y1 / 100, x2 / 100, y2 / 100)
    with preview:
        scale = max(frame.shape[1] / 1280, 1.0)
        st.image(draw_line(frame, line, int(4 * scale)), channels="BGR", width="stretch")

    st.subheader("2. Analyse")
    st.caption(f"Vidéo : {info.width} x {info.height}, {info.fps:.0f} images/s, {duration:.0f} s.")
    limit = int(min(duration, MAX_ANALYSIS_SECONDS)) if duration else MAX_ANALYSIS_SECONDS
    if limit > 5:
        seconds = st.slider("Durée analysée (secondes)", 5, limit, min(limit, 30),
                            help="Le calcul tourne sur processeur : une durée courte donne un résultat rapide.")
    else:
        seconds = max(limit, 1)
    if st.button("Lancer l'analyse", type="primary"):
        settings = Settings(model=MODELS[model_label], classes=classes, line=line, conf=conf,
                            imgsz=imgsz, tracker=tracker)
        st.session_state.result = analyse(path, settings, int(seconds * info.fps), info)

    if "result" in st.session_state:
        show_results(st.session_state.result)


main()
