import os
import urllib.request
import cv2
import numpy as np
import gradio as gr
import mediapipe as mp
import pandas as pd
from datetime import datetime
import time
from utils.pose_utils import draw_pose_basic, count_exercise_reps, draw_pose_art, EXERCISES, PALETAS
from utils.hand_utils  import draw_hands, count_fingers, _dedos_label

# ── API MediaPipe Tasks ─────────────────────────────────────────────────────
_vision            = mp.tasks.vision
BaseOptions        = mp.tasks.BaseOptions
PoseLandmarker     = _vision.PoseLandmarker
PoseLandmarkerOpts = _vision.PoseLandmarkerOptions
HandLandmarker     = _vision.HandLandmarker
HandLandmarkerOpts = _vision.HandLandmarkerOptions
RunningMode        = _vision.RunningMode

# ── Descarga de modelos ─────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BASE_URL   = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "pose_landmarker.task": f"{BASE_URL}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

def _ensure_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for fname, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, fname)
        if not os.path.exists(dest):
            print(f"Descargando {fname}...")
            urllib.request.urlretrieve(url, dest)

_ensure_models()

POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker.task")
HAND_MODEL = os.path.join(MODELS_DIR, "hand_landmarker.task")

# ── Estado global ejercicios ────────────────────────────────────────────────
rep_state = {"count": 0, "stage": None}

# ── Helpers ─────────────────────────────────────────────────────────────────
def _pose_det():
    return PoseLandmarker.create_from_options(PoseLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
    ))

def _hand_det():
    return HandLandmarker.create_from_options(HandLandmarkerOpts(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.5,
    ))

def _to_bgr(frame):
    frame = np.array(frame)
    if frame.ndim == 2:          return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:      return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def _mp_img(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

# ── Procesadores de frame único ─────────────────────────────────────────────
def _frame_pose_basic(bgr):
    with _pose_det() as det:
        r = det.detect(_mp_img(bgr))
        out = bgr.copy()
        if r.pose_landmarks:
            lms = r.pose_landmarks[0]
            draw_pose_basic(out, lms)
            conf = np.mean([lm.visibility for lm in lms])
            return out, f"✅ Pose detectada — confianza: {conf:.2f}"
        return out, "❌ No se detectó pose."

def _frame_exercise(bgr, exercise):
    with _pose_det() as det:
        r = det.detect(_mp_img(bgr))
        out = bgr.copy()
        if r.pose_landmarks:
            lms = r.pose_landmarks[0]
            reps, stage, angle, info = count_exercise_reps(lms, exercise, rep_state)
            rep_state["stage"] = stage
            rep_state["count"] = int(reps)
            draw_pose_basic(out, lms)
            cv2.putText(out, f"Reps: {reps}",       (20,  50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,100), 3)
            cv2.putText(out, f"Estado: {stage}",    (20,  95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,200,0),  2)
            cv2.putText(out, f"Angulo: {angle:.1f}",(20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200),2)
            return out, info, str(rep_state["count"])
        return out, "❌ No se detectó pose.", str(rep_state["count"])

def _frame_hands(bgr):
    with _hand_det() as det:
        r = det.detect(_mp_img(bgr))
        out = bgr.copy()
        if r.hand_landmarks:
            # Filtrar detecciones duplicadas de la misma mano.
            # MediaPipe a veces detecta una mano dos veces cuando está cerca
            # de la cámara. Descartamos detecciones cuya muñeca (landmark 0)
            # esté a menos de 0.15 de distancia normalizada de otra ya aceptada.
            hands_ok = []
            for h in r.hand_landmarks:
                wrist = h[0]
                duplicado = False
                for prev in hands_ok:
                    d = ((wrist.x - prev[0].x)**2 + (wrist.y - prev[0].y)**2)**0.5
                    if d < 0.15:
                        duplicado = True
                        break
                if not duplicado:
                    hands_ok.append(h)

            total = sum(count_fingers(h) for h in hands_ok)
            for h in hands_ok:
                draw_hands(out, h)

            label = _dedos_label(total)
            # Mostrar cuántas manos se detectaron
            n_manos = len(hands_ok)
            manos_str = f"{'Una mano' if n_manos == 1 else f'{n_manos} manos'}"
            cv2.putText(out, manos_str, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(out, label,     (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,220,0),  3)
            return out, f"{label} ({manos_str})"
        return out, "✋ No se detectaron manos."

def _frame_arte(bgr, estilo):
    with _pose_det() as det:
        r = det.detect(_mp_img(bgr))
        out = np.zeros_like(bgr)
        if r.pose_landmarks:
            draw_pose_art(out, r.pose_landmarks[0], estilo)
            return out, f"🎨 {estilo}"
        return out, "❌ No se detectó pose."

# ── Wrappers para webcam (devuelven RGB) ────────────────────────────────────
def pose_webcam(frame):
    if frame is None: return None, "Sin entrada."
    out, info = _frame_pose_basic(_to_bgr(frame))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), info

def exercise_webcam(frame, exercise):
    if frame is None: return None, "Sin entrada.", str(rep_state["count"])
    out, info, reps = _frame_exercise(_to_bgr(frame), exercise)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), info, reps

def hands_webcam(frame):
    if frame is None: return None, "Sin entrada."
    out, info = _frame_hands(_to_bgr(frame))
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), info

def arte_webcam(frame, estilo):
    if frame is None: return None, "Sin entrada."
    out, info = _frame_arte(_to_bgr(frame), estilo)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), info

# ── Procesadores de VIDEO ────────────────────────────────────────────────────
def _process_video(video_path, frame_fn, *args):
    """
    Procesa un video frame a frame aplicando frame_fn.
    Escribe con mp4v (raw) y luego recodifica a H264 con ffmpeg
    para garantizar compatibilidad con el navegador.
    """
    if video_path is None:
        return None, "Sin video."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "No se pudo abrir el video."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Archivo temporal (mp4v, no compatible con todos los navegadores)
    # tmp_path   = video_path + "_tmp.mp4"
    # Archivo final (H264, compatible con navegadores)
    # final_path = video_path + "_out.mp4"
    
    timestamp = int(time.time())
    tmp_path   = f"{video_path}_{timestamp}_tmp.mp4"
    final_path = f"{video_path}_{timestamp}_out.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

    last_info = ""
    frames_ok = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result    = frame_fn(frame, *args)
        processed = result[0]
        last_info = result[1] if len(result) > 1 else ""
        writer.write(processed)
        frames_ok += 1

    cap.release()
    writer.release()

    if frames_ok == 0:
        return None, "❌ No se procesaron frames."

    # Recodificar a H264 usando imageio-ffmpeg (trae su propio binario,
    # funciona en Windows/Linux/Mac sin instalar ffmpeg externamente)
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        import subprocess
        duration = frames_ok / fps
        ret = subprocess.run(
            [ffmpeg_exe, "-y",
             "-i", tmp_path,
             "-vcodec", "libx264",
             "-pix_fmt", "yuv420p",
             "-r", str(round(fps)),
             "-movflags", "+faststart",
             "-t", str(duration),
             final_path],
            capture_output=True
        )
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        if ret.returncode != 0:
            # ffmpeg falló — devolver el tmp igual (al menos se puede descargar)
            total_s = frames_ok / fps
            return tmp_path, f"⚠️ Video procesado pero sin recodificar. {frames_ok} frames · {total_s:.1f}s"
    except Exception:
        # imageio-ffmpeg no disponible — devolver mp4v igual
        final_path = tmp_path

    total_s = frames_ok / fps
    return final_path, f"✅ Video procesado — {frames_ok} frames · {total_s:.1f}s · {last_info}"

def pose_video(video_path):
    def fn(frame): return _frame_pose_basic(frame)
    return _process_video(video_path, fn)

def exercise_video(video_path, exercise):
    def fn(frame, ex): return _frame_exercise(frame, ex)[:2]
    return _process_video(video_path, fn, exercise)

def hands_video(video_path):
    def fn(frame): return _frame_hands(frame)
    return _process_video(video_path, fn)

def arte_video(video_path, estilo):
    def fn(frame, e): return _frame_arte(frame, e)
    return _process_video(video_path, fn, estilo)

def reset_reps():
    rep_state["count"] = 0
    rep_state["stage"] = None
    return "0", "🔄 Contador reiniciado."
    
# ── Excel ─────────────────────────────────────────────────────────
def exportar_reporte_excel(exercise):
    count = rep_state["count"]
    if count == 0:
        return None     # Si no hay repeticiones, no generamos nada
    
    # Armamos la estructura de datos
    data = {
        "Fecha y Hora": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Ejercicio": [exercise],
        "Repeticiones Completadas": [count]
    }
    
    df = pd.DataFrame(data)
    filepath = "reporte_rutina.xlsx"
    
    # Exportamos a Excel sin la columna de índices
    df.to_excel(filepath, index=False, engine='openpyxl')
    
    return filepath


# ── Interfaz Gradio ─────────────────────────────────────────────────────────
CSS = "#titulo{text-align:center} #subtitulo{text-align:center;color:#888;margin-top:-10px}"

with gr.Blocks(title="PoseVision") as demo:
    gr.Markdown("# 🤸 PoseVision", elem_id="titulo")
    gr.Markdown(
        "Detección de pose y manos con **MediaPipe** · Trabajo Práctico Final",
        elem_id="subtitulo",
    )

    with gr.Tabs():

        # ── Tab 1: Pose básica ──────────────────────────────────────────
        with gr.TabItem("📐 Pose básica"):
            gr.Markdown("### Detección de esqueleto (33 landmarks)")
            with gr.Tabs():
                with gr.TabItem("📷 Webcam"):
                    with gr.Row():
                        cam1 = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                        with gr.Column():
                            out1_cam  = gr.Image(label="Resultado")
                            info1_cam = gr.Textbox(label="Estado")
                    cam1.stream(pose_webcam, [cam1], [out1_cam, info1_cam])

                with gr.TabItem("🎬 Subir video"):
                    with gr.Row():
                        with gr.Column():
                            vid1    = gr.Video(label="Subir video (MP4, AVI, MOV)")
                            btn1    = gr.Button("▶ Procesar video")
                        with gr.Column():
                            out1_vid  = gr.Video(label="Video procesado")
                            info1_vid = gr.Textbox(label="Estado")
                    btn1.click(pose_video, [vid1], [out1_vid, info1_vid])

        # ── Tab 2: Ejercicios ───────────────────────────────────────────
        with gr.TabItem("💪 Contador de ejercicios"):
            gr.Markdown("### Conteo de repeticiones por ángulo articular")
            ex_sel = gr.Radio(list(EXERCISES.keys()), value="Curl de bíceps (Izquierdo)", label="Ejercicio")
            with gr.Tabs():
                with gr.TabItem("📷 Webcam"):
                    with gr.Row():
                        with gr.Column():
                            cam2 = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                            btn_reset = gr.Button("🔄 Reiniciar contador")
                            btn_export = gr.Button("📥 Descargar Reporte (Excel)")
                        with gr.Column():
                            out2_cam  = gr.Image(label="Resultado")
                            reps_cam  = gr.Textbox(label="Repeticiones", value="0")
                            info2_cam = gr.Textbox(label="Estado")
                            archivo_excel = gr.File(label="Tu Reporte")
                    cam2.stream(exercise_webcam, [cam2, ex_sel], [out2_cam, info2_cam, reps_cam])
                    btn_reset.click(reset_reps, [], [reps_cam, info2_cam])
                    
                    # Conectamos el botón de exportar con la función
                    btn_export.click(exportar_reporte_excel, inputs=[ex_sel], outputs=[archivo_excel])

                with gr.TabItem("🎬 Subir video"):
                    with gr.Row():
                        with gr.Column():
                            vid2  = gr.Video(label="Subir video lateral del ejercicio")
                            btn2  = gr.Button("▶ Procesar video")
                            btn_reset2 = gr.Button("🔄 Reiniciar contador")
                        with gr.Column():
                            out2_vid  = gr.Video(label="Video procesado")
                            info2_vid = gr.Textbox(label="Estado")
                    btn2.click(exercise_video, [vid2, ex_sel], [out2_vid, info2_vid])
                    btn_reset2.click(reset_reps, [], [gr.Textbox(visible=False), info2_vid])

        # ── Tab 3: Dedos ────────────────────────────────────────────────
        with gr.TabItem("✋ Conteo de dedos"):
            gr.Markdown("### Detección de manos — hasta 10 dedos simultáneos")
            with gr.Tabs():
                with gr.TabItem("📷 Webcam"):
                    with gr.Row():
                        cam3 = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                        with gr.Column():
                            out3_cam  = gr.Image(label="Resultado")
                            info3_cam = gr.Textbox(label="Dedos detectados")
                    cam3.stream(hands_webcam, [cam3], [out3_cam, info3_cam])

                with gr.TabItem("🎬 Subir video"):
                    with gr.Row():
                        with gr.Column():
                            vid3  = gr.Video(label="Subir video")
                            btn3  = gr.Button("▶ Procesar video")
                        with gr.Column():
                            out3_vid  = gr.Video(label="Video procesado")
                            info3_vid = gr.Textbox(label="Estado")
                    btn3.click(hands_video, [vid3], [out3_vid, info3_vid])

            gr.Markdown(
                "**Tip:** palma mirando a la cámara · buena iluminación frontal · "
                "con dos manos detecta hasta 10 dedos."
            )

        # ── Tab 4: Modo Arte ────────────────────────────────────────────
        with gr.TabItem("🎨 Modo Arte"):
            gr.Markdown("### Esqueleto artístico sobre fondo oscuro")
            estilo_sel = gr.Radio(list(PALETAS.keys()), value="Neón morado", label="Estilo")
            with gr.Tabs():
                with gr.TabItem("📷 Webcam"):
                    with gr.Row():
                        cam4 = gr.Image(sources=["webcam"], streaming=True, label="Webcam")
                        with gr.Column():
                            out4_cam  = gr.Image(label="Resultado")
                            info4_cam = gr.Textbox(label="Estado")
                    cam4.stream(arte_webcam, [cam4, estilo_sel], [out4_cam, info4_cam])

                with gr.TabItem("🎬 Subir video"):
                    with gr.Row():
                        with gr.Column():
                            vid4  = gr.Video(label="Subir video")
                            btn4  = gr.Button("▶ Procesar video")
                        with gr.Column():
                            out4_vid  = gr.Video(label="Video procesado")
                            info4_vid = gr.Textbox(label="Estado")
                    btn4.click(arte_video, [vid4, estilo_sel], [out4_vid, info4_vid])

    gr.Markdown(
        "---\n[MediaPipe](https://mediapipe.dev) · [Gradio](https://gradio.app) · "
        "[Hugging Face Spaces](https://huggingface.co/spaces)"
    )

if __name__ == "__main__":
    ## demo.launch(css=CSS)
    demo.launch(
    css=CSS,
    server_name="0.0.0.0",      # ← escucha en todas las interfaces
    server_port=7860,           # ← puerto explícito
)