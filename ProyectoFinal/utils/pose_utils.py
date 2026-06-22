import cv2
import numpy as np
import mediapipe as mp

_vision = mp.tasks.vision
_du     = _vision.drawing_utils
_PoseLandmark = _vision.PoseLandmark
_PoseConn     = _vision.PoseLandmarksConnections.POSE_LANDMARKS

PALETAS = {
    "Neón morado": {"lines": (180,120,255), "joints": (230,200,255), "glow": (120,60,200)},
    "Matrix verde": {"lines": (0,220,80),   "joints": (100,255,150), "glow": (0,140,50)},
    "Fuego ámbar":  {"lines": (0,140,255),  "joints": (0,200,255),   "glow": (0,80,200)},
    "Cyber rosa":   {"lines": (180,60,210), "joints": (220,130,240), "glow": (120,30,160)},
}

EXERCISES = {
    "Curl de bíceps (Izquierdo)": {
        "joints": (int(_PoseLandmark.LEFT_SHOULDER),
                   int(_PoseLandmark.LEFT_ELBOW),
                   int(_PoseLandmark.LEFT_WRIST)),
        "up_threshold":   55,    # ángulo < 55 → "arriba" (brazo doblado)
        "down_threshold": 155,   # ángulo > 155 → "abajo" (brazo extendido)
        "up_label":   "arriba",
        "down_label": "abajo",
    },
    "Curl de bíceps (Derecho)": {
        "joints": (int(_PoseLandmark.RIGHT_SHOULDER),
                   int(_PoseLandmark.RIGHT_ELBOW),
                   int(_PoseLandmark.RIGHT_WRIST)),
        "up_threshold":   55,
        "down_threshold": 155,
        "up_label":   "arriba",
        "down_label": "abajo",
    },
    "Sentadilla": {
        "joints": (int(_PoseLandmark.LEFT_HIP),
                   int(_PoseLandmark.LEFT_KNEE),
                   int(_PoseLandmark.LEFT_ANKLE)),
        "up_threshold":   155,   # ángulo > 155 → "de pie"
        "down_threshold": 115,   # ángulo < 115 → "abajo"
        "up_label":   "de pie",
        "down_label": "abajo",
    },
    "Flexión de hombro": {
        "joints": (int(_PoseLandmark.LEFT_HIP),
                   int(_PoseLandmark.LEFT_SHOULDER),
                   int(_PoseLandmark.LEFT_ELBOW)),
        "up_threshold":   155,
        "down_threshold": 35,
        "up_label":   "arriba",
        "down_label": "abajo",
    },
}


def draw_pose_basic(frame, landmarks):
    _du.draw_landmarks(
        frame, landmarks, _PoseConn,
        landmark_drawing_spec=_du.DrawingSpec(color=(0,200,100), thickness=2, circle_radius=3),
        connection_drawing_spec=_du.DrawingSpec(color=(200,200,200), thickness=2),
    )


def _angle_between(a, b, c):
    va = np.array([a.x, a.y])
    vb = np.array([b.x, b.y])
    vc = np.array([c.x, c.y])
    ba = va - vb
    bc = vc - vb
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def count_exercise_reps(landmarks, exercise_name, state):
    """
    Cuenta repeticiones completas usando una máquina de estados simple:
      - Estado NEUTRO al inicio
      - Transición a DOWN cuando el ángulo cruza down_threshold
      - Transición a UP   cuando el ángulo cruza up_threshold (viniendo de DOWN)
      - Solo ahí se incrementa el contador
    Esto evita contar frames individuales.
    """
    cfg = EXERCISES.get(exercise_name, EXERCISES["Curl de bíceps (Izquierdo)"])
    a_idx, b_idx, c_idx = cfg["joints"]
    angle = _angle_between(
        landmarks[a_idx], landmarks[b_idx], landmarks[c_idx]
    )

    stage = state.get("stage")
    count = state.get("count", 0)

    ex = exercise_name

    if ex == "Sentadilla":
        # Sentadilla: DOWN cuando rodilla < 100°, UP cuando > 155°
        if angle < cfg["down_threshold"] and stage != "abajo":
            stage = "abajo"
        elif angle > cfg["up_threshold"] and stage == "abajo":
            stage = "de pie"
            count += 1
    else:
        # Curl / Flexión: DOWN cuando ángulo > down_threshold, UP cuando < up_threshold
        if angle > cfg["down_threshold"] and stage != "abajo":
            stage = "abajo"
        elif angle < cfg["up_threshold"] and stage == "abajo":
            stage = "arriba"
            count += 1

    info = (f"✅ {exercise_name} | ángulo: {angle:.1f}° | "
            f"estado: {stage or 'inicio'} | reps: {count}")
    return str(count), stage, angle, info


def draw_pose_art(frame, landmarks, estilo="Neón morado"):
    h, w = frame.shape[:2]
    paleta = PALETAS.get(estilo, PALETAS["Neón morado"])

    def to_px(lmk):
        return int(lmk.x * w), int(lmk.y * h)

    for conn in _PoseConn:
        pt_a = to_px(landmarks[conn.start])
        pt_b = to_px(landmarks[conn.end])
        cv2.line(frame, pt_a, pt_b, paleta["glow"], 6)
        cv2.line(frame, pt_a, pt_b, paleta["lines"], 2)

    for lmk in landmarks:
        px = to_px(lmk)
        cv2.circle(frame, px, 6, paleta["glow"], -1)
        cv2.circle(frame, px, 3, paleta["joints"], -1)
