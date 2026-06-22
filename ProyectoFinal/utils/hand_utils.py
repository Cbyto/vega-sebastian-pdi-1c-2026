import cv2
import numpy as np
import mediapipe as mp

_vision   = mp.tasks.vision
_du       = _vision.drawing_utils
_HandConn = _vision.HandLandmarksConnections.HAND_CONNECTIONS

WRIST      = 0
THUMB_CMC  = 1
THUMB_MCP  = 2
THUMB_IP   = 3
THUMB_TIP  = 4
INDEX_MCP  = 5
INDEX_PIP  = 6
INDEX_TIP  = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP   = 13
RING_PIP   = 14
RING_TIP   = 16
PINKY_MCP  = 17
PINKY_PIP  = 18
PINKY_TIP  = 20

FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def _dist(a, b) -> float:
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _angle(a, b, c) -> float:
    """Ángulo en el punto b entre los segmentos b→a y b→c."""
    va = np.array([a.x - b.x, a.y - b.y])
    vc = np.array([c.x - b.x, c.y - b.y])
    cos = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def count_fingers(landmarks) -> int:
    """
    Dedos 2-5: extendido si TIP está más lejos de la muñeca que su MCP * 1.2.

    Pulgar — criterio doble (ambas condiciones deben cumplirse):
      1. Ángulo IP (MCP→IP→TIP) >= 150°  →  dedo recto, no doblado
      2. TIP más lejos de WRIST que THUMB_MCP * 1.1
         →  el TIP no está metido adentro de la palma

    El criterio doble resuelve el caso donde el pulgar cerrado
    queda orientado de costado a la cámara (ángulo grande pero TIP
    cerca de la palma).
    """
    wrist = landmarks[WRIST]

    # ── 4 dedos largos ────────────────────────────────────────────────────
    count = 0
    # Comparamos la punta (TIP) contra la articulación media (PIP)
    for mcp_idx, tip_idx, pip_idx in zip(FINGER_MCPS, FINGER_TIPS, FINGER_PIPS):
        # Ángulo en el PIP (entre el nudillo MCP y la punta TIP)
        ang = _angle(landmarks[mcp_idx], landmarks[pip_idx], landmarks[tip_idx])
        
        # Un dedo perfectamente recto tiene 180°. 
        # Usamos 150° como umbral de tolerancia para considerarlo "estirado".
        if ang >= 150:
            count += 1

    # ── Pulgar (criterio doble) ───────────────────────────────────────────
    angle_ip   = _angle(landmarks[THUMB_MCP], landmarks[THUMB_IP], landmarks[THUMB_TIP])
    d_tip      = _dist(landmarks[THUMB_TIP], wrist)
    d_thumb_mcp = _dist(landmarks[THUMB_MCP], wrist)

    thumb_straight  = angle_ip >= 160          # dedo no doblado
    thumb_out       = d_tip > d_thumb_mcp * 1.1  # TIP no metido en la palma

    if thumb_straight and thumb_out:
        count += 1

    return count


def _dedos_label(n: int) -> str:
    labels = {
        0:  "✊ Puño cerrado",
        1:  " ☝  Un dedo",
        2:  "✌️ Dos dedos",
        3:  "🤟 Tres dedos",
        4:  "🖖 Cuatro dedos",
        5:  "🖐️ Mano abierta (5)",
        6:  "6 dedos",
        7:  "7 dedos",
        8:  "8 dedos",
        9:  "9 dedos",
        10: "🙌 ¡Diez dedos!",
    }
    return labels.get(n, f"👐 {n} dedos")


def draw_hands(frame, landmarks):
    _du.draw_landmarks(
        frame, landmarks, _HandConn,
        landmark_drawing_spec=_du.DrawingSpec(color=(0, 220, 255), thickness=2, circle_radius=4),
        connection_drawing_spec=_du.DrawingSpec(color=(255, 200, 0), thickness=2),
    )
