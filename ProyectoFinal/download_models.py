"""
Script auxiliar: descarga los modelos .task de MediaPipe al directorio models/.
Se ejecuta automáticamente en el primer arranque de la app si los modelos no existen.
"""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
BASE_URL = "https://storage.googleapis.com/mediapipe-models"

MODELS = {
    "pose_landmarker.task": f"{BASE_URL}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": f"{BASE_URL}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(dest):
            print(f"Descargando {filename}...")
            urllib.request.urlretrieve(url, dest)
            print(f"  ✅ {filename} descargado.")
        else:
            print(f"  ✓ {filename} ya existe.")

if __name__ == "__main__":
    download_models()
