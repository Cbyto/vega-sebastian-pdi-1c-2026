# 🤸 PoseVision — Detección de Pose y Manos con MediaPipe

> Trabajo Práctico Final · Materia: Procesamiento Digital de Imágenes  
> Deploy en producción: [Hugging Face Spaces](https://cbyto-pdi-posevision.hf.space/) · Interfaz: Gradio

---

## 📌 Descripción del proyecto

**PoseVision** es una aplicación web interactiva que utiliza [MediaPipe](https://mediapipe.dev/) para detectar y analizar la pose del cuerpo humano y los gestos de las manos en tiempo real. La aplicación acepta entrada desde **webcam** o **archivos de video cargados**, y ofrece cuatro modos de uso independientes accesibles desde una interfaz de pestañas construida con **Gradio**.

El proyecto fue desarrollado como trabajo práctico final en el marco de la cursada de Visión por Computadora, con el objetivo de aplicar conocimientos de detección de pose usando modelos pre-entrenados, procesamiento de video con OpenCV, y despliegue de aplicaciones de machine learning en la nube.

---

## 🧩 Modos de la aplicación

### 1. 📐 Pose básica
Detecta el esqueleto completo del cuerpo humano usando el modelo **MediaPipe Pose**, que identifica **33 landmarks** (puntos de referencia anatómicos) como hombros, codos, caderas, rodillas y tobillos. Dibuja las conexiones entre articulaciones y muestra el score de confianza promedio de la detección.

### 2. 💪 Contador de ejercicios
Cuenta repeticiones de ejercicios en tiempo real calculando el **ángulo entre tres articulaciones** mediante geometría vectorial. Detecta automáticamente las fases del movimiento (subida / bajada) y actualiza el contador cuando se completa un ciclo.

Ejercicios soportados:
- **Curl de bíceps** — ángulo en el codo (hombro → codo → muñeca)
- **Sentadilla** — ángulo en la rodilla (cadera → rodilla → tobillo)
- **Flexión de hombro** — ángulo en el hombro (cadera → hombro → codo)

### 3. ✋ Conteo de dedos
Utiliza **MediaPipe Hands** para detectar hasta dos manos simultáneamente (21 landmarks por mano) y determina cuántos dedos están extendidos comparando la posición del tip de cada dedo con su nudillo (PIP). Muestra una leyenda descriptiva para cada combinación (puño, dos dedos, mano abierta, etc.).

### 4. 🎨 Modo Arte
Renderiza el esqueleto de MediaPipe Pose sobre un **fondo completamente oscuro** con un efecto visual de tipo neón/glow simulado con OpenCV. El usuario puede elegir entre cuatro paletas de color:
- Neón morado
- Matrix verde
- Fuego ámbar
- Cyber rosa

---

## 🛠️ Tecnologías utilizadas

| Tecnología | Rol |
|---|---|
| [MediaPipe](https://mediapipe.dev/) | Detección de pose y manos |
| [OpenCV](https://opencv.org/) | Procesamiento de video y dibujo |
| [Gradio](https://gradio.app/) | Interfaz web interactiva |
| [NumPy](https://numpy.org/) | Cálculo de ángulos y operaciones matriciales |
| [Python 3.10](https://python.org) | Lenguaje base |
| [Docker](https://docker.com) | Contenerización y reproducibilidad |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Deploy en la nube |

---

## 🚀 Cómo ejecutar el proyecto

### Opción A — Ejecución local con Python

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/posevision.git
cd posevision

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la aplicación
python app.py
```

La app quedará disponible en `http://localhost:7860`

---

### Opción B — Ejecución con Docker

```bash
# 1. Construir la imagen
docker build -t posevision .

# 2. Ejecutar el contenedor
docker run -p 7860:7860 posevision
```

La app quedará disponible en `http://localhost:7860`

> **Nota sobre webcam y Docker:** El acceso a la cámara web desde dentro de un contenedor requiere pasar el dispositivo al contenedor. En Linux:
> ```bash
> docker run -p 7860:7860 --device=/dev/video0:/dev/video0 posevision
> ```
> En Windows/Mac se recomienda usar la versión local directamente para la función de webcam.

---

### Opción C — Hugging Face Spaces (deploy en la nube)

La aplicación está desplegada y disponible públicamente en:  
👉 **[Hugging Face](https://cbyto-pdi-posevision.hf.space/)**

No requiere instalación. Se puede usar directamente desde el navegador.

---

## 📁 Estructura del repositorio

```
posevision/
│
├── app.py                  # Aplicación principal (Gradio + lógica de cada modo)
├── download_modesl.py      # Opcional para la descarga de los modelos previamente (la función _ensure_models() lo hace al arrancar)
├── requirements.txt        # Dependencias Python
├── Dockerfile              # Imagen Docker para contenerización
├── .dockerignore           # Archivos excluidos de la imagen Docker
├── .gitignore
├── README.md
│
└── utils/
    ├── pose_utils.py       # Dibujo de pose, cálculo de ángulos, modo arte
    └── hand_utils.py       # Detección de manos, conteo de dedos
```

---

## 🔬 Decisiones técnicas relevantes

**¿Por qué MediaPipe y no OpenPose u otros?**  
MediaPipe ofrece modelos pre-entrenados muy livianos que corren en CPU sin GPU, lo que es ideal para deploy en Hugging Face Spaces (entorno sin GPU gratuito). Su API de Python es simple y bien documentada.

**¿Por qué `opencv-python-headless`?**  
La versión `headless` de OpenCV no incluye dependencias de GUI (Qt, GTK), lo que reduce considerablemente el tamaño de la imagen Docker y evita errores en entornos sin display, como los servidores de Hugging Face.

**Contador de ejercicios — lógica de ángulos**  
El conteo se basa en detectar el cruce de dos umbrales de ángulo (umbral "abajo" y umbral "arriba"). Solo se cuenta una repetición cuando se completa el ciclo completo: abajo → arriba. Esto evita falsos positivos por movimientos parciales.

**Conteo de dedos — heurística**  
Para los cuatro dedos largos, un dedo se considera extendido cuando su tip (punta) tiene una coordenada Y menor que su PIP (nudillo medio), ya que en la imagen Y crece hacia abajo. Para el pulgar se usa la coordenada X como proxy de extensión lateral.

---

## 🔮 Trabajo futuro

Una extensión natural de este proyecto sería la implementación de **reconocimiento de lenguaje de señas (LSA — Lengua de Señas Argentina)**. Hay dos niveles de complejidad posibles:

- **Señas estáticas (letras A-Z):** Se puede entrenar un clasificador (por ejemplo, una red neuronal densa o un SVM) usando como features los 21 landmarks de la mano normalizados. El dataset puede generarse capturando imágenes con la propia app. Esto es factible como extensión directa del Modo 3.

- **Señas dinámicas (palabras y frases):** Requiere analizar secuencias temporales de landmarks. Una arquitectura típica combina MediaPipe Hands con un modelo LSTM o Transformer para clasificar gestos en el tiempo. Este enfoque implica recolección de datos temporales, mayor complejidad de entrenamiento y mayor latencia de inferencia.

La infraestructura actual del proyecto (streaming de webcam, procesamiento frame a frame, interfaz Gradio) ya está preparada para soportar esta extensión sin cambios estructurales importantes.

---

## 👤 Autor

Desarrollado por **[Sebastián Vega]**

---

## 📄 Licencia

MIT License — libre para uso académico y personal.
