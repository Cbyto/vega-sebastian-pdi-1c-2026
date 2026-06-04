# Trabajo Práctico 006 — Fotografía Digital

## De la cámara oscura a la imagen intencional

## Descripción del proyecto

Este repositorio contiene el desarrollo del Trabajo Práctico 006, centrado en el proceso de construcción de una imagen fotográfica desde tres dimensiones complementarias:

- el funcionamiento óptico de la cámara oscura;
- la toma de decisiones compositivas en fotografía digital;
- el postprocesamiento de imágenes mediante Python y OpenCV.

El objetivo del trabajo no es solamente producir imágenes estéticamente agradables, sino analizar cómo una fotografía se construye a partir de decisiones técnicas y visuales: qué mostrar, qué excluir, desde dónde mirar y cómo modificar la luz para reforzar una intención.

---

## Estructura del repositorio

```text
006_fotografia_digital/
│
├── README.md
├── presentacion.pdf
│
├── imagenes/
│   ├── originales/
│   ├── procesadas/
│   └── descartes/
│
├── codigo/
│   ├── Fotografia_Digital.ipynb
│
└── recursos/
    └── referencias_opcionales/
```    
---

## Contenidos principales

### 1. Cámara oscura y procesamiento digital

Se construyó una cámara oscura utilizando cartulina, papel de aluminio y una pequeña abertura o estenopo. La imagen proyectada se forma a partir de la propagación rectilínea de la luz y aparece invertida sobre el plano de imagen.

A partir de una captura realizada con este dispositivo, se aplicó un procesamiento digital:

1. carga de imagen original;
2. rotación de la imagen para corregir la inversión propia de la cámara oscura;
3. recorte de bordes negros para trabajar sobre la zona útil;
4. conversión de RGB a HSV;
5. separación de canales H, S y V;
6. ecualización únicamente del canal V;
7. recomposición de la imagen;
8. comparación visual e histogramas antes/después.

La decisión de ecualizar el canal V se debe a que este canal representa la luminosidad. De esta forma se mejora el contraste sin intervenir directamente sobre el matiz ni la saturación de la imagen.

---

### 2. Simplicidad visual

Se trabajó con una fotografía donde el objetivo fue identificar claramente el sujeto principal. Para simplificar la lectura visual, la imagen fue convertida a escala de grises.

La eliminación del color permite concentrar la atención en:

- la forma;
- la textura;
- el contraste;
- la relación entre sujeto y fondo.

Esta decisión reduce la información cromática y evita que elementos secundarios compitan con el motivo principal.

---

### 3. Reencuadre y reinterpretación

Se partió de una fotografía amplia de una escena urbana nocturna y se produjeron dos recortes diferentes.

El objetivo fue demostrar que el encuadre modifica el sentido de una imagen. Al seleccionar distintas regiones de una misma fotografía, cambian:

- el sujeto principal;
- la información contextual disponible;
- la relación entre figura y entorno

El reencuadre permite comprobar que fotografiar también implica excluir información.

---

### 4. Punto de vista y construcción narrativa

Se fotografió un mismo sujeto/objeto desde dos puntos de vista distintos:

- una vista cenital, que organiza los objetos como si fueran parte de un mapa;
- una vista a nivel de ojos, que genera mayor cercanía e inmersión.

La comparación muestra cómo la posición de la cámara modifica la escala, el contexto y la información.

---

### 5. Fotografía basada en la luz

Se trabajó con una imagen donde la luz funciona como elemento estructural. La iluminación cálida permite construir una atmósfera íntima y dirigir la atención hacia el sujeto.

El análisis considera:

- dirección de la luz;
- contraste;
- sombras;
- volumen;
- dominante cromática cálida;
- separación entre sujeto y fondo.

También se incluyó un esquema simple de dirección de la luz y un análisis de canales RGB para observar la presencia dominante del canal rojo.

---

### 6. Selección crítica

El trabajo incluye una instancia de selección y descarte de imágenes. Se comparan miniaturas de fotografías descartadas con una imagen final elegida.

La selección se realizó considerando:

- claridad del sujeto;
- uso expresivo de la luz;
- reducción de distracciones;
- relación entre figura y fondo;
- coherencia con la intención visual.

Esta etapa permite entender que fotografiar también implica comparar, editar y decidir qué imagen comunica mejor.

---

### 7. Reflexión final

El proceso permitió comprender que fotografiar no es solamente registrar una escena, sino construir una imagen mediante decisiones ópticas, técnicas, compositivas y expresivas.

La cámara oscura muestra el origen óptico de la imagen: la luz se proyecta sobre un plano y forma una representación. Sin embargo, esa proyección necesita ser interpretada. La composición, el punto de vista, el recorte, la luz y el postprocesamiento transforman la captura inicial en una imagen con intención.

---

## Procesamiento digital realizado

El procesamiento fue desarrollado en Python utilizando principalmente OpenCV, NumPy y Matplotlib.

Operaciones principales:

- lectura de imágenes con OpenCV;
- conversión BGR → RGB para visualización correcta;
- conversión RGB → HSV;
- separación de canales H, S y V;
- ecualización del canal V;
- recomposición HSV → RGB;
- conversión a escala de grises;
- recortes mediante coordenadas;
- generación de histogramas;
- guardado de imágenes procesadas.

---

## Dependencias

Para ejecutar la notebook o los scripts se requiere tener instalado Python 3 y las siguientes librerías:

```bash
pip install opencv-python matplotlib numpy
```

---

## Ejecución

La notebook principal se encuentra en:

Para ejecutarla:

1. abrir la notebook en Jupyter Notebook, JupyterLab o Visual Studio Code;
2. verificar que las imágenes originales estén dentro de `imagenes/originales/`;
3. ejecutar las celdas en orden;
4. revisar las imágenes generadas en `imagenes/procesadas/`.

---

## Criterio general del trabajo

La idea central del proyecto es que una fotografía no es únicamente un registro automático del mundo. Es una construcción visual que depende de decisiones:

- qué mostrar;
- qué dejar afuera;
- desde dónde mirar;
- cómo usar la luz;
- cómo procesar la imagen;
- cómo seleccionar la versión final.