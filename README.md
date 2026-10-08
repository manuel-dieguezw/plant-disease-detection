# Clasificación de enfermedades en hojas de plantas

Trabajo final de Visión por Computadora II (CEIA, FIUBA). Comparación de arquitecturas preentrenadas (transfer learning) para clasificar enfermedades en hojas, usando el dataset [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease).

## Dataset

15 clases (Pepper, Potato, Tomato), ~20.600 imágenes de 256×256. No está incluido en el repo por su peso: descargarlo y descomprimirlo en `PlantVillage/` en la raíz del proyecto, con una subcarpeta por clase.

## Notebooks

1. **`01_EDA_y_augmentation.ipynb`**: análisis exploratorio, detección de duplicados, split estratificado (`splits/split.csv`) y pipeline de data augmentation.
2. **`02_comparacion_arquitecturas.ipynb`**: comparación de backbones preentrenados (MobileNetV3, EfficientNet-B0, ResNet18) con linear probe sobre embeddings.
3. **`03_finetuning_transfer_learning.ipynb`**: fine-tuning parcial de los mejores modelos, con y sin augmentation.

`notebooks/common.py` tiene el código compartido (datos, transforms, entrenamiento, métricas).

## Instalación

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Requiere Python 3.11+. Los notebooks se corren en orden: el 01 genera el split que usan el 02 y el 03.

## Demo

`app.py` levanta una interfaz web (Gradio) para clasificar imágenes de hojas con el modelo final (EfficientNet-B0, corrida `effb0_aug`). Muestra el top-5 de clases y el mapa Grad-CAM de la clase predicha o de la que se elija.

### Requisitos

- Dependencias instaladas (ver [Instalación](#instalación)); incluye `gradio`.
- El checkpoint del modelo final en `cache/ckpt_full/effb0_aug.pt`, que genera el notebook 03 (corrida en modo full). Sin este archivo la aplicación no arranca.
- Opcional, para la galería de ejemplos: el dataset en `PlantVillage/` (se usan imágenes del split de test) y las imágenes de PlantDoc en `data/real_world/` (las descarga el notebook 04). Si no están, la aplicación funciona igual pero sin ejemplos precargados.

### Ejecución

Desde la raíz del repositorio:

```bash
python app.py
```

Cuando la consola muestra `Running on local URL: http://127.0.0.1:7860`, abrir esa dirección en el navegador. Para detener la aplicación, `Ctrl+C` en la consola.

### Uso

1. Cargar una imagen de hoja: arrastrarla, subirla desde el disco, tomarla con la webcam o pegarla desde el portapapeles. También se puede hacer clic en una de las imágenes de ejemplo.
2. La clasificación se ejecuta automáticamente (o con el botón **Clasificar**) y muestra:
   - las 5 clases más probables con su probabilidad;
   - el mapa Grad-CAM superpuesto a la imagen (rojo: regiones que más pesaron en la decisión);
   - la latencia de inferencia.
3. **Explicar la clase** permite ver el Grad-CAM de una clase distinta a la predicha, y **Opacidad del Grad-CAM** regula la transparencia del mapa.

El modelo sólo reconoce las 15 clases de PlantVillage (pimiento, papa y tomate). Con imágenes de otras especies, o tomadas en condiciones muy distintas a las del dataset (fondo complejo, varias hojas), las predicciones pierden confiabilidad.

## Resultados

Se guardan en `results/` (CSV/JSON) a medida que se corren los notebooks; `cache/` guarda embeddings y checkpoints intermedios (no versionado, se regenera solo).
