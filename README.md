# Clasificación de enfermedades en hojas de plantas

Trabajo final de Visión por Computadora II (CEIA, FIUBA). Desarrollamos un prototipo para clasificar enfermedades en hojas de pimiento, papa y tomate del dataset [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease), usando arquitecturas livianas preentrenadas en ImageNet (transfer learning) y entrenando íntegramente en CPU. Además del desempeño, se evalúa la robustez ante perturbaciones, la explicabilidad con Grad-CAM y la generalización a imágenes de campo (PlantDoc).

El informe final, en formato IEEE, está en [`paper/paper_final.pdf`](paper/paper_final.pdf).

**Integrantes:** Santiago Federico Bettig, Manuel Dieguez, Pedro Ignacio Rubiolo, Lourdes Tolotto.

## Resultados principales

| Modelo | Etapa | Test macro-F1 |
|---|---|---|
| ResNet18 | linear probe | 0,942 |
| MobileNetV3-Small | linear probe | 0,958 |
| MobileNetV3-Large | linear probe | 0,960 |
| EfficientNet-B0 | linear probe | 0,961 |
| MobileNetV3-Small (sin / con augmentation) | fine-tuning parcial | 0,987 / 0,985 |
| EfficientNet-B0 (sin / con augmentation) | fine-tuning parcial | 0,990 / **0,993** |

- **Modelo final: EfficientNet-B0 con data augmentation** (`effb0_aug`): 17 errores en 2.949 imágenes de test y 17,5 ms por imagen en CPU (MobileNetV3-Small: 7,1 ms).
- **Robustez:** el augmentation casi no cambia el test original, pero es decisivo ante perturbaciones. Con desenfoque, el macro-F1 de EfficientNet-B0 cae 0,23 sin augmentation y solo 0,013 con él.
- **Grad-CAM:** el 65,7 % de la activación recae sobre la hoja, que ocupa en promedio el 40,2 % de la imagen (pointing game: 0,965).
- **Evaluación externa:** sobre 30 imágenes de campo de PlantDoc, el accuracy cae al 36,7 %. Un desempeño cercano al 100 % en PlantVillage no garantiza utilidad en condiciones reales, por lo que el prototipo no debería usarse para diagnosticar en el campo.

## Dataset

15 clases (Pepper, Potato, Tomato), ~20.600 imágenes de 256×256. No está incluido en el repo por su peso: descargarlo y descomprimirlo en `PlantVillage/` en la raíz del proyecto, con una subcarpeta por clase. También se puede indicar otra ubicación con la variable de entorno `PV_DATA_DIR`.

El split se hace de forma estratificada y agrupada por imágenes casi duplicadas (dHash), para reducir la fuga de datos entre train, val y test. Queda versionado en `splits/split.csv`.

## Notebooks

Están en `notebooks/` y se corren en orden; cada uno usa lo que generan los anteriores.

1. **`01_EDA_y_augmentation.ipynb`**: análisis exploratorio, detección de duplicados, split estratificado (`splits/split.csv`) y pipeline de data augmentation.
2. **`02_comparacion_arquitecturas.ipynb`**: comparación de backbones preentrenados (MobileNetV3-Small, MobileNetV3-Large, EfficientNet-B0, ResNet18) con linear probe sobre embeddings.
3. **`03_finetuning_transfer_learning.ipynb`**: fine-tuning parcial de MobileNetV3-Small y EfficientNet-B0, con y sin augmentation (4 corridas). Con `FT_QUICK=1` hace una prueba rápida que guarda en `results/*_quick/`.
4. **`04_analisis_robustez_con_evaluacion_externa.ipynb`**: robustez de los cuatro modelos ante perturbaciones del test (oscurecer, aclarar, rotación, zoom, desenfoque) y evaluación externa sobre imágenes de PlantDoc, que descarga en `data/real_world/`.
5. **`05_gradcam.ipynb`**: Grad-CAM sobre el modelo final, análisis cuantitativo de la activación sobre la hoja (energía y pointing game), análisis de errores y medición de latencia en CPU.
6. **`06_modelo_final_y_demo.ipynb`**: prototipo de inferencia con el modelo final sobre imágenes individuales (clase, confianza y top-3).

`notebooks/common.py` tiene el código compartido (datos, transforms, entrenamiento, métricas).

> **Importante:** no volver a correr el notebook 01 completo, porque regeneraría `splits/split.csv` y dejaría inválidos los resultados del resto.

## Instalación

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Requiere Python 3.11+. El índice de PyTorch indicado instala la versión para CPU; con GPU, el notebook 06 usa CUDA automáticamente si está disponible.

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

## Estructura del repositorio

```
├── app.py                 # demo interactiva (Gradio + Grad-CAM)
├── notebooks/             # notebooks 01–06 y common.py
├── splits/                # split train/val/test y mapeo clase → índice
├── results/
│   ├── linear_probe.csv   # notebook 02
│   ├── finetune_full/     # notebook 03: métricas, logs y logits de las 4 corridas
│   ├── robustness/        # notebook 04: perturbaciones y PlantDoc
│   └── gradcam/           # notebook 05: activación sobre la hoja y latencia
├── paper/                 # informe IEEE (LaTeX, figuras y PDF final)
└── requirements.txt
```

`cache/` (embeddings y checkpoints) y `data/` no están versionados: se regeneran al correr los notebooks.

## Paper

El código LaTeX está en `paper/` (clase IEEEtran). Para compilarlo desde esa carpeta:

```bash
latexmk -pdf main.tex
```
