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

## Resultados

Se guardan en `results/` (CSV/JSON) a medida que se corren los notebooks; `cache/` guarda embeddings y checkpoints intermedios (no versionado, se regenera solo).
