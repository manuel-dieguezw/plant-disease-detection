"""Demo interactiva del modelo final (EfficientNet-B0 fine-tuneado, corrida effb0_aug) con Grad-CAM.

Uso (desde la raíz del repo):
    python app.py

Abre una interfaz web en http://127.0.0.1:7860. Requiere el checkpoint cache/ckpt_full/effb0_aug.pt
(generado por el notebook 03). Las imágenes de ejemplo se toman del split de test de PlantVillage
y de las imágenes de PlantDoc descargadas por el notebook 04, si están disponibles.
"""
import sys
import time
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parent / "notebooks"))
from common import CLASS_NAMES_ES, DATA_DIR, DEVICE, ROOT, load_split, make_transforms  # noqa: E402

FINAL_RUN = "effb0_aug"
IMG_SIZE_FT = 160
CKPT_PATH = ROOT / "cache" / "ckpt_full" / f"{FINAL_RUN}.pt"
N_EXAMPLES_TEST = 6  # imágenes de PlantVillage (test) en la galería de ejemplos
N_EXAMPLES_EXT = 4   # imágenes externas (PlantDoc) en la galería de ejemplos

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"No se encontró {CKPT_PATH}. Ejecutar primero el notebook 03 en modo full.")

df, classes = load_split()
labels_es = [CLASS_NAMES_ES[c] for c in classes]

# Arquitectura sin pesos de ImageNet: el checkpoint ya tiene todos los parámetros fine-tuneados
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

_, eval_tf = make_transforms(IMG_SIZE_FT, augment=False)

# ---------------------------------------------------------------------------
# Grad-CAM sobre la última etapa convolucional (misma capa que el notebook 05)
# ---------------------------------------------------------------------------
_store = {}
target_layer = model.features[-1]
target_layer.register_forward_hook(lambda m, i, o: _store.__setitem__("act", o))
target_layer.register_full_backward_hook(lambda m, gi, go: _store.__setitem__("grad", go[0]))


def gradcam(x, class_idx):
    """Mapa Grad-CAM normalizado a [0, 1] (resolución de la entrada) para la clase `class_idx`."""
    model.zero_grad()
    logits = model(x)
    logits[0, class_idx].backward()
    A, G = _store["act"].detach(), _store["grad"].detach()
    cam = F.relu((G.mean(dim=(2, 3), keepdim=True) * A).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    mn, mx = cam.min(), cam.max()
    return ((cam - mn) / (mx - mn)).cpu().numpy() if mx > mn else np.zeros(cam.shape, np.float32)


def overlay(img, cam, alpha):
    """Superpone el heatmap (colormap jet) sobre la imagen original, a su resolución."""
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)
    heat = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    out = (1 - alpha) * np.asarray(img, np.float32) / 255 + alpha * heat
    return np.uint8(np.clip(out, 0, 1) * 255)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------
def classify(img, target, alpha):
    if img is None:
        return None, None, "Cargar una imagen para clasificar."
    img = img.convert("RGB")
    x = eval_tf(img).unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000

    pred = int(probs.argmax())
    cam_idx = pred if target == "Clase predicha" else labels_es.index(target)
    cam = gradcam(x, cam_idx)

    info = (f"**Predicción:** {labels_es[pred]} ({probs[pred]:.1%})  \n"
            f"**Grad-CAM de:** {labels_es[cam_idx]}  \n"
            f"**Latencia de inferencia:** {latency_ms:.1f} ms ({DEVICE.type.upper()})")
    return {labels_es[i]: float(p) for i, p in enumerate(probs)}, overlay(img, cam, alpha), info


def build_examples():
    """Ejemplos disponibles localmente: algunas imágenes de test de PlantVillage y de PlantDoc."""
    examples = []
    test_df = df[df["split"] == "test"]
    if DATA_DIR.exists():
        rng = np.random.default_rng(0)
        for y in rng.choice(len(classes), size=min(N_EXAMPLES_TEST, len(classes)), replace=False):
            row = test_df[test_df["y"] == y].iloc[0]
            examples.append(DATA_DIR / row["relpath"])
    ext_dir = ROOT / "data" / "real_world"
    if ext_dir.exists():
        for d in sorted(p for p in ext_dir.iterdir() if p.is_dir())[:N_EXAMPLES_EXT]:
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if imgs:
                examples.append(imgs[0])
    return [str(p) for p in examples]


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
with gr.Blocks(title="Detección de enfermedades en hojas") as demo:
    gr.Markdown(
        "# Detección de enfermedades en hojas\n"
        f"EfficientNet-B0 fine-tuneado sobre PlantVillage ({len(classes)} clases de pimiento, papa y tomate). "
        "El mapa Grad-CAM muestra qué regiones de la imagen pesaron más en la decisión."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Imagen de la hoja", sources=["upload", "webcam", "clipboard"])
            target = gr.Dropdown(["Clase predicha"] + labels_es, value="Clase predicha", label="Explicar la clase")
            alpha = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="Opacidad del Grad-CAM")
            btn = gr.Button("Clasificar", variant="primary")
        with gr.Column():
            out_label = gr.Label(num_top_classes=5, label="Top-5 clases")
            out_cam = gr.Image(label="Grad-CAM", interactive=False)
            out_info = gr.Markdown()

    outputs = [out_label, out_cam, out_info]
    examples = build_examples()
    if examples:
        # Al elegir un ejemplo cambia la imagen y eso dispara la clasificación (inp.change)
        gr.Examples(examples, inputs=inp, examples_per_page=len(examples),
                    label="Ejemplos (PlantVillage test y PlantDoc)")

    btn.click(classify, [inp, target, alpha], outputs)
    inp.change(classify, [inp, target, alpha], outputs)
    target.change(classify, [inp, target, alpha], outputs)
    alpha.release(classify, [inp, target, alpha], outputs)

if __name__ == "__main__":
    demo.launch()
