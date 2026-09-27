"""Código compartido entre notebooks: datos, transforms, backbones y métricas."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PV_DATA_DIR", ROOT / "PlantVillage"))
SPLITS_DIR = ROOT / "splits"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SEED = 42
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_tf = v2.Compose([
    v2.Resize(IMG_SIZE), v2.CenterCrop(IMG_SIZE),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

train_tf = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomApply([v2.RandomRotation(30)], p=0.5),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
    v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
    v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class PlantDataset(Dataset):
    def __init__(self, split_df, root=DATA_DIR, transform=None):
        self.df = split_df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(self.root / r["relpath"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(r["y"])


def load_split():
    """Devuelve (DataFrame con columna 'split', lista de clases ordenada por índice)."""
    df = pd.read_csv(SPLITS_DIR / "split.csv")
    class_to_idx = json.load(open(SPLITS_DIR / "class_to_idx.json"))
    classes = sorted(class_to_idx, key=class_to_idx.get)
    return df, classes


def build_backbone(name):
    """Modelo preentrenado en ImageNet sin la capa de clasificación. Devuelve (modelo, dim_features)."""
    net = models.get_model(name, weights="DEFAULT")
    if name.startswith("resnet"):
        dim = net.fc.in_features
        net.fc = torch.nn.Identity()
    elif name.startswith("efficientnet"):
        dim = net.classifier[1].in_features
        net.classifier = torch.nn.Identity()
    elif name.startswith("mobilenet_v3"):
        dim = net.classifier[0].in_features
        net.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Backbone no soportado: {name}")
    return net.eval(), dim


@torch.no_grad()
def extract_features(net, split_df, batch_size=64, num_workers=0, desc=""):
    from tqdm.auto import tqdm
    loader = DataLoader(PlantDataset(split_df, transform=eval_tf), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    feats = []
    for xb, _ in tqdm(loader, desc=desc):
        feats.append(net(xb).numpy())
    return np.concatenate(feats)


def get_features(name, df):
    """Embeddings de todo el dataset (cacheados en cache/feats_<name>.npz), en el orden de df."""
    path = CACHE_DIR / f"feats_{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["X"], z["seconds"].item()
    net, _ = build_backbone(name)
    t0 = time.time()
    X = extract_features(net, df, desc=name)
    secs = time.time() - t0
    np.savez(path, X=X, seconds=secs)
    return X, secs


def metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


# ---------------------------------------------------------------------------
# Transfer learning / fine-tuning
# ---------------------------------------------------------------------------
import copy


def make_transforms(size, augment):
    """Pipeline de train (con o sin augmentation) y de evaluación a una resolución dada."""
    tail = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if augment:
        train = v2.Compose([
            v2.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomApply([v2.RandomRotation(30)], p=0.5),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
            v2.RandomApply([v2.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.2),
            *tail,
        ])
    else:
        train = v2.Compose([v2.Resize((size, size)), *tail])
    return train, v2.Compose([v2.Resize((size, size)), *tail])


def finetune_model(name, num_classes, n_unfreeze):
    """Backbone preentrenado con nuevo head; solo las últimas `n_unfreeze` etapas de `features` y el head se entrenan."""
    net = models.get_model(name, weights="DEFAULT")
    if name.startswith("mobilenet_v3"):
        net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, num_classes)
    elif name.startswith("efficientnet"):
        net.classifier[1] = torch.nn.Linear(net.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Modelo no soportado: {name}")
    for p in net.parameters():
        p.requires_grad = False
    blocks = list(net.features)
    for b in blocks[-n_unfreeze:] if n_unfreeze else []:
        for p in b.parameters():
            p.requires_grad = True
    for p in net.classifier.parameters():
        p.requires_grad = True
    net.frozen_blocks = blocks[:len(blocks) - n_unfreeze]
    return net


def set_train_mode(net):
    """train() salvo en las etapas congeladas: sus BatchNorm no deben actualizar estadísticas."""
    net.train()
    for b in getattr(net, "frozen_blocks", []):
        b.eval()


@torch.no_grad()
def predict(net, loader):
    net.eval()
    logits, ys = [], []
    for xb, yb in loader:
        logits.append(net(xb).numpy()); ys.append(yb.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def fit(net, train_loader, val_loader, epochs, lr_head=1e-3, lr_backbone=3e-4, weight_decay=1e-4,
        seed=SEED, log_path=None):
    """Entrena con AdamW + cosine schedule. Devuelve (historial, mejor state_dict por macro-F1 en val)."""
    torch.manual_seed(seed)
    head_ids = {id(p) for p in net.classifier.parameters()}
    head = [p for p in net.parameters() if p.requires_grad and id(p) in head_ids]
    back = [p for p in net.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": back, "lr": lr_backbone}] if back else [])
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(train_loader))
    loss_fn = torch.nn.CrossEntropyLoss()
    history, best_f1, best_state = [], -1.0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        set_train_mode(net)
        run_loss, n = 0.0, 0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward(); opt.step(); sched.step()
            run_loss += loss.item() * len(xb); n += len(xb)
        logits, yv = predict(net, val_loader)
        val_loss = loss_fn(torch.tensor(logits), torch.tensor(yv)).item()
        m = metrics(yv, logits.argmax(1))
        rec = {"epoch": ep, "train_loss": run_loss / n, "val_loss": val_loss,
               "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"], "seconds": time.time() - t0}
        history.append(rec)
        if m["macro_f1"] > best_f1:
            best_f1, best_state = m["macro_f1"], copy.deepcopy(net.state_dict())
        line = (f"ep {ep}/{epochs} | train_loss {rec['train_loss']:.4f} | val_loss {val_loss:.4f} | "
                f"val_acc {m['accuracy']:.4f} | val_macro_f1 {m['macro_f1']:.4f} | {rec['seconds']:.0f}s")
        print(line, flush=True)
        if log_path:
            with open(log_path, "a") as f:
                f.write(line + "\n")
    return history, best_state
