"""
=============================================================================
  Binary Tongue Coating Classification — Systematic Experimentation Pipeline
  EEM561 / Medical Image Analysis Homework
=============================================================================
  How to run
  ----------
  1. pip install torch torchvision scikit-learn matplotlib seaborn pillow
  2. Place your dataset in one of these layouts:
       Layout A (class sub-folders):
           dataset/
               coated/   (or "1", "positive", etc.)
               normal/   (or "0", "negative", etc.)
       Layout B (flat with CSV labels):
           dataset/images/
           dataset/labels.csv   (columns: filename, label)
  3. Set DATASET_ROOT below, then: python tongue_coating_classification.py
=============================================================================
"""

# ── user settings ─────────────────────────────────────────────────────────
DATASET_ROOT = "./dataset"          # <── change this to your data folder
IMG_SIZE     = 224                  # resize to IMG_SIZE × IMG_SIZE
BATCH_SIZE   = 32
EPOCHS_BASE  = 30
EPOCHS_FINE  = 20
RANDOM_SEED  = 42
DEVICE       = "cuda"               # "cuda" or "cpu"
# ──────────────────────────────────────────────────────────────────────────

import os, json, random, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── reproducibility ────────────────────────────────────────────────────────
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

seed_everything()
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ══════════════════════════════════════════════════════════════════════════
#  1.  TRANSFORMS
# ══════════════════════════════════════════════════════════════════════════
# ── Step 0  No preprocessing, no augmentation (pure baseline) ─────────────
tf_baseline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ── Step 1  Preprocessing only (normalisation) ────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

tf_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Step 2  Preprocessing + augmentation (training only) ──────────────────
tf_augment = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ══════════════════════════════════════════════════════════════════════════
#  2.  DATASET LOADING & SPLITTING
# ══════════════════════════════════════════════════════════════════════════
def load_dataset(root, transform):
    """Load from ImageFolder-style directory."""
    return datasets.ImageFolder(root=root, transform=transform)


def split_indices(dataset, val_ratio=0.10, test_ratio=0.10, seed=RANDOM_SEED):
    """80/10/10 stratified split — returns (train_idx, val_idx, test_idx)."""
    labels  = [s[1] for s in dataset.samples]
    indices = list(range(len(dataset)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio),
        stratify=labels, random_state=seed)

    temp_labels = [labels[i] for i in temp_idx]
    relative_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=relative_test_ratio,
        stratify=temp_labels, random_state=seed)

    print(f"  Split → train:{len(train_idx)}  val:{len(val_idx)}  test:{len(test_idx)}")
    return train_idx, val_idx, test_idx


def make_loaders(root, train_tf, eval_tf, batch_size=BATCH_SIZE):
    """
    Build DataLoaders:
      • training set  → train_tf  (may include augmentation)
      • val + test    → eval_tf   (no augmentation, only normalisation)
    """
    # full dataset with eval transform to get samples list right
    full_eval = load_dataset(root, eval_tf)
    train_idx, val_idx, test_idx = split_indices(full_eval)

    # training subset uses its own transform
    full_train = load_dataset(root, train_tf)
    train_subset = Subset(full_train, train_idx)
    val_subset   = Subset(full_eval,  val_idx)
    test_subset  = Subset(full_eval,  test_idx)

    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, generator=g)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size,
                              shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, full_eval.classes


# ══════════════════════════════════════════════════════════════════════════
#  3.  MODELS
# ══════════════════════════════════════════════════════════════════════════
class BasicCNN(nn.Module):
    """Simple 3-block CNN — the 'basic model' required by the homework."""
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            # block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            # block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        feat_dim = (IMG_SIZE // 8) ** 2 * 128
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),       nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18(num_classes=2, pretrained=True, freeze_backbone=False):
    """ResNet-18 with optional ImageNet weights — 'enhanced model'."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


# ══════════════════════════════════════════════════════════════════════════
#  4.  TRAIN / EVALUATE HELPERS
# ══════════════════════════════════════════════════════════════════════════
def compute_class_weights(dataset, train_idx):
    labels = [dataset.samples[i][1] for i in train_idx]
    counts = np.bincount(labels)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scheduler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += X.size(0)
    if scheduler: scheduler.step()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        all_preds .extend(out.argmax(1).cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    n = len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / n, acc, np.array(all_preds), np.array(all_labels)


def metrics(preds, labels, classes):
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="weighted", zero_division=0)
    f1   = f1_score(labels, preds, average="weighted", zero_division=0)
    return dict(accuracy=acc, precision=prec, f1=f1)


def full_train(model, train_loader, val_loader, criterion,
               optimizer, scheduler, epochs, tag=""):
    """Train loop with early stopping on val loss (patience=7)."""
    best_val_loss = float("inf")
    patience, no_imp = 7, 0
    best_state = None
    history = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, scheduler)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc" ].append(tr_acc)
        history["val_loss"  ].append(vl_loss)
        history["val_acc"   ].append(vl_acc)

        if vl_loss < best_val_loss - 1e-4:
            best_val_loss = vl_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep % 5 == 0 or ep == 1:
            print(f"  [{tag}] Ep {ep:3d}/{epochs} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                  f"val_loss={vl_loss:.4f} val_acc={vl_acc:.3f}")

        if no_imp >= patience:
            print(f"  [{tag}] Early stop at epoch {ep}.")
            break

    model.load_state_dict(best_state)
    return model, history


# ══════════════════════════════════════════════════════════════════════════
#  5.  EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════
RESULTS = {}   # experiment_name → {metrics, history, cm}

def run_experiment(name, model, train_loader, val_loader, test_loader,
                   classes, epochs, lr=1e-3, use_class_weights=False,
                   dataset_full=None, train_idx=None,
                   scheduler_type="cosine"):
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    seed_everything()
    model = model.to(device)

    # loss
    if use_class_weights and dataset_full and train_idx is not None:
        cw = compute_class_weights(dataset_full, train_idx).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        print(f"  Class weights: {cw.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4)

    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    model, history = full_train(model, train_loader, val_loader,
                                criterion, optimizer, scheduler,
                                epochs, tag=name)

    # test evaluation
    _, _, preds, labels = evaluate(model, test_loader, criterion)
    m  = metrics(preds, labels, classes)
    cm = confusion_matrix(labels, preds)

    print(f"\n  TEST RESULTS  →  acc={m['accuracy']:.4f}  "
          f"prec={m['precision']:.4f}  f1={m['f1']:.4f}")
    print(classification_report(labels, preds, target_names=classes))

    RESULTS[name] = dict(metrics=m, history=history, cm=cm,
                         preds=preds, labels=labels)
    return model


# ══════════════════════════════════════════════════════════════════════════
#  6.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════
def plot_training_curves(ax_loss, ax_acc, history, label):
    ep = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(ep, history["train_loss"], label=f"{label} train")
    ax_loss.plot(ep, history["val_loss"],   label=f"{label} val", linestyle="--")
    ax_acc.plot (ep, history["train_acc"],  label=f"{label} train")
    ax_acc.plot (ep, history["val_acc"],    label=f"{label} val",  linestyle="--")


def save_all_figures(out_dir="./outputs"):
    os.makedirs(out_dir, exist_ok=True)

    n_exp   = len(RESULTS)
    exp_names = list(RESULTS.keys())

    # ── Figure 1: training curves ─────────────────────────────────────────
    fig, axes = plt.subplots(n_exp, 2, figsize=(14, 4 * n_exp))
    if n_exp == 1: axes = [axes]
    for ax_row, name in zip(axes, exp_names):
        h = RESULTS[name]["history"]
        ep = range(1, len(h["train_loss"]) + 1)
        ax_row[0].plot(ep, h["train_loss"], label="train"); ax_row[0].plot(ep, h["val_loss"], label="val", ls="--")
        ax_row[0].set_title(f"{name} — Loss"); ax_row[0].legend(); ax_row[0].set_xlabel("Epoch")
        ax_row[1].plot(ep, h["train_acc"],  label="train"); ax_row[1].plot(ep, h["val_acc"],  label="val", ls="--")
        ax_row[1].set_title(f"{name} — Accuracy"); ax_row[1].legend(); ax_row[1].set_xlabel("Epoch")
    plt.tight_layout()
    path1 = f"{out_dir}/training_curves.png"
    fig.savefig(path1, dpi=150); plt.close(fig)
    print(f"[SAVED] {path1}")

    # ── Figure 2: confusion matrices ──────────────────────────────────────
    cols = min(n_exp, 3)
    rows = (n_exp + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    for ax, name in zip(axes, exp_names):
        cm = RESULTS[name]["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for ax in axes[n_exp:]:
        ax.axis("off")
    plt.tight_layout()
    path2 = f"{out_dir}/confusion_matrices.png"
    fig.savefig(path2, dpi=150); plt.close(fig)
    print(f"[SAVED] {path2}")

    # ── Figure 3: metric comparison bar chart ─────────────────────────────
    metric_keys = ["accuracy", "precision", "f1"]
    x  = np.arange(len(exp_names))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 2 * n_exp), 5))
    for i, mk in enumerate(metric_keys):
        vals = [RESULTS[n]["metrics"][mk] for n in exp_names]
        bars = ax.bar(x + i * w, vals, width=w, label=mk.capitalize())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w)
    ax.set_xticklabels(exp_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score"); ax.legend()
    ax.set_title("Experiment Comparison — Accuracy / Precision / F1")
    plt.tight_layout()
    path3 = f"{out_dir}/metric_comparison.png"
    fig.savefig(path3, dpi=150); plt.close(fig)
    print(f"[SAVED] {path3}")

    return path1, path2, path3


# ══════════════════════════════════════════════════════════════════════════
#  7.  REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════
def generate_report(out_dir="./outputs"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append("=" * 70)
    lines.append("  Binary Tongue Coating Classification — Experiment Report")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("1. EXPERIMENTAL SETUP")
    lines.append("-" * 40)
    lines.append(f"  Image size   : {IMG_SIZE}×{IMG_SIZE}")
    lines.append(f"  Batch size   : {BATCH_SIZE}")
    lines.append(f"  Random seed  : {RANDOM_SEED}")
    lines.append(f"  Device       : {device}")
    lines.append(f"  Split        : 80% train / 10% val / 10% test (stratified)")
    lines.append("")

    lines.append("2. RESULTS SUMMARY")
    lines.append("-" * 40)
    header = f"  {'Experiment':<30} {'Accuracy':>10} {'Precision':>10} {'F1':>10}"
    lines.append(header)
    lines.append("  " + "-" * 62)
    for name, res in RESULTS.items():
        m = res["metrics"]
        lines.append(f"  {name:<30} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['f1']:>10.4f}")
    lines.append("")

    lines.append("3. DETAILED CONFUSION MATRICES")
    lines.append("-" * 40)
    for name, res in RESULTS.items():
        lines.append(f"  [{name}]")
        cm = res["cm"]
        lines.append(f"  {cm}")
        lines.append("")

    lines.append("4. PROGRESSIVE IMPROVEMENT ANALYSIS")
    lines.append("-" * 40)
    exp_list = list(RESULTS.items())
    baseline_f1 = exp_list[0][1]["metrics"]["f1"]
    for name, res in exp_list[1:]:
        delta = res["metrics"]["f1"] - baseline_f1
        sign  = "+" if delta >= 0 else ""
        lines.append(f"  {name} vs baseline: F1 {sign}{delta:.4f}")
    lines.append("")

    lines.append("5. HYPERPARAMETERS")
    lines.append("-" * 40)
    lines.append("  Baseline CNN:")
    lines.append("    Conv layers  : 3 (32→64→128 filters, 3×3, BatchNorm+ReLU+MaxPool)")
    lines.append("    FC layers    : 256 → 64 → 2  (Dropout 0.5 / 0.25)")
    lines.append("    Optimizer    : AdamW  lr=1e-3  weight_decay=1e-4")
    lines.append("    Scheduler    : CosineAnnealingLR")
    lines.append("    Early stop   : patience=7 epochs")
    lines.append("")
    lines.append("  Enhanced ResNet-18:")
    lines.append("    Backbone     : ResNet-18  (ImageNet pretrained)")
    lines.append("    Head         : Dropout(0.4) → Linear(512→2)")
    lines.append("    Optimizer    : AdamW  lr=3e-4")
    lines.append("    Scheduler    : CosineAnnealingLR")
    lines.append("")

    lines.append("6. APPROACHES USED")
    lines.append("-" * 40)
    lines.append("""
  Step 0 — Baseline CNN (no preprocessing, no augmentation)
    A simple 3-block CNN trained on raw resized images.

  Step 1 — Preprocessing
    ImageNet normalisation (mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]) applied to all splits.

  Step 2 — Data Augmentation (training only)
    Random crop, horizontal/vertical flip, rotation ±20°,
    colour jitter (brightness, contrast, saturation, hue),
    and occasional random greyscale.
    Augmentation is NEVER applied to validation or test sets.

  Step 3 — Class-Weighted Loss
    Inverse-frequency class weights in CrossEntropyLoss to handle
    potential class imbalance without discarding samples.

  Step 4 — Transfer Learning (ResNet-18)
    ImageNet pretrained ResNet-18 fine-tuned with a custom
    classification head; cosine LR schedule.
""")

    lines.append("7. FUTURE WORK")
    lines.append("-" * 40)
    lines.append("""
  • Larger backbones (EfficientNet-B4, ViT-Base) with progressive resizing.
  • Test-Time Augmentation (TTA): average predictions over flipped/rotated views.
  • Grad-CAM visualisation to identify which tongue regions drive predictions.
  • Self-supervised pre-training (SimCLR / DINO) on unlabelled tongue images
    before supervised fine-tuning.
  • Ensemble of CNN + ViT models with late fusion.
  • Semi-supervised learning if unlabelled samples are available.
  • Hyperparameter optimisation with Optuna / Ray Tune.
  • Cross-dataset evaluation to measure clinical generalisation.
""")

    lines.append("=" * 70)
    report_text = "\n".join(lines)
    path = f"{out_dir}/report.txt"
    with open(path, "w") as f:
        f.write(report_text)
    print(f"[SAVED] {path}")
    print("\n" + report_text)
    return path


# ══════════════════════════════════════════════════════════════════════════
#  8.  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    OUT_DIR = "./outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── check dataset exists ──────────────────────────────────────────────
    if not os.path.isdir(DATASET_ROOT):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset folder not found: '{DATASET_ROOT}'\n"
            "Please set DATASET_ROOT at the top of this script.")

    # ── Experiment 0: BASELINE (no preprocessing, no augmentation) ────────
    print("\n[LOADING] Experiment 0: Baseline")
    tr0, vl0, te0, classes = make_loaders(
        DATASET_ROOT, train_tf=tf_baseline, eval_tf=tf_baseline)
    full_ds = load_dataset(DATASET_ROOT, tf_baseline)
    _, val_idx0, _ = split_indices(full_ds)   # for class-weight ref later

    model0 = BasicCNN(num_classes=len(classes))
    run_experiment(
        "0_Baseline_CNN",
        model0, tr0, vl0, te0, classes,
        epochs=EPOCHS_BASE, lr=1e-3)

    # ── Experiment 1: + Preprocessing (normalisation) ─────────────────────
    print("\n[LOADING] Experiment 1: + Preprocessing")
    tr1, vl1, te1, _ = make_loaders(
        DATASET_ROOT, train_tf=tf_preprocess, eval_tf=tf_preprocess)

    model1 = BasicCNN(num_classes=len(classes))
    run_experiment(
        "1_CNN_+_Preprocess",
        model1, tr1, vl1, te1, classes,
        epochs=EPOCHS_BASE, lr=1e-3)

    # ── Experiment 2: + Augmentation ──────────────────────────────────────
    print("\n[LOADING] Experiment 2: + Augmentation")
    tr2, vl2, te2, _ = make_loaders(
        DATASET_ROOT, train_tf=tf_augment, eval_tf=tf_preprocess)

    model2 = BasicCNN(num_classes=len(classes))
    run_experiment(
        "2_CNN_+_Augment",
        model2, tr2, vl2, te2, classes,
        epochs=EPOCHS_BASE, lr=1e-3)

    # ── Experiment 3: + Class-weighted loss ───────────────────────────────
    print("\n[LOADING] Experiment 3: + Class-Weighted Loss")
    full_ds3 = load_dataset(DATASET_ROOT, tf_augment)
    train_idx3, _, _ = split_indices(full_ds3)

    model3 = BasicCNN(num_classes=len(classes))
    run_experiment(
        "3_CNN_+_Aug_+_CW",
        model3, tr2, vl2, te2, classes,
        epochs=EPOCHS_BASE, lr=1e-3,
        use_class_weights=True,
        dataset_full=full_ds3, train_idx=train_idx3)

    # ── Experiment 4: Transfer Learning — ResNet-18 ───────────────────────
    print("\n[LOADING] Experiment 4: Transfer Learning (ResNet-18)")
    tr4, vl4, te4, _ = make_loaders(
        DATASET_ROOT, train_tf=tf_augment, eval_tf=tf_preprocess)

    model4 = build_resnet18(num_classes=len(classes), pretrained=True)
    run_experiment(
        "4_ResNet18_Transfer",
        model4, tr4, vl4, te4, classes,
        epochs=EPOCHS_FINE, lr=3e-4)

    # ── Save figures & report ─────────────────────────────────────────────
    save_all_figures(OUT_DIR)
    generate_report(OUT_DIR)

    print("\n[DONE] All outputs saved to:", OUT_DIR)
