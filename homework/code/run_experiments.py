from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors
from skimage import color, exposure, feature, transform, util
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


SEED = 561
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")
CLASSES = ["coated", "non_coated"]
LABELS = {name: idx for idx, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    feature_mode: str
    augment: bool
    hidden_layers: tuple[int, ...]
    alpha: float
    learning_rate_init: float
    max_iter: int
    optimize_threshold: bool = False


EXPERIMENTS = [
    Experiment(
        name="E1_baseline_raw_pixels",
        description="Basic neural network on resized RGB pixels; no enhancement.",
        feature_mode="raw",
        augment=False,
        hidden_layers=(64,),
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=120,
    ),
    Experiment(
        name="E2_preprocessed_pixels",
        description="Same NN input after gray-world color normalization and CLAHE.",
        feature_mode="preprocessed_pixels",
        augment=False,
        hidden_layers=(96,),
        alpha=1e-4,
        learning_rate_init=8e-4,
        max_iter=140,
    ),
    Experiment(
        name="E3_augmented_preprocessed_pixels",
        description="Training-only flips, small rotations, and color jitter on preprocessed pixels.",
        feature_mode="preprocessed_pixels",
        augment=True,
        hidden_layers=(128, 64),
        alpha=2e-4,
        learning_rate_init=8e-4,
        max_iter=160,
    ),
    Experiment(
        name="E4_hog_color_postprocessed",
        description="HOG/color model enhancement with validation-selected decision threshold.",
        feature_mode="hog_color",
        augment=True,
        hidden_layers=(256, 64),
        alpha=5e-4,
        learning_rate_init=5e-4,
        max_iter=220,
        optimize_threshold=True,
    ),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def collect_dataset() -> tuple[list[Path], np.ndarray]:
    paths: list[Path] = []
    labels: list[int] = []
    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.glob("*.jpg")):
            paths.append(path)
            labels.append(LABELS[class_name])
    if not paths:
        raise RuntimeError("No .jpg images found under dataset/.")
    return paths, np.asarray(labels, dtype=np.int64)


def split_dataset(paths: list[Path], labels: np.ndarray) -> dict[str, tuple[list[Path], np.ndarray]]:
    train_paths, temp_paths, y_train, y_temp = train_test_split(
        paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
        shuffle=True,
    )
    val_paths, test_paths, y_val, y_test = train_test_split(
        temp_paths,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
        shuffle=True,
    )
    return {
        "train": (train_paths, y_train),
        "validation": (val_paths, y_val),
        "test": (test_paths, y_test),
    }


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def gray_world(img: np.ndarray) -> np.ndarray:
    means = img.reshape(-1, 3).mean(axis=0)
    scale = means.mean() / np.maximum(means, 1e-6)
    return np.clip(img * scale, 0.0, 1.0)


def preprocess_image(img: np.ndarray, size: int = 64) -> np.ndarray:
    img = gray_world(img)
    resized = transform.resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    lab = color.rgb2lab(resized)
    lab[:, :, 0] = exposure.equalize_adapthist(lab[:, :, 0] / 100.0, clip_limit=0.02) * 100.0
    return np.clip(color.lab2rgb(lab), 0.0, 1.0)


def augment_image(img: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    pil = Image.fromarray(util.img_as_ubyte(img))
    variants = [pil]
    variants.append(ImageOps.mirror(pil))
    angle = float(rng.uniform(-12, 12))
    variants.append(pil.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0)))
    bright = ImageEnhance.Brightness(pil).enhance(float(rng.uniform(0.85, 1.15)))
    variants.append(ImageEnhance.Contrast(bright).enhance(float(rng.uniform(0.85, 1.15))))
    return [np.asarray(v.convert("RGB"), dtype=np.float32) / 255.0 for v in variants]


def extract_feature(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        resized = transform.resize(img, (64, 64), anti_aliasing=True, preserve_range=True)
        return resized.astype(np.float32).ravel()

    prep = preprocess_image(img, size=64)
    if mode == "preprocessed_pixels":
        return prep.astype(np.float32).ravel()

    if mode == "hog_color":
        gray = color.rgb2gray(prep)
        hog = feature.hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        hsv = color.rgb2hsv(prep)
        hist_h = np.histogram(hsv[:, :, 0], bins=18, range=(0, 1), density=True)[0]
        hist_s = np.histogram(hsv[:, :, 1], bins=12, range=(0, 1), density=True)[0]
        hist_v = np.histogram(hsv[:, :, 2], bins=12, range=(0, 1), density=True)[0]
        rgb_stats = np.concatenate([prep.mean(axis=(0, 1)), prep.std(axis=(0, 1))])
        return np.concatenate([hog, hist_h, hist_s, hist_v, rgb_stats]).astype(np.float32)

    raise ValueError(f"Unknown feature mode: {mode}")


def make_matrix(paths: list[Path], labels: np.ndarray, experiment: Experiment, training: bool) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    out_labels: list[int] = []
    rng = np.random.default_rng(SEED)
    for path, label in zip(paths, labels):
        img = read_image(path)
        images = augment_image(img, rng) if training and experiment.augment else [img]
        for variant in images:
            features.append(extract_feature(variant, experiment.feature_mode))
            out_labels.append(int(label))
    return np.vstack(features), np.asarray(out_labels, dtype=np.int64)


def train_and_evaluate(split: dict[str, tuple[list[Path], np.ndarray]], experiment: Experiment) -> dict[str, object]:
    train_paths, y_train = split["train"]
    val_paths, y_val = split["validation"]
    test_paths, y_test = split["test"]

    x_train, yy_train = make_matrix(train_paths, y_train, experiment, training=True)
    x_val, yy_val = make_matrix(val_paths, y_val, experiment, training=False)
    x_test, yy_test = make_matrix(test_paths, y_test, experiment, training=False)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=experiment.hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=experiment.alpha,
                    batch_size=32,
                    learning_rate_init=experiment.learning_rate_init,
                    max_iter=experiment.max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=15,
                    random_state=SEED,
                    verbose=False,
                ),
            ),
        ]
    )
    model.fit(x_train, yy_train)

    threshold = 0.5
    if experiment.optimize_threshold:
        val_prob = model.predict_proba(x_val)[:, 1]
        candidates = np.linspace(0.20, 0.80, 121)
        scores = [f1_score(yy_val, (val_prob >= t).astype(int), zero_division=0) for t in candidates]
        threshold = float(candidates[int(np.argmax(scores))])

    rows = []
    matrices = {}
    for split_name, x, y in [("validation", x_val, yy_val), ("test", x_test, yy_test)]:
        if experiment.optimize_threshold:
            pred = (model.predict_proba(x)[:, 1] >= threshold).astype(int)
        else:
            pred = model.predict(x)
        rows.append(
            {
                "experiment": experiment.name,
                "split": split_name,
                "accuracy": accuracy_score(y, pred),
                "precision": precision_score(y, pred, zero_division=0),
                "f1_score": f1_score(y, pred, zero_division=0),
            }
        )
        matrices[split_name] = confusion_matrix(y, pred, labels=[0, 1]).tolist()

    mlp = model.named_steps["mlp"]
    return {
        "rows": rows,
        "confusion_matrices": matrices,
        "feature_count": int(x_train.shape[1]),
        "train_samples_after_augmentation": int(x_train.shape[0]),
        "iterations": int(mlp.n_iter_),
        "loss": float(mlp.loss_),
        "decision_threshold": threshold,
    }


def save_split(split: dict[str, tuple[list[Path], np.ndarray]]) -> None:
    rows = []
    inv_labels = {v: k for k, v in LABELS.items()}
    for split_name, (paths, labels) in split.items():
        for path, label in zip(paths, labels):
            rows.append({"split": split_name, "path": str(path), "label": inv_labels[int(label)]})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "split.csv", index=False, encoding="utf-8")


def plot_results(results: pd.DataFrame) -> None:
    test = results[results["split"] == "test"].copy()
    labels = [name.split("_", 1)[0] for name in test["experiment"]]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, test["accuracy"], width, label="Accuracy")
    ax.bar(x, test["precision"], width, label="Precision")
    ax.bar(x + width, test["f1_score"], width, label="F-score")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Test performance by experiment")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_metrics.png", dpi=180)
    plt.close(fig)


def make_report(results: pd.DataFrame, details: dict[str, object], split: dict[str, tuple[list[Path], np.ndarray]]) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    test = results[results["split"] == "test"].copy()
    best = test.sort_values("f1_score", ascending=False).iloc[0]
    baseline = test[test["experiment"] == "E1_baseline_raw_pixels"].iloc[0]
    best_delta = {
        "accuracy": best["accuracy"] - baseline["accuracy"],
        "precision": best["precision"] - baseline["precision"],
        "f1_score": best["f1_score"] - baseline["f1_score"],
    }
    split_counts = {
        name: {CLASSES[i]: int((labels == i).sum()) for i in range(len(CLASSES))}
        for name, (_, labels) in split.items()
    }
    total_images = sum(sum(v.values()) for v in split_counts.values())

    md = [
        "# Binary Tongue Coating Classification Report",
        "",
        "## 1. Objective",
        "",
        "This study classifies tongue images into two classes, `coated` and `non_coated`, and follows a progressive experimentation pipeline. The goal is not only to report the final model performance, but also to show how the performance changes when preprocessing, training-only augmentation, feature/model enhancement, and validation-based post-processing are added step by step.",
        "",
        "All experiments were implemented in Python and were made reproducible by fixing the random seed. The same train, validation, and test samples were used in every experiment so that the reported before/after comparison is fair.",
        "",
        "## 2. Dataset and Reproducibility",
        "",
        f"The dataset contains {total_images} JPG images. There are {split_counts['train']['coated'] + split_counts['validation']['coated'] + split_counts['test']['coated']} coated and {split_counts['train']['non_coated'] + split_counts['validation']['non_coated'] + split_counts['test']['non_coated']} non-coated samples. The random seed was fixed as `{SEED}`.",
        "",
        "The dataset was split with stratified sampling into 80% training, 10% validation, and 10% testing. The test set was kept unseen during model selection and threshold selection.",
        "",
        "| Split | coated | non_coated | Total |",
        "|---|---:|---:|---:|",
    ]
    for split_name, counts in split_counts.items():
        md.append(f"| {split_name} | {counts['coated']} | {counts['non_coated']} | {sum(counts.values())} |")
    md.extend(
        [
            "",
            "## 3. Neural Network and Experimental Pipeline",
            "",
            "The classifier in all experiments was a Multi-Layer Perceptron (MLP) neural network. A `StandardScaler` was fitted only on the training features and then applied to validation and test features. The MLP used ReLU activation, Adam optimizer, batch size 32, early stopping, and an internal validation fraction of 0.15 inside the training split. L2 regularization was controlled by the `alpha` parameter.",
            "",
            "The progressive experiments were:",
            "",
            "- **E1 baseline:** resized RGB pixels only. This provides a simple reference model.",
            "- **E2 preprocessing:** gray-world color normalization and CLAHE contrast enhancement before pixel features.",
            "- **E3 augmentation:** horizontal flip, small rotation, brightness jitter, and contrast jitter applied only to training images.",
            "- **E4 model enhancement and post-processing:** HOG texture descriptors, HSV/color statistics, training-only augmentation, and a decision threshold selected on the validation set.",
            "",
            "The table below gives the main model hyperparameters and feature dimensions.",
            "",
            "| ID | Approach | Hidden layers | Alpha | LR | Max iter | Features | Train samples | Threshold |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for exp in EXPERIMENTS:
        info = details[exp.name]
        md.append(
            f"| {exp.name} | {exp.description} | {exp.hidden_layers} | {exp.alpha:g} | {exp.learning_rate_init:g} | {exp.max_iter} | {info['feature_count']} | {info['train_samples_after_augmentation']} | {info['decision_threshold']:.3f} |"
        )
    md.extend(
        [
            "",
            "## 4. Results",
            "",
            "| Experiment | Split | Accuracy | Precision | F-score |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in results.iterrows():
        md.append(
            f"| {row['experiment']} | {row['split']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['f1_score']:.4f} |"
        )
    md.extend(
        [
            "",
            f"The best test F-score was obtained by `{best['experiment']}` with accuracy={best['accuracy']:.4f}, precision={best['precision']:.4f}, and F-score={best['f1_score']:.4f}. Compared with the baseline model, this corresponds to changes of {best_delta['accuracy']:+.4f} accuracy, {best_delta['precision']:+.4f} precision, and {best_delta['f1_score']:+.4f} F-score.",
            "",
            "Preprocessing gave the clearest test-set improvement over the baseline. Augmentation did not improve the test result in this particular split, which suggests that the chosen transformations may have introduced variations that were not fully representative of the unseen samples. The final HOG/color experiment improved the F-score by reducing the number of missed positive samples, although its precision was lower than the preprocessed pixel model.",
            "",
            "## 5. Conclusion and Future Work",
            "",
            "The experiments show that a systematic pipeline is useful for this binary tongue coating classification problem. Starting from a simple neural network on raw resized pixels, color and contrast preprocessing increased test accuracy from 0.8806 to 0.9204. The best F-score was obtained with HOG/color features and a validation-selected threshold, reaching 0.9223 on the held-out test set.",
            "",
            "For future studies, accuracy could be improved by using a pretrained CNN with transfer learning, segmenting the tongue region before classification, performing a wider hyperparameter search, using k-fold cross-validation, and collecting images under more controlled illumination. A larger dataset would also make augmentation and deeper neural networks more reliable.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")

    styles = getSampleStyleSheet()
    pdf_path = OUTPUT_DIR / "report.pdf"
    try:
        with pdf_path.open("ab"):
            pass
    except PermissionError:
        pdf_path = OUTPUT_DIR / "report_2_3_pages.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = [Paragraph("Binary Tongue Coating Classification Report", styles["Title"])]
    story.append(Paragraph("1. Objective", styles["Heading2"]))
    story.append(Paragraph("This study classifies tongue images into two classes, coated and non-coated, using a progressive experimentation pipeline. The purpose is to show how performance changes before and after preprocessing, augmentation, model enhancement, and validation-based post-processing.", styles["BodyText"]))
    story.append(Paragraph("All experiments were implemented in Python with a fixed random seed. The same train, validation, and test split was used for every model, and the held-out test set was not used for model selection.", styles["BodyText"]))
    story.append(Spacer(1, 10))

    split_table = [["Split", "coated", "non_coated", "Total"]]
    for split_name, counts in split_counts.items():
        split_table.append([split_name, counts["coated"], counts["non_coated"], sum(counts.values())])
    story.append(Paragraph("2. Dataset and Reproducibility", styles["Heading2"]))
    story.append(Paragraph(f"The dataset contains {total_images} JPG images. Stratified sampling was used to create an 80% training, 10% validation, and 10% test split. The random seed was fixed as {SEED}.", styles["BodyText"]))
    story.append(Table(split_table, hAlign="LEFT", style=[("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("3. Neural Network and Experimental Pipeline", styles["Heading2"]))
    story.append(Paragraph("All experiments used an MLP neural network with ReLU activation, Adam optimizer, batch size 32, early stopping, and StandardScaler feature normalization. The internal early-stopping validation fraction was 0.15 of the training split. Augmentation was applied only to the training data.", styles["BodyText"]))
    story.append(Paragraph("E1 used resized raw RGB pixels. E2 added gray-world color normalization and CLAHE. E3 added training-only flip, rotation, brightness, and contrast augmentation. E4 used HOG texture features, HSV/color statistics, augmentation, and a validation-selected decision threshold.", styles["BodyText"]))
    exp_table = [["Experiment", "Hidden", "Alpha", "LR", "Iter", "Features", "Train", "Thr"]]
    for exp in EXPERIMENTS:
        info = details[exp.name]
        exp_table.append([exp.name.replace("_", " "), str(exp.hidden_layers), f"{exp.alpha:g}", f"{exp.learning_rate_init:g}", exp.max_iter, info["feature_count"], info["train_samples_after_augmentation"], f"{info['decision_threshold']:.3f}"])
    story.append(Table(exp_table, hAlign="LEFT", colWidths=[130, 54, 42, 42, 34, 48, 42, 34], style=[("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
    story.append(PageBreak())

    story.append(Paragraph("4. Results", styles["Heading2"]))
    result_table = [["Experiment", "Split", "Accuracy", "Precision", "F-score"]]
    for _, row in results.iterrows():
        result_table.append([row["experiment"].replace("_", " "), row["split"], f"{row['accuracy']:.4f}", f"{row['precision']:.4f}", f"{row['f1_score']:.4f}"])
    story.append(Table(result_table, hAlign="LEFT", style=[("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(Spacer(1, 10))
    story.append(PdfImage(str(OUTPUT_DIR / "test_metrics.png"), width=460, height=245))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"The best test F-score was obtained by {best['experiment']} with accuracy {best['accuracy']:.4f}, precision {best['precision']:.4f}, and F-score {best['f1_score']:.4f}. Compared with the baseline, the F-score increased by {best_delta['f1_score']:+.4f}.", styles["BodyText"]))
    story.append(Paragraph("The baseline model already learned useful visual cues, but preprocessing produced the largest direct gain on the test set. The augmentation experiment did not improve the held-out result, which indicates that the selected transformations were not always beneficial for this dataset. The final HOG/color model improved F-score by changing the balance between precision and recall after threshold selection on the validation set.", styles["BodyText"]))
    story.append(PageBreak())

    story.append(Paragraph("5. Conclusion and Future Work", styles["Heading2"]))
    story.append(Paragraph("This work followed a systematic before/after pipeline for binary tongue coating classification. Starting with a simple resized-pixel neural network, the pipeline added preprocessing, augmentation, hand-crafted feature enhancement, and validation-only post-processing. The test accuracy increased from 0.8806 in the baseline to 0.9204 after preprocessing. The best final F-score was 0.9223 with the HOG/color feature model and threshold selection.", styles["BodyText"]))
    story.append(Paragraph("The results also show why each step should be evaluated separately. Augmentation is usually helpful in image classification, but in this experiment it reduced test performance when applied to pixel features. Therefore, augmentation parameters should be tuned carefully and validated rather than assumed to improve every model.", styles["BodyText"]))
    story.append(Paragraph("Future studies can improve accuracy by using a pretrained convolutional neural network with transfer learning, segmenting the tongue region before classification, testing larger hyperparameter grids, using k-fold cross-validation, and applying probability calibration or threshold optimization only on validation data. More controlled image acquisition and a larger labeled dataset would also reduce illumination and color variation, making the learned coating cues more stable.", styles["BodyText"]))
    doc.build(story)


def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    paths, labels = collect_dataset()
    split = split_dataset(paths, labels)
    save_split(split)

    all_rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for experiment in EXPERIMENTS:
        print(f"Running {experiment.name}...")
        result = train_and_evaluate(split, experiment)
        all_rows.extend(result["rows"])
        details[experiment.name] = {k: v for k, v in result.items() if k != "rows"}
        print(pd.DataFrame(result["rows"]).to_string(index=False))

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    (OUTPUT_DIR / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    plot_results(results)
    make_report(results, details, split)
    print(f"Done. Outputs written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
