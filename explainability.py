"""
explainability.py
-----------------
Explainability and visualisation methods for the multimodal affective pipeline.

Provides tools for interpreting and explaining model predictions, including
feature importance ranking, saliency maps, attention visualization, SHAP and
LIME-based explanations, and modality contribution analysis across
physiological, audio, visual, textual, and web-based input streams.

Visualisation functions ported from:
    - ai-fresco-replicated/fresco/utils/visualization/distribution_analysis.py
    - ai-fresco-replicated/fresco/utils/visualization/radar_plot.py
    - ai-occulo-video-insights/emotion/plots.py
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ===========================================================================
# Feature distribution visualisation  (FRESCO distribution_analysis.py)
# ===========================================================================

def plot_feature_distribution(
    data: dict[str, list[float]],
    output_path: str | None = None,
    title: str = "Feature Distribution",
    width: int = 1200,
    height: int = 1300,
    font_size: int = 18,
) -> Any:
    """Plot a horizontal violin chart of feature value distributions.

    Each key in *data* becomes one violin on the y-axis.  Useful for
    visualising the spread of semiotic distance metrics or acoustic features
    across a dataset.

    Args:
        data:        ``{feature_name: [values]}`` dict.
        output_path: Optional path to save the figure (PDF / PNG / SVG).
                     When ``None`` the figure is returned without saving.
        title:       Figure title.
        width:       Figure width in pixels.
        height:      Figure height in pixels.
        font_size:   Global font size.

    Returns:
        A ``plotly.graph_objects.Figure`` object.

    Requires ``plotly`` and optionally ``kaleido`` for export.
    """
    import plotly.graph_objects as go
    from plotly.colors import n_colors

    keys   = list(data.keys())
    colors = n_colors("rgb(154,157,246)", "rgb(202,185,105)",
                      max(len(keys), 2), colortype="rgb")

    fig = go.Figure()
    for (k, vals), color in zip(data.items(), colors):
        fig.add_trace(go.Violin(x=vals, line_color=color, name=k))

    fig.update_traces(orientation="h", side="positive", width=5, points=False)
    fig.update_layout(
        title=title,
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        plot_bgcolor="white",
        font_size=font_size,
    )
    fig.update_xaxes(autorangeoptions_clipmax=1.18, autorangeoptions_clipmin=-0.18)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out), scale=3, width=width, height=height)
        logger.info("Saved distribution plot → %s", out)

    return fig


# ===========================================================================
# Radar / spider plots  (FRESCO radar_plot.py)
# ===========================================================================

def plot_radar(
    scores_a: list[float],
    labels: list[str],
    scores_b: list[float] | None = None,
    title: str = "Radar Plot",
    label_a: str = "Item A",
    label_b: str = "Item B",
    value_range: tuple[float, float] = (0.0, 1.0),
    output_path: str | None = None,
    font_size: int = 16,
) -> Any:
    """Draw a radar / spider chart of per-dimension scores.

    When *scores_b* is provided, two traces are overlaid (useful for pairwise
    semiotic distance visualisation).  When it is ``None``, a single trace
    against the maximum reference (all-ones) is shown.

    Args:
        scores_a:    Values for the first trace (or the only trace).
        labels:      Category names (axis labels), same length as *scores_a*.
        scores_b:    Optional values for the second trace.
        title:       Chart title.
        label_a:     Legend label for the first trace.
        label_b:     Legend label for the second trace.
        value_range: ``(min, max)`` of the radial axis.
        output_path: Optional path to save the figure.
        font_size:   Angular axis tick font size.

    Returns:
        A ``plotly.graph_objects.Figure``.

    Requires ``plotly``.
    """
    import plotly.graph_objects as go

    ref_b = scores_b if scores_b is not None else [value_range[1]] * len(labels)

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=scores_a, theta=labels, fill="toself",
                name=label_a, line_color="blue", opacity=0.5,
            ),
            go.Scatterpolar(
                r=ref_b, theta=labels, fill="toself",
                name=label_b, line_color="gold", fillcolor="gold", opacity=0.6,
            ),
        ],
        layout=go.Layout(
            title=go.layout.Title(text=title, x=0.5, font_size=int(font_size * 2)),
            polar={
                "radialaxis": {
                    "visible": True,
                    "range":   list(value_range),
                    "linecolor": "gray",
                },
                "angularaxis": {"tickfont": {"size": font_size}},
            },
            showlegend=True,
        ),
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out), scale=3, width=1024, height=800)
        logger.info("Saved radar plot → %s", out)

    return fig


def plot_pairwise_distances(
    distance_file: str,
    output_dir: str = "./output_images/radar_plot",
) -> None:
    """Generate a full suite of radar plots from a FRESCO pairwise distance JSON.

    Reproduces the radar-plot battery from ``radar_plot.py``: chromatic
    categories, topological categories, content participants, face attributes,
    emotion, watcher-looked systems, semiotic levels, and overall similarity.

    Args:
        distance_file: Path to the FRESCO distance JSON.
        output_dir:    Directory where all radar images are saved.

    Requires ``plotly`` and ``kaleido``.
    """
    import json

    if not os.path.isfile(distance_file):
        raise FileNotFoundError(f"Distance file not found: {distance_file}")

    with open(distance_file) as fh:
        d = json.load(fh)

    def _save(scores_a, labels, title, scores_b=None):
        out = os.path.join(output_dir, f"{title}.jpg")
        plot_radar(scores_a, labels, scores_b=scores_b, title=title, output_path=out)

    pl = d["plastic_level"]
    ch = pl["chromatic_categories"]
    tp = pl["topological_categories"]
    fl = d["figurative_level"]
    nl = d["narrative_level"]

    # Chromatic
    _save(
        [ch["colors"]["is_grayscale"], ch["brightness"], ch["saturation"],
         ch["colors"]["palette"], ch["colors"]["color_distribution"]],
        ["Grayscale", "Brightness", "Saturation", "Palette", "Color distribution"],
        "Chromatic categories",
    )
    # Topological
    _save(
        [tp["obj_positions"]["vertical_ratio"][1],
         tp["obj_positions"]["horizontal_ratio"][1],
         tp["obj_centralities"][1],
         tp["person_distances_from_camera"][1],
         tp["mc_avg_depth_distance"][1],
         tp["background_avg_depth_distance"],
         tp["semantic_palette"]],
        ["Object positions (v)", "Object positions (h)", "Object centralities",
         "Person distances from camera", "MC avg depth",
         "Background avg depth", "Semantic palette"],
        "Topological categories",
    )
    # Emotion
    em = fl["emotion"]
    _save(
        [em["intensity"]["mean_valence"], em["intensity"]["mean_arousal"],
         em["intensity"]["mean_intensity"], em["emotion"][1]],
        ["Valence", "Arousal", "Mean intensity", "Emotion"],
        "Emotion",
    )
    # Levels
    _save(
        [pl["mean_plastic_level"], fl["mean_figurative_level"],
         nl["mean_narrative_level"]],
        ["Plastic level", "Figurative level", "Enunciational level"],
        "Levels of analysis",
    )
    # Overall
    _save([d["Overall_distance"]], ["Overall similarity"], "Overall similarity")


# ===========================================================================
# Video emotion model plots  (ai-occulo-video-insights/emotion/plots.py)
# ===========================================================================

def plot_training_curves(
    train_losses: list[float],
    val_accs: list[float],
    fold: int,
    save_path: str | None = None,
) -> None:
    """Plot per-fold training loss and validation accuracy curves.

    Args:
        train_losses: List of per-epoch cross-entropy losses.
        val_accs:     List of per-epoch validation accuracies.
        fold:         Fold index (used in the title).
        save_path:    Optional path to save the figure (PNG, PDF, …).

    Requires ``matplotlib``.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(train_losses, color="steelblue")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"Fold {fold} — Train Loss")
    ax2.plot(val_accs, color="darkorange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Fold {fold} — Val Accuracy")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved training curves → %s", save_path)
    plt.close()


def plot_cv_summary(
    all_fold_train_losses: list[list[float]],
    all_fold_val_accs: list[list[float]],
    save_path: str | None = None,
) -> None:
    """Plot mean ± std cross-validated training loss and accuracy curves.

    Args:
        all_fold_train_losses: Per-fold loss lists.
        all_fold_val_accs:     Per-fold accuracy lists.
        save_path:             Optional save path.
    """
    import matplotlib.pyplot as plt

    max_len = max(len(x) for x in all_fold_train_losses)

    def _pad(series_list: list[list[float]]) -> np.ndarray:
        return np.array(
            [s + [s[-1]] * (max_len - len(s)) for s in series_list], dtype=float
        )

    losses = _pad(all_fold_train_losses)
    accs   = _pad(all_fold_val_accs)
    epochs = np.arange(1, max_len + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for arr, ax, color, ylabel, title in [
        (losses, ax1, "steelblue",  "Cross-Entropy Loss", "Train Loss — mean ± std"),
        (accs,   ax2, "darkorange", "Accuracy",           "Val Accuracy — mean ± std"),
    ]:
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(epochs, mean, color=color, linewidth=2, label="mean")
        ax.fill_between(epochs, mean - std, mean + std,
                        alpha=0.25, color=color, label="±1 std")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved CV summary → %s", save_path)
    plt.close()


def plot_confusion_matrix(
    labels_true: list[int],
    labels_pred: list[int],
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> None:
    """Plot a normalised confusion matrix.

    Args:
        labels_true:  Ground-truth integer class indices.
        labels_pred:  Predicted integer class indices.
        class_names:  Display names for each class.
        title:        Figure title.
        save_path:    Optional save path.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    if class_names is None:
        class_names = [str(i) for i in sorted(set(labels_true))]

    cm   = confusion_matrix(labels_true, labels_pred,
                            labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=class_names).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved confusion matrix → %s", save_path)
    plt.close()


def plot_model_comparison(
    results: dict[str, dict],
    save_path: str | None = None,
) -> None:
    """Bar chart comparing multiple model variants by accuracy.

    Args:
        results:   ``{model_key: {"label": str, "accuracy": float}}`` dict.
        save_path: Optional save path.
    """
    import matplotlib.pyplot as plt

    ordered = sorted(results.values(), key=lambda x: -x["accuracy"])
    labels  = [r["label"]    for r in ordered]
    accs    = [r["accuracy"] for r in ordered]

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(labels)), 5))
    x    = np.arange(len(labels))
    bars = ax.bar(x, accs, color="steelblue", edgecolor="white", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Model Benchmark — Accuracy (higher is better)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.01,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved model comparison → %s", save_path)
    plt.close()


def plot_temporal_prediction(
    result_df,
    video_name: str = "",
    save_path: str | None = None,
) -> None:
    """Plot per-second arousal predictions over time.

    Args:
        result_df:  DataFrame from ``inference.predict_temporal`` with columns
                    ``second``, ``quadrant``, ``confidence``, and per-class
                    probability columns.
        video_name: Title label.
        save_path:  Optional save path.
    """
    import matplotlib.pyplot as plt

    seconds = result_df["second"].tolist()
    conf    = result_df["confidence"].tolist()
    quads   = result_df["quadrant"].tolist()

    fig, ax = plt.subplots(figsize=(max(8, len(seconds) // 2), 4))
    ax.plot(seconds, conf, marker="o", color="steelblue", linewidth=1.5)
    for s, q, c in zip(seconds, quads, conf):
        ax.annotate(q, xy=(s, c), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xlabel("Second"); ax.set_ylabel("Confidence")
    ax.set_title(f"Temporal arousal predictions — {video_name}")
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved temporal prediction plot → %s", save_path)
    plt.close()


def plot_single_prediction(
    result: dict,
    video_path: str = "",
    save_path: str | None = None,
) -> None:
    """Bar chart of class probabilities for a single-clip prediction.

    Args:
        result:     Dict from ``inference.predict`` with ``probabilities``,
                    ``quadrant``, and ``confidence`` keys.
        video_path: Source video path (used in the title).
        save_path:  Optional save path.
    """
    import matplotlib.pyplot as plt

    probs  = result["probabilities"]
    names  = list(probs.keys())
    values = list(probs.values())
    colors = ["darkorange" if n == result["quadrant"] else "steelblue" for n in names]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Probability")
    ax.set_title(
        f"{os.path.basename(video_path)}\n"
        f"Predicted: {result['quadrant']}  ({result['confidence']:.1%})"
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Saved prediction plot → %s", save_path)
    plt.close()
