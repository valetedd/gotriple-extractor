#!/usr/bin/env python3
"""
Analyze benchmark_results/combined_results.jsonl and compute traditional metrics.

Outputs:
  - CSV tables in benchmark/benchmark_results/analysis
  - Figures in benchmark/benchmark_results/figures

The script is streaming-friendly and does not load the full JSONL into memory.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None


# Optional language detection (best effort)
HAVE_LANGDETECT = True
try:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 0
    HAVE_LANGDETECT = True
except Exception:
    HAVE_LANGDETECT = False


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float


def prf(correct: int, pred: int, gold: int) -> PRF:
    p = correct / pred if pred else 0.0
    r = correct / gold if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return PRF(p, r, f1)


def macro_f1(stats_by_label: Dict[str, Dict[str, int]]) -> float:
    if not stats_by_label:
        return 0.0
    f1_vals = [prf(c["correct"], c["pred"], c["gold"]).f1 for c in stats_by_label.values()]
    if not f1_vals:
        return 0.0
    if np is not None:
        return float(np.mean(f1_vals))
    return float(sum(f1_vals) / len(f1_vals))


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_gold(ents: Optional[list]) -> List[Tuple[str, str]]:
    if not ents:
        return []
    out: List[Tuple[str, str]] = []
    for ent in ents:
        if isinstance(ent, (list, tuple)) and len(ent) >= 2:
            text, label = ent[0], ent[1]
        elif isinstance(ent, dict):
            text = ent.get("text") or ent.get("entity") or ent.get("mention")
            label = ent.get("label")
        else:
            continue
        if text is None or label is None:
            continue
        out.append((str(text), str(label)))
    return out


def normalize_pred(ents: Optional[list]) -> List[Tuple[str, str]]:
    if not ents:
        return []
    out: List[Tuple[str, str]] = []
    for ent in ents:
        if isinstance(ent, dict):
            text = ent.get("text")
            label = ent.get("label")
        elif isinstance(ent, (list, tuple)) and len(ent) >= 2:
            text, label = ent[0], ent[1]
        else:
            continue
        if text is None or label is None:
            continue
        out.append((str(text), str(label)))
    return out


def bucket_len(n: int) -> str:
    if n <= 3:
        return "1-3"
    if n <= 7:
        return "4-7"
    if n <= 15:
        return "8-15"
    if n <= 30:
        return "16-30"
    return "31+"


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze benchmark JSONL results.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark/benchmark_results/combined_results.jsonl"),
        help="Path to combined_results.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark/benchmark_results/analysis"),
        help="Directory for CSV/JSON outputs",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("benchmark/benchmark_results/figures"),
        help="Directory for plots",
    )
    parser.add_argument(
        "--detect-language",
        action="store_true",
        help="Attempt language detection (requires langdetect)",
    )
    parser.add_argument(
        "--top-labels",
        type=int,
        default=20,
        help="Top labels by gold support to plot",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    out_dir: Path = args.out_dir
    fig_dir: Path = args.fig_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    ensure_dirs(out_dir, fig_dir)

    totals = defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0})
    by_dataset = defaultdict(
        lambda: defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0})
    )
    by_dataset_label = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0}))
    )
    by_label = defaultdict(
        lambda: defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0})
    )
    by_length = defaultdict(
        lambda: defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0})
    )
    by_lang = defaultdict(
        lambda: defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0})
    )
    by_lang_label = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"gold": 0, "pred": 0, "correct": 0}))
    )
    confusion = defaultdict(Counter)  # model -> Counter[(gold_label, pred_label)]

    models_seen: set[str] = set()

    for rec in iter_jsonl(input_path):
        gold_raw = rec.get("gold_entities", [])
        gold_norm = normalize_gold(gold_raw)
        gold_counter = Counter(gold_norm)

        dataset = rec.get("source", "unknown")

        lang = None
        if args.detect_language and HAVE_LANGDETECT:
            try:
                lang = detect(rec.get("text", "") or "")
            except Exception:
                lang = None

        pred_keys = [k for k in rec.keys() if k.startswith("preds_")]
        if not pred_keys:
            continue

        for pred_key in pred_keys:
            model = pred_key.replace("preds_", "")
            models_seen.add(model)

            pred_norm = normalize_pred(rec.get(pred_key, []))
            pred_counter = Counter(pred_norm)

            correct = 0
            for k in gold_counter:
                if k in pred_counter:
                    correct += min(gold_counter[k], pred_counter[k])

            gold_n = sum(gold_counter.values())
            pred_n = sum(pred_counter.values())

            totals[model]["gold"] += gold_n
            totals[model]["pred"] += pred_n
            totals[model]["correct"] += correct

            by_dataset[model][dataset]["gold"] += gold_n
            by_dataset[model][dataset]["pred"] += pred_n
            by_dataset[model][dataset]["correct"] += correct

            if lang:
                by_lang[model][lang]["gold"] += gold_n
                by_lang[model][lang]["pred"] += pred_n
                by_lang[model][lang]["correct"] += correct

            # Label-level aggregates
            for (text, label), count in gold_counter.items():
                by_label[model][label]["gold"] += count
                by_dataset_label[model][dataset][label]["gold"] += count
                if lang:
                    by_lang_label[model][lang][label]["gold"] += count
                by_length[model][bucket_len(len(text))]["gold"] += count

            for (text, label), count in pred_counter.items():
                by_label[model][label]["pred"] += count
                by_dataset_label[model][dataset][label]["pred"] += count
                if lang:
                    by_lang_label[model][lang][label]["pred"] += count
                by_length[model][bucket_len(len(text))]["pred"] += count

            for text, label in gold_counter:
                if (text, label) in pred_counter:
                    c = min(gold_counter[(text, label)], pred_counter[(text, label)])
                    by_label[model][label]["correct"] += c
                    by_dataset_label[model][dataset][label]["correct"] += c
                    if lang:
                        by_lang_label[model][lang][label]["correct"] += c
                    by_length[model][bucket_len(len(text))]["correct"] += c

            # Confusion: same surface form, different label (best-effort)
            gold_by_text: Dict[str, Counter] = defaultdict(Counter)
            pred_by_text: Dict[str, Counter] = defaultdict(Counter)
            for text, label in gold_norm:
                gold_by_text[text][label] += 1
            for text, label in pred_norm:
                pred_by_text[text][label] += 1

            for text, pred_labels in pred_by_text.items():
                if text not in gold_by_text:
                    continue
                gold_labels = gold_by_text[text]
                primary_gold = max(gold_labels.items(), key=lambda x: x[1])[0]
                for pred_label, pred_count in pred_labels.items():
                    if pred_label == primary_gold:
                        continue
                    confusion[model][(primary_gold, pred_label)] += pred_count

    # Build summary tables
    summary_rows = []
    for model, c in totals.items():
        metrics = prf(c["correct"], c["pred"], c["gold"])
        summary_rows.append(
            {
                "model": model,
                "gold": c["gold"],
                "pred": c["pred"],
                "correct": c["correct"],
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "macro_f1": macro_f1(by_label[model]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(by="f1", ascending=False)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    dataset_rows = []
    for model, ds_map in by_dataset.items():
        for ds, c in ds_map.items():
            metrics = prf(c["correct"], c["pred"], c["gold"])
            dataset_rows.append(
                {
                    "model": model,
                    "dataset": ds,
                    "gold": c["gold"],
                    "pred": c["pred"],
                    "correct": c["correct"],
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "macro_f1": macro_f1(by_dataset_label[model][ds]),
                }
            )
    dataset_df = pd.DataFrame(dataset_rows).sort_values(
        by=["dataset", "f1"], ascending=[True, False]
    )
    dataset_df.to_csv(out_dir / "by_dataset.csv", index=False)

    label_rows = []
    for model, lab_map in by_label.items():
        for label, c in lab_map.items():
            metrics = prf(c["correct"], c["pred"], c["gold"])
            label_rows.append(
                {
                    "model": model,
                    "label": label,
                    "gold": c["gold"],
                    "pred": c["pred"],
                    "correct": c["correct"],
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )
    label_df = pd.DataFrame(label_rows)
    label_df = label_df.sort_values(by=["gold", "f1"], ascending=[False, False])
    label_df.to_csv(out_dir / "by_label.csv", index=False)

    length_rows = []
    for model, b_map in by_length.items():
        for bucket, c in b_map.items():
            metrics = prf(c["correct"], c["pred"], c["gold"])
            length_rows.append(
                {
                    "model": model,
                    "bucket": bucket,
                    "gold": c["gold"],
                    "pred": c["pred"],
                    "correct": c["correct"],
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )
    length_df = pd.DataFrame(length_rows)
    length_df.to_csv(out_dir / "by_length_bucket.csv", index=False)

    if by_lang:
        lang_rows = []
        for model, lmap in by_lang.items():
            for lang, c in lmap.items():
                metrics = prf(c["correct"], c["pred"], c["gold"])
                lang_rows.append(
                    {
                        "model": model,
                        "lang": lang,
                        "gold": c["gold"],
                        "pred": c["pred"],
                        "correct": c["correct"],
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1": metrics.f1,
                        "macro_f1": macro_f1(by_lang_label[model][lang]),
                    }
                )
        lang_df = pd.DataFrame(lang_rows).sort_values(
            by=["lang", "f1"], ascending=[True, False]
        )
        lang_df.to_csv(out_dir / "by_language.csv", index=False)

    # Variance across labels (per model)
    variance_rows = []
    for model in label_df["model"].unique():
        sub = label_df[label_df["model"] == model]
        if sub.empty:
            continue
        f1_vals = sub["f1"].values
        if np is not None:
            mean_f1 = float(np.mean(f1_vals))
            std_f1 = float(np.std(f1_vals))
        else:
            mean_f1 = float(sum(f1_vals) / len(f1_vals))
            std_f1 = float(
                math.sqrt(sum((x - mean_f1) ** 2 for x in f1_vals) / len(f1_vals))
            )
        variance_rows.append(
            {
                "model": model,
                "labels": len(sub),
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "cv_f1": (std_f1 / mean_f1) if mean_f1 else 0.0,
            }
        )
    variance_df = pd.DataFrame(variance_rows).sort_values(by="cv_f1", ascending=False)
    variance_df.to_csv(out_dir / "label_variance.csv", index=False)

    # Confusion summary
    conf_rows = []
    for model, conf_counter in confusion.items():
        for (gold_label, pred_label), count in conf_counter.items():
            conf_rows.append(
                {
                    "model": model,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "count": count,
                }
            )
    conf_df = pd.DataFrame(conf_rows).sort_values(by="count", ascending=False)
    conf_df.to_csv(out_dir / "label_confusions.csv", index=False)

    # Quick console summary
    print("Models found:", ", ".join(sorted(models_seen)) if models_seen else "none")
    print(summary_df.to_string(index=False))

    # Visualizations
    if plt is None:
        print("matplotlib not available: skipping plots")
        return 0

    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("ggplot")

    # 1) F1 by dataset
    if not dataset_df.empty:
        pivot = dataset_df.pivot(index="dataset", columns="model", values="f1")
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title("F1 by Dataset")
        ax.set_ylabel("F1")
        ax.set_xlabel("Dataset")
        ax.legend(title="Model", loc="best")
        plt.tight_layout()
        plt.savefig(fig_dir / "f1_by_dataset.png", dpi=200)
        plt.close()

    # 2) Macro-F1 by dataset
    if not dataset_df.empty:
        pivot = dataset_df.pivot(index="dataset", columns="model", values="macro_f1")
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title("Macro-F1 by Dataset")
        ax.set_ylabel("Macro-F1")
        ax.set_xlabel("Dataset")
        ax.legend(title="Model", loc="best")
        plt.tight_layout()
        plt.savefig(fig_dir / "macro_f1_by_dataset.png", dpi=200)
        plt.close()

    # 3) F1 by label (top labels by gold support, across models)
    if not label_df.empty:
        top_labels = (
            label_df.groupby("label")["gold"]
            .sum()
            .sort_values(ascending=False)
            .head(args.top_labels)
            .index
        )
        sub = label_df[label_df["label"].isin(top_labels)]
        pivot = sub.pivot(index="label", columns="model", values="f1").fillna(0.0)
        ax = pivot.plot(kind="bar", figsize=(12, 6))
        ax.set_title(f"F1 by Label (Top {len(top_labels)})")
        ax.set_ylabel("F1")
        ax.set_xlabel("Label")
        ax.legend(title="Model", loc="best")
        plt.tight_layout()
        plt.savefig(fig_dir / "f1_by_label_top.png", dpi=200)
        plt.close()

    # 4) Label support vs F1
    if not label_df.empty:
        plt.figure(figsize=(7, 5))
        for model in label_df["model"].unique():
            sub = label_df[label_df["model"] == model]
            plt.scatter(
                sub["gold"],
                sub["f1"],
                alpha=0.6,
                label=model,
            )
        plt.xscale("log")
        plt.title("Label Support vs F1")
        plt.xlabel("Gold Support (log scale)")
        plt.ylabel("F1")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "label_support_vs_f1.png", dpi=200)
        plt.close()

    # 5) Length bucket analysis
    if not length_df.empty:
        pivot = length_df.pivot(index="bucket", columns="model", values="f1").fillna(
            0.0
        )
        ax = pivot.plot(kind="bar", figsize=(8, 5))
        ax.set_title("F1 by Entity Length Bucket")
        ax.set_xlabel("Length (chars)")
        ax.set_ylabel("F1")
        ax.legend(title="Model", loc="best")
        plt.tight_layout()
        plt.savefig(fig_dir / "f1_by_length_bucket.png", dpi=200)
        plt.close()

    # 6) Confusion heatmap (top labels)
    if not conf_df.empty:
        # Keep frequency order so the heatmap aligns with expectations.
        top_gold = (
            conf_df.groupby("gold_label")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )
        top_pred = (
            conf_df.groupby("pred_label")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )
        for model in conf_df["model"].unique():
            sub = conf_df[
                (conf_df["model"] == model)
                & (conf_df["gold_label"].isin(top_gold))
                & (conf_df["pred_label"].isin(top_pred))
            ]
            if sub.empty:
                continue
            mat = sub.pivot_table(
                index="gold_label",
                columns="pred_label",
                values="count",
                aggfunc="sum",
                fill_value=0,
            )
            mat = mat.reindex(index=top_gold, columns=top_pred).fillna(0)
            fig, ax = plt.subplots(figsize=(8, 6))
            if sns is not None:
                sns.heatmap(
                    mat,
                    annot=False,
                    cmap="Reds",
                    linewidths=0.5,
                    linecolor="white",
                    square=True,
                    cbar=True,
                    cbar_kws={"label": "Count"},
                    ax=ax,
                )
            else:
                im = ax.imshow(mat.values, aspect="auto", cmap="Reds")
                ax.set_xticks(range(len(mat.columns)))
                ax.set_xticklabels(mat.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(mat.index)))
                ax.set_yticklabels(mat.index)
                fig.colorbar(im, ax=ax, label="Count")
            ax.set_title(f"Label Confusions (Top 10) — {model}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Gold Label")
            fig.tight_layout()
            fig.savefig(fig_dir / f"label_confusions_{model}.png", dpi=200)
            plt.close()

    # 7) Precision/Recall by model
    if not summary_df.empty:
        x = range(len(summary_df))
        plt.figure(figsize=(8, 4))
        plt.bar(
            [i - 0.2 for i in x], summary_df["precision"], width=0.4, label="Precision"
        )
        plt.bar([i + 0.2 for i in x], summary_df["recall"], width=0.4, label="Recall")
        plt.xticks(list(x), summary_df["model"].tolist(), rotation=0)
        plt.ylim(0, 1.0)
        plt.title("Precision and Recall by Model")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "precision_recall_by_model.png", dpi=200)
        plt.close()

    # 8) Micro-F1 vs Macro-F1 by model
    if not summary_df.empty:
        x = range(len(summary_df))
        plt.figure(figsize=(8, 4))
        plt.bar([i - 0.2 for i in x], summary_df["f1"], width=0.4, label="Micro-F1")
        plt.bar([i + 0.2 for i in x], summary_df["macro_f1"], width=0.4, label="Macro-F1")
        plt.xticks(list(x), summary_df["model"].tolist(), rotation=0)
        plt.ylim(0, 1.0)
        plt.title("Micro-F1 and Macro-F1 by Model")
        plt.ylabel("Score")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "micro_vs_macro_f1_by_model.png", dpi=200)
        plt.close()

    print(f"Wrote analysis to {out_dir}")
    print(f"Wrote figures to {fig_dir}")
    if args.detect_language and not HAVE_LANGDETECT:
        print("Language detection requested but langdetect is not installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
