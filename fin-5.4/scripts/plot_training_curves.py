#!/usr/bin/env python3
import argparse
import json
from html import escape
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


TRAIN_LOSS_KEYS = ("loss", "train_loss", "train/loss")
EVAL_LOSS_KEYS = ("eval_loss", "eval/loss")
LR_KEYS = ("learning_rate", "lr", "train/learning_rate")
STEP_KEYS = ("step", "global_step", "train/global_step", "global_step/max_steps")


def parse_step(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.split("/", 1)[0]
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def pick_number(payload: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            continue
    return None


def pick_step(payload: dict) -> int | None:
    for key in STEP_KEYS:
        step = parse_step(payload.get(key))
        if step is not None:
            return step
    return None


def load_jsonl_metrics(path: Path) -> dict[str, list[tuple[int, float]]]:
    train_loss: dict[int, float] = {}
    eval_loss: dict[int, float] = {}
    learning_rate: dict[int, float] = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            step = pick_step(payload)
            if step is None:
                continue

            train_value = pick_number(payload, TRAIN_LOSS_KEYS)
            eval_value = pick_number(payload, EVAL_LOSS_KEYS)
            lr_value = pick_number(payload, LR_KEYS)

            if train_value is not None:
                train_loss[step] = train_value
            if eval_value is not None:
                eval_loss[step] = eval_value
            if lr_value is not None:
                learning_rate[step] = lr_value

    return {
        "train_loss": sorted(train_loss.items()),
        "eval_loss": sorted(eval_loss.items()),
        "learning_rate": sorted(learning_rate.items()),
    }


def load_trainer_state_metrics(path: Path) -> dict[str, list[tuple[int, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("log_history", [])
    train_loss: dict[int, float] = {}
    eval_loss: dict[int, float] = {}
    learning_rate: dict[int, float] = {}

    for row in history:
        if not isinstance(row, dict):
            continue
        step = pick_step(row)
        if step is None:
            continue

        train_value = pick_number(row, TRAIN_LOSS_KEYS)
        eval_value = pick_number(row, EVAL_LOSS_KEYS)
        lr_value = pick_number(row, LR_KEYS)

        if train_value is not None:
            train_loss[step] = train_value
        if eval_value is not None:
            eval_loss[step] = eval_value
        if lr_value is not None:
            learning_rate[step] = lr_value

    return {
        "train_loss": sorted(train_loss.items()),
        "eval_loss": sorted(eval_loss.items()),
        "learning_rate": sorted(learning_rate.items()),
    }


def moving_average(series: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(series) <= 2:
        return series
    smoothed: list[tuple[int, float]] = []
    for idx in range(len(series)):
        start = max(0, idx - window + 1)
        chunk = [value for _, value in series[start : idx + 1]]
        smoothed.append((series[idx][0], sum(chunk) / len(chunk)))
    return smoothed


def infer_metric_paths(run_dir: Path) -> tuple[Path | None, Path | None]:
    jsonl_candidates = (
        run_dir / "logging_enriched.jsonl",
        run_dir / "logging.jsonl",
    )
    trainer_state_candidates = (
        run_dir / "trainer_state.json",
        run_dir / "checkpoint-last" / "trainer_state.json",
    )

    jsonl_path = next((path for path in jsonl_candidates if path.exists()), None)
    trainer_state_path = next((path for path in trainer_state_candidates if path.exists()), None)

    if trainer_state_path is None:
        checkpoints = sorted(
            [path for path in run_dir.glob("checkpoint-*") if path.is_dir()],
            key=lambda path: int(path.name.split("-")[-1]),
        )
        for checkpoint in reversed(checkpoints):
            candidate = checkpoint / "trainer_state.json"
            if candidate.exists():
                trainer_state_path = candidate
                break

    return jsonl_path, trainer_state_path


def merge_metrics(primary: dict[str, list[tuple[int, float]]], fallback: dict[str, list[tuple[int, float]]]) -> dict[str, list[tuple[int, float]]]:
    merged: dict[str, list[tuple[int, float]]] = {}
    for key in ("train_loss", "eval_loss", "learning_rate"):
        chosen = primary.get(key) or fallback.get(key) or []
        merged[key] = chosen
    return merged


def build_polyline(
    series: list[tuple[int, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    if not series:
        return ""
    points: list[str] = []
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-12)
    for step, value in series:
        x = left + ((step - x_min) / x_span) * width
        y = top + height - ((value - y_min) / y_span) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def save_svg_plot(
    metrics: dict[str, list[tuple[int, float]]],
    title: str,
    output_path: Path,
    smooth_window: int,
    best_step: int | None,
) -> None:
    width = 1320
    height = 560
    margin = 48
    header = 60
    gutter = 36
    panel_width = (width - margin * 2 - gutter) / 2
    panel_height = height - margin - header - 40
    panel_top = header
    left_panel = margin
    right_panel = margin + panel_width + gutter

    train_loss = moving_average(metrics["train_loss"], smooth_window)
    eval_loss = moving_average(metrics["eval_loss"], max(3, smooth_window // 4))
    learning_rate = metrics["learning_rate"]

    all_loss_steps = [step for step, _ in train_loss + eval_loss] or [0, 1]
    all_loss_values = [value for _, value in train_loss + eval_loss] or [0.0, 1.0]
    loss_x_min = min(all_loss_steps)
    loss_x_max = max(all_loss_steps)
    loss_y_min = min(all_loss_values)
    loss_y_max = max(all_loss_values)
    if loss_y_min == loss_y_max:
        loss_y_max = loss_y_min + 1.0

    lr_steps = [step for step, _ in learning_rate] or [0, 1]
    lr_values = [value for _, value in learning_rate] or [0.0, 1.0]
    lr_x_min = min(lr_steps)
    lr_x_max = max(lr_steps)
    lr_y_min = min(lr_values)
    lr_y_max = max(lr_values)
    if lr_y_min == lr_y_max:
        lr_y_max = lr_y_min + 1.0

    train_polyline = build_polyline(
        train_loss, loss_x_min, loss_x_max, loss_y_min, loss_y_max, left_panel, panel_top, panel_width, panel_height
    )
    eval_polyline = build_polyline(
        eval_loss, loss_x_min, loss_x_max, loss_y_min, loss_y_max, left_panel, panel_top, panel_width, panel_height
    )
    lr_polyline = build_polyline(
        learning_rate, lr_x_min, lr_x_max, lr_y_min, lr_y_max, right_panel, panel_top, panel_width, panel_height
    )

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#f6f4ef"/>',
        f'<text x="{margin}" y="34" font-size="26" font-weight="700" fill="#222">{escape(title)}</text>',
        f'<text x="{margin}" y="54" font-size="13" fill="#555">Rendered from trainer_state.json</text>',
    ]

    for panel_x, panel_title in ((left_panel, "Loss"), (right_panel, "Learning Rate")):
        svg.extend(
            [
                f'<rect x="{panel_x}" y="{panel_top}" width="{panel_width}" height="{panel_height}" rx="18" fill="#fffdf7" stroke="#ddd6c8"/>',
                f'<text x="{panel_x + 20}" y="{panel_top + 28}" font-size="18" font-weight="700" fill="#222">{panel_title}</text>',
            ]
        )
        for idx in range(1, 5):
            y = panel_top + panel_height * idx / 5
            svg.append(
                f'<line x1="{panel_x + 18}" y1="{y:.2f}" x2="{panel_x + panel_width - 18}" y2="{y:.2f}" stroke="#e5dfd3" stroke-width="1"/>'
            )

    if best_step is not None and loss_x_max > loss_x_min:
        x = left_panel + ((best_step - loss_x_min) / max(loss_x_max - loss_x_min, 1.0)) * panel_width
        svg.extend(
            [
                f'<line x1="{x:.2f}" y1="{panel_top + 18}" x2="{x:.2f}" y2="{panel_top + panel_height - 18}" stroke="#9c6644" stroke-width="2" stroke-dasharray="7 5"/>',
                f'<text x="{x + 8:.2f}" y="{panel_top + 26}" font-size="12" fill="#9c6644">best step {best_step}</text>',
            ]
        )

    if train_polyline:
        svg.append(f'<polyline fill="none" stroke="#d1495b" stroke-width="3" points="{train_polyline}"/>')
    if eval_polyline:
        svg.append(f'<polyline fill="none" stroke="#2e86ab" stroke-width="3" points="{eval_polyline}"/>')
    if lr_polyline:
        svg.append(f'<polyline fill="none" stroke="#3f7d20" stroke-width="3" points="{lr_polyline}"/>')
    else:
        svg.append(
            f'<text x="{right_panel + panel_width / 2:.2f}" y="{panel_top + panel_height / 2:.2f}" text-anchor="middle" font-size="16" fill="#666">No LR metric found</text>'
        )

    svg.extend(
        [
            f'<circle cx="{left_panel + 24}" cy="{height - 30}" r="6" fill="#d1495b"/>',
            f'<text x="{left_panel + 38}" y="{height - 25}" font-size="13" fill="#333">train loss</text>',
            f'<circle cx="{left_panel + 138}" cy="{height - 30}" r="6" fill="#2e86ab"/>',
            f'<text x="{left_panel + 152}" y="{height - 25}" font-size="13" fill="#333">eval loss</text>',
            f'<circle cx="{right_panel + 24}" cy="{height - 30}" r="6" fill="#3f7d20"/>',
            f'<text x="{right_panel + 38}" y="{height - 25}" font-size="13" fill="#333">learning rate</text>',
        ]
    )
    svg.append("</svg>")

    output_path.write_text("\n".join(svg), encoding="utf-8")


def plot_series(ax, series: list[tuple[int, float]], label: str, color: str, smooth_window: int) -> None:
    if not series:
        return
    xs = [step for step, _ in series]
    ys = [value for _, value in series]
    ax.plot(xs, ys, color=color, alpha=0.25, linewidth=1.2)
    smoothed = moving_average(series, smooth_window)
    ax.plot(
        [step for step, _ in smoothed],
        [value for _, value in smoothed],
        color=color,
        linewidth=2.0,
        label=label,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render training curves from ms-swift or HF trainer logs.")
    parser.add_argument("--run-dir", help="Training run directory containing logging.jsonl or trainer_state.json.")
    parser.add_argument("--log-file", help="Path to logging.jsonl or logging_enriched.jsonl.")
    parser.add_argument("--trainer-state", help="Path to trainer_state.json.")
    parser.add_argument("--output", required=True, help="Output image path, for example assets/training_curves.png.")
    parser.add_argument("--title", default="Training Curves", help="Figure title.")
    parser.add_argument("--smooth-window", type=int, default=25, help="Moving-average window for displayed curves.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    jsonl_metrics = {"train_loss": [], "eval_loss": [], "learning_rate": []}
    trainer_metrics = {"train_loss": [], "eval_loss": [], "learning_rate": []}

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        jsonl_path, trainer_state_path = infer_metric_paths(run_dir)
    else:
        jsonl_path = Path(args.log_file).expanduser().resolve() if args.log_file else None
        trainer_state_path = (
            Path(args.trainer_state).expanduser().resolve() if args.trainer_state else None
        )

    if jsonl_path and jsonl_path.exists():
        jsonl_metrics = load_jsonl_metrics(jsonl_path)
    if trainer_state_path and trainer_state_path.exists():
        trainer_metrics = load_trainer_state_metrics(trainer_state_path)

    metrics = merge_metrics(jsonl_metrics, trainer_metrics)
    if not any(metrics.values()):
        raise SystemExit("No plottable metrics found. Provide logging.jsonl or trainer_state.json.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_step = None
    if trainer_state_path and trainer_state_path.exists():
        payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        best_step = parse_step(payload.get("best_global_step"))

    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
        fig.patch.set_facecolor("#f6f4ef")
        for ax in axes:
            ax.set_facecolor("#fffdf7")
            ax.grid(True, alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plot_series(axes[0], metrics["train_loss"], "train loss", "#d1495b", args.smooth_window)
        plot_series(axes[0], metrics["eval_loss"], "eval loss", "#2e86ab", max(3, args.smooth_window // 4))
        if best_step is not None:
            axes[0].axvline(best_step, color="#9c6644", linestyle="--", linewidth=1.5, label=f"best step {best_step}")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        if metrics["train_loss"] or metrics["eval_loss"] or best_step is not None:
            axes[0].legend(frameon=False)

        plot_series(axes[1], metrics["learning_rate"], "learning rate", "#3f7d20", 1)
        axes[1].set_title("Learning Rate")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("LR")
        if metrics["learning_rate"]:
            axes[1].legend(frameon=False)
        else:
            axes[1].text(0.5, 0.5, "No LR metric found", ha="center", va="center", transform=axes[1].transAxes)

        fig.suptitle(args.title, fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
    else:
        save_svg_plot(metrics, args.title, output_path, args.smooth_window, best_step)

    print(f"Saved figure to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
