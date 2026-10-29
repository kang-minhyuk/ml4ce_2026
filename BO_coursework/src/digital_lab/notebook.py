"""Helpers for the student notebook: run several algorithm files and plot how they compare."""

import hashlib
import json
import re
import shutil
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl

from digital_lab.constants import DEFAULT_LAB_CONFIG_DIR, SEEDS
from digital_lab.experiment import aggregate_histories, per_seed_metrics, run_algorithm
from digital_lab.plotting import convergence_figure, fmt_mean_std
from digital_lab.submission import audit_submission, load_algorithm, validate_submission

# Relative to the notebook's folder (student/).
ALGORITHMS_DIR = Path("algorithms")
SAVED_DIR = ALGORITHMS_DIR / "saved"
RESULTS_DIR = Path("results")  # saved run results, reused while nothing changed
BUILTIN_ALGORITHMS = ("my_algorithm", "reference_screen_then_confirm", "reference_mfbo")

# Fixed color per scale in the scale-choice plot, and for the end-of-run marker.
SCALE_COLORS = {"MTP": "#1baf7a", "MBR": "#eda100", "Pilot": "#2a78d6"}
END_COLOR = "#d03b3b"

# Fixed colors per algorithm, so a line keeps its color whichever algorithms you select.
FIXED_COLORS = {"my_algorithm": "#2a78d6", "reference_screen_then_confirm": "#8a5cd1", "reference_mfbo": "#eb6834"}
VERSION_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#c43c3c", "#52514e"]


def check_my_algorithm(path: str | Path) -> bool:
    """Print whether a submission file passes the submission rules.

    Parameters
    ----------
    path : str or pathlib.Path
        The student's my_algorithm.py.

    Returns
    -------
    bool
        True if the file is valid for submission.
    """
    problems = validate_submission(path)
    notes = audit_submission(path)
    if notes:
        print(
            f"Note: {Path(path).name} contains code the GTA will review "
            "(it may be fine, but explain it if asked):"
        )
        for note in notes:
            print(f"  - {note}")
    if problems:
        print(f"WARNING: {Path(path).name} would be rejected at submission:")
        for problem in problems:
            print(f"  - {problem}")
        return False
    print(f"{Path(path).name} passes the submission checks.")
    return True


def save_version(name: str, source: str | Path = "algorithms/my_algorithm.py") -> Path:
    """Save a copy of your current algorithm under a name, so you can compare against it later.

    Parameters
    ----------
    name : str
        Version name, e.g. "v1_random_search". Letters, digits, "_" and "-" only.
    source : str or pathlib.Path, optional
        File to copy, by default your `algorithms/my_algorithm.py`.

    Returns
    -------
    pathlib.Path
        The saved file, `algorithms/saved/<name>.py`.

    Raises
    ------
    ValueError
        If `name` contains other characters or clashes with a built-in algorithm name.
    """
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in BUILTIN_ALGORITHMS:
        raise ValueError(
            f"Invalid version name {name!r}: use letters, digits, '_' and '-', and not one of {BUILTIN_ALGORITHMS}."
        )
    target = SAVED_DIR / f"{name}.py"
    if target.exists():
        print(f"Overwriting the saved version {name!r}.")
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"Saved {source} as version {name!r}. Saved versions: {list_versions()}")
    return target


def list_versions() -> list[str]:
    """Names of your saved algorithm versions.

    Returns
    -------
    list[str]
        Version names, sorted.
    """
    return sorted(p.stem for p in SAVED_DIR.glob("*.py"))


def _resolve(name: str) -> Path:
    """Find the file for an algorithm name: a built-in one in `algorithms/` or a saved version.

    Parameters
    ----------
    name : str
        "my_algorithm", "reference_screen_then_confirm", "reference_mfbo" or a saved version name.

    Returns
    -------
    pathlib.Path
        The algorithm file.

    Raises
    ------
    FileNotFoundError
        If no such algorithm exists.
    """
    for path in (ALGORITHMS_DIR / f"{name}.py", SAVED_DIR / f"{name}.py"):
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No algorithm called {name!r}. "
        f"Choose from {list(BUILTIN_ALGORITHMS)} or your saved versions {list_versions()}."
    )


def _fingerprint_of_lab_and_code() -> str:
    """Hash of everything besides the algorithm that a run's result depends on.

    That is the course code (lab, scoring, constants such as budget and batch sizes) and
    the lab's configuration files (clones, reactors, noise). If either changes, every
    saved result is out of date and gets re-run.

    Returns
    -------
    str
        Hex digest.
    """
    digest = hashlib.sha256()
    package_dir = Path(__file__).resolve().parent
    for path in sorted(package_dir.glob("*.py")) + sorted(DEFAULT_LAB_CONFIG_DIR.glob("*.json")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _result_files(name: str, path: Path, seed: int, lab_fingerprint: str) -> tuple:
    """Where the saved result of one (algorithm, seed) run lives.

    The file name contains a hash of the algorithm's code, the seed, and the lab/code
    fingerprint, so an edited algorithm, or a changed lab, never reuses an old result.

    Parameters
    ----------
    name : str
        Algorithm name (folder name).
    path : pathlib.Path
        Algorithm file.
    seed : int
        Seed of the run.
    lab_fingerprint : str
        From `_fingerprint_of_lab_and_code`.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        History file (parquet) and info file (json).
    """
    key = hashlib.sha256(path.read_bytes() + f"|{seed}|".encode() + lab_fingerprint.encode()).hexdigest()[:16]
    stem = RESULTS_DIR / name / f"seed{seed}_{key}"
    return stem.with_suffix(".parquet"), stem.with_suffix(".json")


def run_comparison(names: list[str], seeds: list[int] | None = None, rerun: bool = False) -> dict:
    """Run the named algorithms over the seeds, reusing saved results where nothing changed.

    Every run (one algorithm, one seed) is saved to `results/<name>/`. A later call loads
    it instead of running again, as long as the algorithm's code, the seed, the lab and the
    course code are all unchanged. So after editing `my_algorithm`, only it re-runs; the
    references and your saved versions are loaded instantly.

    Parameters
    ----------
    names : list[str]
        Algorithms to compare: "my_algorithm", "reference_screen_then_confirm", "reference_mfbo"
        and/or the names of versions saved with `save_version`.
    seeds : list[int] or None, optional
        Seeds to run, by default all of `SEEDS`.
    rerun : bool, optional
        Ignore saved results and run everything again, by default False.

    Returns
    -------
    dict
        Per-algorithm "per_seed", "aggregated", "infos" and "metrics".
    """
    seeds = SEEDS if seeds is None else seeds
    paths = {name: _resolve(name) for name in names}  # fail fast on a typo, before anything runs
    lab_fingerprint = _fingerprint_of_lab_and_code()
    runs = {}
    for name, path in paths.items():
        print(f"{name} on {len(seeds)} seed(s):")
        algorithm = None
        per_seed, infos = [], []
        for seed in seeds:
            history_file, info_file = _result_files(name, path, seed, lab_fingerprint)
            if not rerun and history_file.exists() and info_file.exists():
                history = pl.read_parquet(history_file)
                info = json.loads(info_file.read_text(encoding="utf-8"))
                source = "saved result"
            else:
                algorithm = algorithm or load_algorithm(path)
                history, info = run_algorithm(algorithm, name, seed)
                history_file.parent.mkdir(parents=True, exist_ok=True)
                history.write_parquet(history_file)
                info_file.write_text(json.dumps(info), encoding="utf-8")
                source = "ran"
            per_seed.append(history)
            infos.append(info)
            bests = " / ".join("-" if b is None else f"{b:.1f}" for b in info["checkpoint_bests"])
            status = " ERROR" if info["error"] else (" STOPPED AT TIME LIMIT" if info["time_limit_hit"] else "")
            print(
                f"  seed {seed}: score {info['score']:.1f} (best Pilot titer at 33/66/100 % of budget: {bests}), "
                f"{info['wall_time']:.1f} s ({source}){status}"
            )
        runs[name] = {"per_seed": per_seed, "aggregated": aggregate_histories(per_seed, name), "infos": infos}
        runs[name]["metrics"] = per_seed_metrics(per_seed, infos)

    _print_warnings(runs)
    return runs


def _print_warnings(runs: dict) -> None:
    """Print errors and time-limit violations.

    Parameters
    ----------
    runs : dict
        Output of `run_comparison`.
    """
    for name, run in runs.items():
        for seed, info in zip(run["per_seed"], run["infos"]):
            seed_id = seed["seed"][0] if seed.height else "?"
            if info["error"]:
                print(f"\nERROR in {name} (seed {seed_id}):\n{info['error']}")
            if info.get("access_warnings"):
                print(
                    f"\nWARNING: {name} (seed {seed_id}) tried to use parts of the lab that aren't public: "
                    f"{info['access_warnings']}. Only the documented lab attributes are available."
                )
            missing = [
                f"${checkpoint:,.0f}"
                for checkpoint, best in zip(info["score_checkpoints"], info["checkpoint_bests"])
                if best is None
            ]
            if missing:
                print(
                    f"\nWARNING: {name} (seed {seed_id}) had no Pilot result by the time it had spent {', '.join(missing)}. "
                    "Those checkpoints score 0: run a Pilot experiment before each checkpoint."
                )
            if info["time_limit_hit"]:
                print(
                    f"\nWARNING: {name} (seed {seed_id}) was stopped at the {info['time_limit']:.0f} s time limit. "
                    "It is graded on the Pilot titers found up to that point, and any budget left is wasted."
                )


def show_results(runs: dict) -> None:
    """Show a summary table (score, best Pilot titer per checkpoint) and, per scale, best-so-far plots.

    Parameters
    ----------
    runs : dict
        Output of `run_comparison`.
    """
    summary = pl.DataFrame([
        {
            "algorithm": name,
            "score": fmt_mean_std(*run["metrics"]["score"], precision=1),
            **{
                f"best at {key.removeprefix('best_at_')} %": fmt_mean_std(*value, precision=1)
                for key, value in run["metrics"].items() if key.startswith("best_at_")
            },
            "max run time [s]": round(max(info["wall_time"] for info in run["infos"]), 1),
        }
        for name, run in runs.items()
    ])
    with pl.Config(
        tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True, fmt_str_lengths=40, tbl_width_chars=200
    ):
        print(summary)

    versions = [name for name in runs if name not in FIXED_COLORS]
    colors = {**FIXED_COLORS, **{n: VERSION_COLORS[i % len(VERSION_COLORS)] for i, n in enumerate(versions)}}
    histories = {name: run["per_seed"] for name, run in runs.items()}
    for x_col in ("spent", "n_experiments"):
        checkpoints = next(iter(runs.values()))["infos"][0]["score_checkpoints"] if x_col == "spent" else None
        convergence_figure(histories, x_col, colors, checkpoints=checkpoints).show()


def show_scale_choices(runs: dict, names: list[str] | None = None) -> None:
    """Plot which scale each algorithm chose at every step (lab call), one row per seed.

    Batch 0 is the initial design. Hover over a square to see the batch's best titer and
    the total spent after it. A red square after the last batch marks the end of the run,
    labelled with the reason it ended (budget, experiment limit, time limit, error, or
    stopped early).

    Parameters
    ----------
    runs : dict
        Output of `run_comparison`.
    names : list[str] or None, optional
        Algorithms to show, by default all of them.
    """
    names = list(runs) if names is None else names
    fig = make_subplots(rows=len(names), cols=1, subplot_titles=names, shared_xaxes=True, vertical_spacing=0.12)
    shown = set()
    last = max((int(h["batch"].max()) for n in names for h in runs[n]["per_seed"] if h.height), default=0)
    for row, name in enumerate(names, start=1):
        for history, info in zip(runs[name]["per_seed"], runs[name]["infos"]):
            if history.height == 0:
                continue
            batches = (
                history.filter(pl.col("batch") > 0)
                .group_by("batch", maintain_order=True)
                .agg(pl.col("reactor").first(), pl.col("titer").max(), pl.col("spent").max(), pl.len().alias("n"))
            )
            seed = f"seed {history['seed'][0]}"
            for reactor, color in SCALE_COLORS.items():
                part = batches.filter(pl.col("reactor") == reactor)
                if part.height == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=part["batch"], y=[seed] * part.height, mode="markers", name=reactor,
                    legendgroup=reactor, showlegend=reactor not in shown,
                    marker={"symbol": "square", "size": 13, "color": color, "line": {"color": "white", "width": 2}},
                    customdata=part.select("n", "titer", "spent").rows(),
                    hovertemplate=(
                        f"{name}, {seed}<br>Batch %{{x}}: {reactor} (%{{customdata[0]}} runs)"
                        "<br>Best titer in batch %{customdata[1]:.0f}<br>Spent after $%{customdata[2]}<extra></extra>"
                    ),
                ), row=row, col=1)
                shown.add(reactor)
            if "end_reason" not in info:
                continue  # run made with an older version of this code: no end reason recorded
            last_batch = int(history["batch"].max())
            fig.add_trace(go.Scatter(
                x=[last_batch + 1], y=[seed], mode="markers+text", name="end of run",
                legendgroup="end", showlegend="end" not in shown,
                marker={
                    "symbol": "square", "size": 13, "color": END_COLOR, "line": {"color": "white", "width": 2},
                },
                text=[info["end_reason"].split(" (")[0]],  # short label; the full reason is in the hover
                textposition="middle right",
                textfont={"size": 11, "color": "#6f6e69"},
                hovertemplate=(
                    f"{name}, {seed}<br>Run ended after batch {last_batch}: {info['end_reason']}"
                    f"<br>Spent ${int(history['spent'].max())}, {history.height} experiments<extra></extra>"
                ),
            ), row=row, col=1)
            shown.add("end")
    fig.update_xaxes(title_text="Batch (lab call)", row=len(names), col=1, dtick=1)
    fig.update_xaxes(range=[-0.5, last + 8])  # room for the end-of-run labels
    fig.update_layout(
        title="Scale chosen at each step",
        template="plotly_white",
        height=180 + 40 * sum(len(runs[n]["per_seed"]) for n in names) + 60 * len(names),
        # Legend centred under the x-axis title, positioned in pixels from the bottom so it works at any height.
        legend={"title": "Scale", "orientation": "h", "yref": "container", "y": 0, "yanchor": "bottom",
                "x": 0.5, "xanchor": "center"},
        margin={"b": 100},
    )
    fig.show()
