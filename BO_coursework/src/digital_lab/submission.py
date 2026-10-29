"""Load and statically validate algorithm files (my_algorithm.py and the references)."""

import ast
import importlib.util
import re
from pathlib import Path
from typing import Callable

from digital_lab.constants import ALLOWED_IMPORTS

SUBMISSION_FUNCTION = "my_algorithm"
NAMES_VARIABLE = "STUDENT_NAMES"


def _function_name(path: Path) -> str:
    """Name of the algorithm function in `path`: `my_algorithm` for submissions, else the file stem.

    Parameters
    ----------
    path : pathlib.Path
        Algorithm file.

    Returns
    -------
    str
        Function name to load.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    return SUBMISSION_FUNCTION if SUBMISSION_FUNCTION in names else path.stem


def read_student_names(path: str | Path) -> list[str]:
    """Read `STUDENT_NAMES` from a submission without executing it.

    Parameters
    ----------
    path : str or pathlib.Path
        Submission file.

    Returns
    -------
    list[str]
        The names, or an empty list if the variable is missing or not a list of strings.
    """
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == NAMES_VARIABLE for t in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return []
            if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
                return [v.strip() for v in value if v.strip()]
    return []


def validate_submission(path: str | Path) -> list[str]:
    """Check a submission's imports, function and `STUDENT_NAMES`.

    Parameters
    ----------
    path : str or pathlib.Path
        Submission file.

    Returns
    -------
    list[str]
        Problems found; empty if the submission is valid.
    """
    path = Path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    problems = []
    disallowed = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            disallowed |= {a.name.split(".")[0] for a in node.names} - ALLOWED_IMPORTS
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module and node.module.split(".")[0] not in ALLOWED_IMPORTS):
                disallowed.add(node.module or ".")
    private = sorted({
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__")
    })
    if private:
        problems.append(f"Uses private attributes {private}: only use the lab's public interface.")
    if disallowed:
        problems.append(f"Disallowed imports: {sorted(disallowed)}. Only {sorted(ALLOWED_IMPORTS)} are allowed.")
    if not any(isinstance(n, ast.FunctionDef) and n.name == SUBMISSION_FUNCTION for n in tree.body):
        problems.append(f"No top-level function named '{SUBMISSION_FUNCTION}'.")
    names = read_student_names(path)
    if not names or any("TODO" in n for n in names):
        problems.append(f"Set {NAMES_VARIABLE} to a list with your full names, e.g. ['Sam Stricker 006547054', 'Antonio del Rio Chanona 001234567'].")
    return problems


# Things an honest algorithm has no reason to use; flagged for the GTA to review.
SUSPICIOUS_CALLS = {
    "open": "reads or writes files",
    "exec": "runs code from a string",
    "eval": "runs code from a string",
    "compile": "builds code from a string",
    "__import__": "imports modules dynamically",
    "getattr": "looks up attributes by name (can reach hidden lab internals)",
    "setattr": "changes attributes by name",
    "delattr": "deletes attributes by name",
    "vars": "inspects an object's internal data",
    "globals": "inspects module internals",
    "locals": "inspects local internals",
    "dir": "lists an object's internals",
    "breakpoint": "starts the debugger",
    "input": "waits for user input",
}
FILE_LOADERS = {"load", "loadtxt", "genfromtxt", "fromfile", "loadmat", "loadarff", "memmap"}
SUSPICIOUS_STRINGS = ("lab_config", ".json", "dbiolab", "digital_lab", "site-packages", "noise_parameters")
CLONE_NAME = re.compile(r"celltype_\d+")


def audit_submission(path: str | Path) -> list[str]:
    """Flag code that looks like it tries to reach data an algorithm shouldn't have.

    These are warnings for the GTA to review, not automatic rejections:
    file access, dynamic code or attribute tricks, dunder internals (e.g. `__dict__`,
    `__closure__`), strings pointing at the lab's configuration or the simulator, and
    hard-coded clone names (a sign of tuning to the practice lab).

    Parameters
    ----------
    path : str or pathlib.Path
        Submission file.

    Returns
    -------
    list[str]
        One message per kind of finding, with line numbers; empty if nothing was found.
    """
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    found: dict[str, set[int]] = {}

    def flag(message: str, node: ast.AST) -> None:
        found.setdefault(message, set()).add(getattr(node, "lineno", 0))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SUSPICIOUS_CALLS:
            flag(f"calls {node.func.id}() — {SUSPICIOUS_CALLS[node.func.id]}", node)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in FILE_LOADERS:
            flag(f"calls .{node.func.attr}() — loads data from a file", node)
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            flag(f"uses {node.attr} — Python internals", node)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            hits = [h for h in SUSPICIOUS_STRINGS if h in node.value]
            if hits:
                flag(f"string mentions {hits[0]!r} — points at lab configuration or simulator files", node)
            if CLONE_NAME.search(node.value):
                flag("hard-coded clone name(s) like 'celltype_21' — the grading lab has different clones", node)
    return [f"{message} (line {', '.join(map(str, sorted(lines)))})" for message, lines in sorted(found.items())]


def run_name(names: list[str]) -> str:
    """Build the MLflow run name from the student names: 'Jane Doe', 'John Smith' -> 'JaneDoe_JohnSmith'.

    Parameters
    ----------
    names : list[str]
        Student names.

    Returns
    -------
    str
        The names, spaces removed, joined by underscores.
    """
    return "_".join("".join(name.split()) for name in names)


def load_algorithm(path: str | Path) -> Callable:
    """Import an algorithm file fresh from disk and return its algorithm function.

    Always re-imports, so edits written from the notebook are picked up without
    restarting the kernel.

    Parameters
    ----------
    path : str or pathlib.Path
        Algorithm file.

    Returns
    -------
    Callable
        `my_algorithm` if the file defines it, else the function named like the file.

    Raises
    ------
    ImportError
        If the file cannot be imported or lacks the function.
    """
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"_algorithm_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load algorithm from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function_name = _function_name(path)
    if not hasattr(module, function_name):
        raise ImportError(f"{path.name} defines no function '{function_name}'.")
    return getattr(module, function_name)
