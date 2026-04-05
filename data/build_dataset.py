"""
ChronoVeritas — Build / verify dataset.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Path config
# ---------------------------------------------------------------------------

DATA_DIR   = Path(__file__).resolve().parent
TASKS_DIR  = DATA_DIR / "tasks"
CORPUS_DIR = DATA_DIR / "corpus"

# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"

def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text

def ok(msg: str)   -> str: return _c(f"[OK]   {msg}", "32")
def fail(msg: str) -> str: return _c(f"[FAIL] {msg}", "31")
def warn(msg: str) -> str: return _c(f"[WARN] {msg}", "33")
def info(msg: str) -> str: return _c(f"[INFO] {msg}", "36")

# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: Any) -> None:
    """Write JSON atomically via a sibling temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic on POSIX; best-effort on Windows
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

REQUIRED_TASK_KEYS = frozenset(
    {"task_id", "difficulty", "max_steps", "claim", "ground_truth", "corpus"}
)
REQUIRED_GT_KEYS = frozenset(
    {"gt_verdict", "gt_mutation_type", "gt_mutation_doc_id",
     "gt_provenance_chain", "gt_timeline"}
)

VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
VALID_VERDICTS     = frozenset({"true", "false", "misleading", "unverifiable"})
VALID_MUTATIONS    = frozenset(
    {"distortion", "omission", "fabrication", "context_shift", "none"}
)


def _verify_single_task(
    task_path: Path,
    raw: Dict[str, Any],
) -> List[str]:
    """
    Run all checks on a single raw task dict.
    Returns a (possibly empty) list of error strings.
    """
    errors: List[str] = []
    task_id = raw.get("task_id", task_path.stem)

    # ── 1. Required top-level keys ────────────────────────────────────
    missing_keys = REQUIRED_TASK_KEYS - set(raw.keys())
    if missing_keys:
        errors.append(f"missing top-level keys: {sorted(missing_keys)}")
        return errors  # can't proceed without structure

    # ── 2. Required ground_truth keys ────────────────────────────────
    gt = raw.get("ground_truth", {})
    missing_gt = REQUIRED_GT_KEYS - set(gt.keys())
    if missing_gt:
        errors.append(f"ground_truth missing keys: {sorted(missing_gt)}")

    # ── 3. Pydantic model validation ──────────────────────────────────
    try:
        # Import here to avoid circular imports at module load
        from env.models import TaskSpec
        TaskSpec(**raw)
    except ValidationError as exc:
        for e in exc.errors():
            loc = " → ".join(str(x) for x in e["loc"])
            errors.append(f"schema error at '{loc}': {e['msg']}")
        # Still continue with manual cross-ref checks below

    # ── 4. Enum validation ────────────────────────────────────────────
    difficulty = raw.get("difficulty", "")
    if difficulty not in VALID_DIFFICULTIES:
        errors.append(
            f"invalid difficulty {difficulty!r}; must be one of {sorted(VALID_DIFFICULTIES)}"
        )

    if gt:
        verdict = gt.get("gt_verdict", "")
        if verdict not in VALID_VERDICTS:
            errors.append(
                f"invalid gt_verdict {verdict!r}; must be one of {sorted(VALID_VERDICTS)}"
            )
        mut_type = gt.get("gt_mutation_type", "")
        if mut_type not in VALID_MUTATIONS:
            errors.append(
                f"invalid gt_mutation_type {mut_type!r}; must be one of {sorted(VALID_MUTATIONS)}"
            )

    # ── 5. Corpus integrity ───────────────────────────────────────────
    corpus = raw.get("corpus", [])
    if not corpus:
        errors.append("corpus is empty — task has no documents")

    corpus_ids: set[str] = set()
    for i, doc in enumerate(corpus):
        doc_id = doc.get("doc_id", "")
        if not doc_id:
            errors.append(f"corpus[{i}] missing doc_id")
            continue
        if doc_id in corpus_ids:
            errors.append(f"duplicate doc_id in corpus: {doc_id!r}")
        corpus_ids.add(doc_id)

        for required_field in ("title", "source", "timestamp", "content"):
            if not doc.get(required_field):
                errors.append(f"corpus doc {doc_id!r} missing/empty field: {required_field!r}")

    # ── 6. Cross-reference: ground truth ↔ corpus ────────────────────
    if gt and corpus_ids:
        mut_doc_id: Optional[str] = gt.get("gt_mutation_doc_id")
        if mut_doc_id and mut_doc_id not in corpus_ids:
            errors.append(
                f"gt_mutation_doc_id {mut_doc_id!r} not found in corpus"
            )

        for doc_id in gt.get("gt_provenance_chain", []):
            if doc_id not in corpus_ids:
                errors.append(
                    f"gt_provenance_chain references unknown doc_id {doc_id!r}"
                )

        unknown_timeline = [
            d for d in gt.get("gt_timeline", []) if d not in corpus_ids
        ]
        if unknown_timeline:
            errors.append(
                f"gt_timeline references unknown doc_ids: {unknown_timeline}"
            )

    # ── 7. Logical consistency ────────────────────────────────────────
    if gt:
        mut_type = gt.get("gt_mutation_type", "none")
        mut_doc  = gt.get("gt_mutation_doc_id")
        if mut_type != "none" and not mut_doc:
            errors.append(
                "gt_mutation_doc_id must be set when gt_mutation_type is not 'none'"
            )
        if mut_type == "none" and mut_doc:
            errors.append(
                "gt_mutation_doc_id should be null when gt_mutation_type is 'none'"
            )

    max_steps = raw.get("max_steps", 0)
    if not isinstance(max_steps, int) or max_steps < 1:
        errors.append(f"max_steps must be a positive integer, got {max_steps!r}")

    return errors


def verify_tasks(*, verbose: bool = True) -> Dict[str, Any]:
    """
    Verify all task files under TASKS_DIR.

    Returns
    -------
    {
        "passed":  [task_id, ...],
        "failed":  {task_id: [error_str, ...]},
        "warned":  {task_id: [warning_str, ...]},
        "total":   int,
    }
    """
    task_paths = sorted(TASKS_DIR.glob("*.json"))
    if not task_paths:
        if verbose:
            print(warn(f"No task files found in {TASKS_DIR}"))
        return {"passed": [], "failed": {}, "warned": {}, "total": 0}

    passed: List[str] = []
    failed: Dict[str, List[str]] = {}
    warned: Dict[str, List[str]] = {}

    for task_path in task_paths:
        try:
            with open(task_path) as f:
                raw = json.load(f)
        except json.JSONDecodeError as exc:
            task_id = task_path.stem
            failed[task_id] = [f"invalid JSON: {exc}"]
            if verbose:
                print(fail(f"{task_path.name}: invalid JSON — {exc}"))
            continue

        task_id = raw.get("task_id", task_path.stem)
        errors = _verify_single_task(task_path, raw)

        # Separate hard errors from warnings (currently all are errors)
        if errors:
            failed[task_id] = errors
            if verbose:
                print(fail(f"{task_path.name} [{task_id}]:"))
                for err in errors:
                    print(f"       • {err}")
        else:
            passed.append(task_id)
            n_docs = len(raw.get("corpus", []))
            if verbose:
                print(
                    ok(
                        f"{task_path.name}: task_id={task_id!r}, "
                        f"difficulty={raw.get('difficulty')}, "
                        f"max_steps={raw.get('max_steps')}, "
                        f"{n_docs} docs"
                    )
                )

    summary = {
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "total":  len(task_paths),
    }

    if verbose:
        print(
            f"\n  {len(passed)}/{len(task_paths)} tasks passed, "
            f"{len(failed)} failed."
        )

    return summary


# ---------------------------------------------------------------------------
# Corpus file extraction
# ---------------------------------------------------------------------------

def build_corpus_files(*, skip_existing: bool = False, verbose: bool = True) -> int:
    """
    Extract corpus documents from task files and write them as individual
    JSON files in data/corpus/<doc_id>.json.

    Parameters
    ----------
    skip_existing:
        If True, files that already exist on disk are not overwritten.
    verbose:
        Print a line per file written.

    Returns
    -------
    Number of files written.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0

    for task_path in sorted(TASKS_DIR.glob("*.json")):
        try:
            with open(task_path) as f:
                task = json.load(f)
        except json.JSONDecodeError as exc:
            print(warn(f"Skipping {task_path.name}: invalid JSON — {exc}"))
            continue

        for doc in task.get("corpus", []):
            doc_id = doc.get("doc_id")
            if not doc_id:
                print(warn(f"  Skipping doc with missing doc_id in {task_path.name}"))
                continue

            doc_path = CORPUS_DIR / f"{doc_id}.json"

            if skip_existing and doc_path.exists():
                if verbose:
                    print(info(f"  Skipped (exists): {doc_path.name}"))
                continue

            _atomic_write(doc_path, doc)
            written += 1
            if verbose:
                print(ok(f"  Wrote {doc_path.name}"))

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ChronoVeritas dataset builder and verifier"
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification; do not write corpus files.",
    )
    p.add_argument(
        "--build-only",
        action="store_true",
        help="Only build corpus files; skip verification.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not overwrite corpus files that already exist on disk.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; only print summary.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit with code 1 if any task fails verification.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    verbose = not args.quiet
    exit_code = 0

    if not args.build_only:
        print(info("=== Verifying tasks ==="))
        result = verify_tasks(verbose=verbose)
        if result["failed"] and args.fail_fast:
            print(fail("Verification failed — exiting."))
            return 1
        if result["failed"]:
            exit_code = 1

    if not args.verify_only:
        print(info("\n=== Building corpus files ==="))
        n = build_corpus_files(skip_existing=args.skip_existing, verbose=verbose)
        if not args.quiet:
            print(info(f"\n  {n} corpus file(s) written to {CORPUS_DIR}"))

    if not args.quiet:
        print(info("\nDone."))

    return exit_code


if __name__ == "__main__":
    sys.exit(main())