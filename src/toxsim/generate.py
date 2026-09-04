"""Command-line generation of JSON and JSONL synthetic-patient datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .simulator import ModelDataError, create_patients, load_model_data


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1, help="Number of patients to generate.")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("."),
        help="Directory to receive patients.json and/or patients.jsonl.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl", "both"),
        default="both",
        help="Output format (default: both).",
    )
    parser.add_argument("--seed", type=int, help="Seed for deterministic output.")
    parser.add_argument(
        "--model-data-dir",
        type=Path,
        help="Directory containing predictive_variables.yml and score YAML files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate patient files and return a shell-compatible exit code."""
    args = build_parser().parse_args(argv)
    if args.count < 0:
        raise SystemExit("--count must be non-negative.")
    try:
        model_data = load_model_data(args.model_data_dir)
    except ModelDataError as error:
        raise SystemExit(f"toxsim-generate: {error}") from error

    patients = create_patients(model_data, args.count, seed=args.seed)
    args.destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if args.format in {"json", "both"}:
        path = args.destination / "patients.json"
        path.write_text(json.dumps(patients, indent=2) + "\n", encoding="utf-8")
        outputs.append(path)
    if args.format in {"jsonl", "both"}:
        path = args.destination / "patients.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for patient in patients:
                handle.write(json.dumps(patient) + "\n")
        outputs.append(path)
    print("Generated", args.count, "patients:", ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
