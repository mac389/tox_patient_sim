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
    destination_group = parser.add_mutually_exclusive_group()
    destination_group.add_argument(
        "--destination",
        type=Path,
        help="Directory to receive patients.json and/or patients.jsonl.",
    )
    destination_group.add_argument(
        "--output",
        type=Path,
        help="Output file path ending in .json or .jsonl; its suffix selects the format.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl", "both"),
        help="Output format for --destination (default: both).",
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
    if args.output is not None:
        output_format = _format_from_output_path(args.output)
        if args.format is not None and args.format != output_format:
            raise SystemExit(
                f"--format {args.format!r} conflicts with --output suffix {args.output.suffix!r}."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_patients(args.output, patients, output_format)
        outputs = [args.output]
    else:
        destination = args.destination or Path(".")
        output_format = args.format or "both"
        destination.mkdir(parents=True, exist_ok=True)
        outputs = _write_destination(destination, patients, output_format)
    print("Generated", args.count, "patients:", ", ".join(str(path) for path in outputs))
    return 0


def _format_from_output_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    raise SystemExit("--output must end in .json or .jsonl.")


def _write_destination(destination: Path, patients: list[dict], output_format: str) -> list[Path]:
    outputs: list[Path] = []
    if output_format in {"json", "both"}:
        path = destination / "patients.json"
        path.write_text(json.dumps(patients, indent=2) + "\n", encoding="utf-8")
        outputs.append(path)
    if output_format in {"jsonl", "both"}:
        path = destination / "patients.jsonl"
        _write_patients(path, patients, "jsonl")
        outputs.append(path)
    return outputs


def _write_patients(path: Path, patients: list[dict], output_format: str) -> None:
    if output_format == "json":
        path.write_text(json.dumps(patients, indent=2) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        for patient in patients:
            handle.write(json.dumps(patient) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
