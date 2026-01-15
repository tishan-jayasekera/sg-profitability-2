from __future__ import annotations

import argparse
from pathlib import Path

from src.build import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SG profitability datasets")
    parser.add_argument("--input", required=True, help="Path to Excel input file")
    parser.add_argument("--fy", default="FY26", help="Fiscal year label")
    parser.add_argument(
        "--include-all-history",
        action="store_true",
        help="Include all revenue months instead of FY-limited window",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        input_path=Path(args.input),
        fy=args.fy,
        include_all_history=args.include_all_history,
    )


if __name__ == "__main__":
    main()
