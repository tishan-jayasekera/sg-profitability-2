from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed parquet to CSV")
    parser.add_argument("--input", required=True, help="Path to parquet file")
    parser.add_argument("--output", required=True, help="Path to CSV output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(Path(args.input))
    df.to_csv(Path(args.output), index=False)


if __name__ == "__main__":
    main()
