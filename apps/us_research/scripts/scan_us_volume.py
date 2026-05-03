from __future__ import annotations

import argparse

from us_research.scanners import run_us_volume_scan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan US symbols for high volume ratio.")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--min-ratio", type=float, default=1.5)
    parser.add_argument("--start-date", type=str, default="2024-01-01")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = run_us_volume_scan(
        symbols=args.symbols,
        lookback=args.lookback,
        min_ratio=args.min_ratio,
        start_date=args.start_date,
    )
    if df.empty:
        print("No symbols above threshold.")
        return
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

