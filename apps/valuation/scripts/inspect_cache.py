from __future__ import annotations

import argparse
import sqlite3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="/data/valuation_cache.sqlite")
    parser.add_argument("--symbol")
    parser.add_argument("--sector")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if args.symbol:
        rows = cur.execute("SELECT * FROM company_snapshot WHERE symbol = ?", (args.symbol.upper(),)).fetchall()
    else:
        rows = cur.execute("SELECT * FROM company_snapshot ORDER BY updated_at DESC LIMIT ?", (args.limit,)).fetchall()
    print("== company_snapshot ==")
    for row in rows:
        print(dict(row))

    if args.sector:
        rows = cur.execute("SELECT * FROM sector_metrics WHERE sector_index = ?", (args.sector.upper(),)).fetchall()
    else:
        rows = cur.execute("SELECT * FROM sector_metrics ORDER BY calculated_at DESC LIMIT ?", (args.limit,)).fetchall()
    print("== sector_metrics ==")
    for row in rows:
        print(dict(row))


if __name__ == "__main__":
    main()
