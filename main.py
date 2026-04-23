# CLI entrypoint

from __future__ import annotations

import argparse
import os
import sys

import database
import scraper
import market_data
import correlation
import reporting
import dashboard
from database import get_cross_reference_tickers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="substack-scraper",
        description="Scrape Substack posts, pull market data, and report ticker correlations.",
    )
    parser.add_argument(
        "--substacks",
        required=True,
        help=(
            "Comma-separated list of Substack slugs or custom-domain URLs to scrape. "
            "e.g. doomberg,https://newsletter.semianalysis.com,tacticzhazel"
        ),
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=90,
        help="Number of days to look back for posts (default: 90).",
    )
    parser.add_argument(
        "--db",
        default="./substack.db",
        help="Path to the SQLite database file (default: ./substack.db).",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Directory for chart HTML and CSV output files (default: ./output).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Market-data window in days around each post (default: 5).",
    )
    return parser.parse_args(argv)


def _header(text: str) -> None:
    """Print a clearly visible step header."""
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  {text}")
    print(separator)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    slugs = [s.strip() for s in args.substacks.split(",") if s.strip()]
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Initialise database
    # ------------------------------------------------------------------
    _header("Step 1 — Initialising database")
    conn = database.init_db(args.db)
    print(f"Database ready: {args.db}")

    # ------------------------------------------------------------------
    # Step 2: Scrape Substacks
    # ------------------------------------------------------------------
    _header("Step 2 — Scraping Substacks")
    print(f"Slugs    : {', '.join(slugs)}")
    print(f"Lookback : {args.lookback} days")
    total_posts = scraper.scrape_substacks(slugs, args.lookback, conn)
    print(f"Total posts scraped: {total_posts}")

    # ------------------------------------------------------------------
    # Step 3: Pull market data
    # ------------------------------------------------------------------
    _header("Step 3 — Pulling market data")
    print(f"Window: {args.window} days around each post")
    total_snapshots = market_data.pull_market_data(conn, window=args.window)
    print(f"Total market-data snapshots stored: {total_snapshots}")

    # ------------------------------------------------------------------
    # Step 4: Compute correlations
    # ------------------------------------------------------------------
    _header("Step 4 — Computing correlations")
    corr_df = correlation.compute_correlations(conn)
    print(f"Tickers analysed: {len(corr_df)}")

    if not corr_df.empty:
        top5 = corr_df.head(5)
        print("\nTop 5 tickers by Pearson r:")
        print(f"  {'Ticker':<10} {'Pearson r':>10} {'p-value':>10} {'Mentions':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for _, row in top5.iterrows():
            print(
                f"  {row['ticker']:<10} "
                f"{row['pearson_r']:>10.4f} "
                f"{row['pearson_p']:>10.4f} "
                f"{int(row['mention_count']):>10}"
            )
    else:
        print("No correlation data available.")

    written_files: list[str] = []

    # ------------------------------------------------------------------
    # Step 5: Cross-reference — tickers mentioned by multiple substacks
    # ------------------------------------------------------------------
    _header("Step 5 — Cross-referencing tickers across Substacks")
    xref_rows = get_cross_reference_tickers(conn, min_slugs=2)
    print(f"Tickers mentioned by 2+ substacks: {len(xref_rows)}")

    if xref_rows:
        print(f"\n  {'Ticker':<10} {'Substacks':>9} {'Mentions':>9}  Publications")
        print(f"  {'-'*10} {'-'*9} {'-'*9}  {'-'*30}")
        for row in xref_rows[:20]:
            print(
                f"  {row['ticker']:<10} "
                f"{row['slug_count']:>9} "
                f"{row['total_mentions']:>9}  "
                f"{row['slugs']}"
            )

    xref_path = reporting.export_cross_reference_csv(xref_rows, output_dir)
    print(f"\nSaved: {xref_path}")
    written_files.append(xref_path)

    # ------------------------------------------------------------------
    # Step 6: Generate per-ticker time-series charts
    # ------------------------------------------------------------------
    _header("Step 6 — Generating time-series charts")

    if not corr_df.empty:
        eligible = corr_df[corr_df["mention_count"] >= 3]["ticker"].tolist()
        print(f"Tickers with >= 3 mentions: {len(eligible)}")
        for ticker in eligible:
            path = reporting.generate_time_series_chart(conn, ticker, output_dir)
            if path:
                print(f"  Saved: {path}")
                written_files.append(path)
            else:
                print(f"  Skipped {ticker} (insufficient data for chart)")
    else:
        print("No eligible tickers — skipping time-series charts.")

    # ------------------------------------------------------------------
    # Step 7: Generate scatter chart
    # ------------------------------------------------------------------
    _header("Step 7 — Generating scatter chart")
    scatter_path = reporting.generate_scatter_chart(corr_df, output_dir)
    if scatter_path:
        print(f"Saved: {scatter_path}")
        written_files.append(scatter_path)
    else:
        print("Scatter chart skipped (insufficient data after filtering).")

    # ------------------------------------------------------------------
    # Step 8: Export CSV
    # ------------------------------------------------------------------
    _header("Step 8 — Exporting CSV")
    csv_path = reporting.export_csv(corr_df, output_dir)
    print(f"Saved: {csv_path}")
    written_files.append(csv_path)

    # ------------------------------------------------------------------
    # Step 9: Generate dashboard
    # ------------------------------------------------------------------
    _header("Step 9 — Generating dashboard")
    dash_path = dashboard.generate_dashboard(conn, corr_df, xref_rows, output_dir)
    print(f"Saved: {dash_path}")
    written_files.append(dash_path)

    # ------------------------------------------------------------------
    # Step 10: Summary
    # ------------------------------------------------------------------
    _header("Step 10 — Summary")
    if written_files:
        print(f"Files written ({len(written_files)} total):")
        for f in written_files:
            print(f"  {f}")
    else:
        print("No output files were written.")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main(sys.argv[1:])
