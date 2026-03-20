"""Interactive DuckDB explorer for project14 datasets.

Usage:
    uv run python scripts/explore_data.py            # interactive SQL shell
    uv run python scripts/explore_data.py --stats    # print stats and exit
    uv run python scripts/explore_data.py --query "SELECT * FROM sft_train LIMIT 5"
    (views: sft_train, sft_val, sft_test, dpo_train, dpo_val, sft_raw, dpo_raw, ...)
"""

import argparse
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, DatasetDict, load_from_disk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "data" / "explore.duckdb"

# HuggingFace DatasetDict views: view_prefix -> DatasetDict path.
# Each split (train/val/test) becomes a separate DuckDB view: {prefix}_{split}.
HF_DICT_DATASETS: dict[str, Path] = {
    "sft": DATA_DIR / "final" / "sft",
    "dpo": DATA_DIR / "final" / "dpo",
}

# HuggingFace Dataset (single split) views: view_name -> Dataset path.
HF_SINGLE_DATASETS: dict[str, Path] = {
    "sft_raw": DATA_DIR / "processed" / "sft_raw",
    "dpo_raw": DATA_DIR / "processed" / "dpo_raw",
    "sft_anonymized": DATA_DIR / "processed" / "sft_anonymized",
    "dpo_anonymized": DATA_DIR / "processed" / "dpo_anonymized",
}

# Raw HuggingFace Arrow IPC streams: view_name -> list of .arrow shard paths.
# Shards are concatenated into a single pyarrow Table before registration.
RAW_DATASETS: dict[str, list[Path]] = {
    "raw_frenchmedmcqa_train": [DATA_DIR / "raw/frenchmedmcqa/train/data-00000-of-00001.arrow"],
    "raw_frenchmedmcqa_val": [DATA_DIR / "raw/frenchmedmcqa/validation/data-00000-of-00001.arrow"],
    "raw_frenchmedmcqa_test": [DATA_DIR / "raw/frenchmedmcqa/test/data-00000-of-00001.arrow"],
    "raw_mediql_mcqu_train": [DATA_DIR / "raw/mediql_mcqu/train/data-00000-of-00001.arrow"],
    "raw_mediql_mcqu_val": [DATA_DIR / "raw/mediql_mcqu/validation/data-00000-of-00001.arrow"],
    "raw_mediql_mcqu_test": [DATA_DIR / "raw/mediql_mcqu/test/data-00000-of-00001.arrow"],
    "raw_mediql_oeq_test": [DATA_DIR / "raw/mediql_oeq/test/data-00000-of-00001.arrow"],
    "raw_medquad_train": [DATA_DIR / "raw/medquad/train/data-00000-of-00001.arrow"],
    "raw_ultramedical_train": [
        DATA_DIR / "raw/ultramedical_preference/train/data-00000-of-00002.arrow",
        DATA_DIR / "raw/ultramedical_preference/train/data-00001-of-00002.arrow",
    ],
    "raw_ultramedical_val": [
        DATA_DIR / "raw/ultramedical_preference/validation/data-00000-of-00001.arrow"
    ],
    "raw_ultramedical_test": [
        DATA_DIR / "raw/ultramedical_preference/test/data-00000-of-00001.arrow"
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_arrow_shards(paths: list[Path]) -> pa.Table | None:
    """Read one or more Arrow IPC stream shards and concatenate them.

    Args:
        paths: Shard file paths (must all exist).

    Returns:
        A single concatenated pyarrow Table, or None on error.
    """
    tables: list[pa.Table] = []
    for path in paths:
        with path.open("rb") as f:
            reader = ipc.open_stream(f)
            tables.append(reader.read_all())
    return pa.concat_tables(tables) if tables else None


def build_connection() -> duckdb.DuckDBPyConnection:
    """Open (or create) the persistent DuckDB file and register all views.

    HuggingFace DatasetDict and Dataset directories are loaded via load_from_disk()
    and registered as in-memory pyarrow relations.
    Raw Arrow IPC streams are loaded via pyarrow and registered similarly.

    Returns:
        An open DuckDB connection with all dataset views registered.
    """
    con = duckdb.connect(str(DB_PATH))

    registered: list[str] = []
    skipped: list[str] = []

    # --- HuggingFace DatasetDict (final splits) ---
    for prefix, path in HF_DICT_DATASETS.items():
        if not path.exists():
            skipped.append(prefix)
            continue
        dataset_dict = DatasetDict(load_from_disk(str(path)))  # type: ignore[arg-type]
        for split_name, split_ds in dataset_dict.items():
            view_name = f"{prefix}_{split_name}"
            con.register(view_name, split_ds.to_arrow())  # type: ignore[reportAttributeAccessIssue]
            registered.append(view_name)

    # --- HuggingFace Dataset (single, processed) ---
    for view_name, path in HF_SINGLE_DATASETS.items():
        if not path.exists():
            skipped.append(view_name)
            continue
        con.register(view_name, Dataset.load_from_disk(str(path)).to_arrow())  # type: ignore[reportAttributeAccessIssue]
        registered.append(view_name)

    # --- Raw Arrow IPC streams ---
    for view_name, paths in RAW_DATASETS.items():
        existing = [p for p in paths if p.exists()]
        if not existing:
            skipped.append(view_name)
            continue
        table = _load_arrow_shards(existing)
        if table is None:
            skipped.append(view_name)
            continue
        # Register the pyarrow Table so DuckDB can query it by name
        con.register(view_name, table)
        registered.append(view_name)

    print(f"\n[DuckDB] Connected  →  {DB_PATH}")
    print(f"[DuckDB] Views registered ({len(registered)}): {', '.join(registered)}")
    if skipped:
        print(f"[DuckDB] Skipped (not found): {', '.join(skipped)}")

    return con


def print_stats(con: duckdb.DuckDBPyConnection) -> None:
    """Print row counts and column schemas for every registered view.

    Args:
        con: An open DuckDB connection.
    """
    views: list[tuple[str]] = con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
    ).fetchall()

    if not views:
        print("No views found.")
        return

    print("\n" + "=" * 60)
    print("  DATASET OVERVIEW")
    print("=" * 60)

    for (name,) in sorted(views):
        try:
            _row = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()
            count = _row[0] if _row is not None else 0
            columns = con.execute(f"DESCRIBE {name}").fetchall()
            col_info = ", ".join(f"{c[0]} ({c[1]})" for c in columns)
            print(f"\n  {name}")
            print(f"    rows    : {count:,}")
            print(f"    columns : {col_info}")

            # Quick value distribution for 'source' if present
            col_names = [c[0] for c in columns]
            if "source" in col_names:
                dist = con.execute(
                    f"SELECT source, COUNT(*) AS n FROM {name} GROUP BY source ORDER BY n DESC"
                ).fetchall()
                dist_str = " | ".join(f"{s}: {n:,}" for s, n in dist)
                print(f"    sources : {dist_str}")

            if "language" in col_names:
                lang_dist = con.execute(
                    f"SELECT language, COUNT(*) AS n FROM {name} GROUP BY language ORDER BY n DESC"
                ).fetchall()
                lang_str = " | ".join(f"{lang}: {n:,}" for lang, n in lang_dist)
                print(f"    langs   : {lang_str}")

        except Exception as exc:  # noqa: BLE001
            print(f"  {name}  →  ERROR: {exc}")

    print("\n" + "=" * 60)


def run_query(con: duckdb.DuckDBPyConnection, query: str) -> None:
    """Execute a SQL query and print the result as a table.

    Args:
        con: An open DuckDB connection.
        query: A valid DuckDB SQL statement.
    """
    try:
        result = con.execute(query)
        df = result.df()
        print(df.to_string(index=False, max_colwidth=80))
        print(f"\n({len(df)} rows)")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}")


def interactive_shell(con: duckdb.DuckDBPyConnection) -> None:
    """Start an interactive REPL for ad-hoc SQL queries.

    Type 'help' for quick reference, 'exit' or Ctrl-D to quit.

    Args:
        con: An open DuckDB connection.
    """
    help_text = """
  Quick reference
  ───────────────
  SHOW TABLES;                                          list all views
  DESCRIBE sft_train;                                   column names & types

  -- Processed / final
  SELECT * FROM sft_train LIMIT 5;
  SELECT source, COUNT(*) FROM sft_train GROUP BY source;
  SELECT * FROM sft_train WHERE language = 'fr' LIMIT 10;
  SELECT instruction, response FROM sft_train WHERE urgency_level = 'HIGH' LIMIT 3;

  -- Raw (HuggingFace originals)
  SELECT * FROM raw_medquad_train LIMIT 5;
  SELECT * FROM raw_frenchmedmcqa_train LIMIT 5;
  SELECT * FROM raw_mediql_mcqu_train WHERE task = 'MedicalCauses' LIMIT 5;
  SELECT * FROM raw_ultramedical_train LIMIT 3;

  Type 'stats' to reprint the overview, 'exit' to quit.
"""
    print(help_text)

    buffer: list[str] = []

    while True:
        prompt = ">>> " if not buffer else "... "
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        lower = line.strip().lower()

        if lower in ("exit", "quit", "\\q"):
            print("Bye!")
            break
        if lower == "help":
            print(help_text)
            continue
        if lower == "stats":
            print_stats(con)
            continue

        buffer.append(line)
        full = " ".join(buffer)

        # Execute when the statement ends with ';' or is a single keyword
        if full.strip().endswith(";") or lower in ("show tables",):
            run_query(con, full)
            buffer = []
        elif full.strip() == "":
            buffer = []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate mode."""
    parser = argparse.ArgumentParser(description="DuckDB explorer for project14 datasets")
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics and exit",
    )
    parser.add_argument(
        "--query",
        metavar="SQL",
        help="Run a single SQL query and exit",
    )
    args = parser.parse_args()

    con = build_connection()

    if args.stats:
        print_stats(con)
    elif args.query:
        run_query(con, args.query)
    else:
        print_stats(con)
        interactive_shell(con)

    con.close()


if __name__ == "__main__":
    main()
