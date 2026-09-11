"""
excel_to_json.py

Converts guillotine.xlsx (Luca's hand-validated data source) into a
structured JSON, used as input by generate_template.py and the batch runner.

The Excel file remains the single source of truth. Every time a clue is
changed or an item is added, rerun this script to regenerate the JSON.
Never edit the JSON by hand: it would be overwritten on the next run and
would go out of sync with the Excel.

Usage:
    python source/excel_to_json.py source/guillotine.xlsx source/guillotine_prompts.json
"""
import sys
import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SHEET_NAME = "guillotine"
REQUIRED_COLUMNS = [
    "prompt_id", "category", "word1", "word2", "word3", "word4", "word5",
    "solution", "difficulty_note", "language", "meaning",
]


def load_prompts(excel_path: Path) -> list[dict]:
    """Reads the guillotine sheet and validates the minimum structure."""
    try:
        df = pd.read_excel(excel_path, sheet_name=SHEET_NAME, engine="openpyxl")
    except FileNotFoundError:
        logger.error(f"File not found: {excel_path}")
        raise
    except ValueError as e:
        logger.error(f"Sheet '{SHEET_NAME}' not found in {excel_path}: {e}")
        raise

    # The header in the real file has stray whitespace (e.g. 'prompt_id ',
    # ' word2 ') - normalize it right away, otherwise the column check
    # below fails on an otherwise valid file.
    stray_whitespace = [c for c in df.columns if c != c.strip()]
    if stray_whitespace:
        logger.warning(f"Normalized headers with stray whitespace: {stray_whitespace}")
    df.columns = df.columns.str.strip()

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in file: {missing_columns}")

    incomplete_mask = df[REQUIRED_COLUMNS].isnull().any(axis=1)
    if incomplete_mask.any():
        bad_ids = df.loc[incomplete_mask, "prompt_id"].tolist()
        logger.warning(f"Rows with missing fields, skipped: {bad_ids}")
        df = df.loc[~incomplete_mask]

    prompts = []
    seen_ids = set()

    for _, row in df.iterrows():
        prompt_id = str(row["prompt_id"]).strip()

        if prompt_id in seen_ids:
            logger.warning(f"Duplicate prompt_id ignored: {prompt_id}")
            continue
        seen_ids.add(prompt_id)

        clues = [str(row[f"word{i}"]).strip() for i in range(1, 6)]
        solution = str(row["solution"]).strip().lower()

        # Sanity check: the solution must not appear as a substring in any
        # clue, otherwise the task becomes trivial (see ex-guillo_08).
        leaking_clues = [c for c in clues if solution in c.lower()]
        if leaking_clues:
            logger.warning(
                f"{prompt_id}: solution '{solution}' appears as a substring "
                f"in clues {leaking_clues} - possible leak, check manually"
            )

        prompts.append({
            "prompt_id": prompt_id,
            "clues": clues,
            "solution": solution,
            "difficulty": str(row["difficulty_note"]).strip().lower(),
            "language": str(row["language"]).strip().lower(),
            "meaning": str(row["meaning"]).strip(),
        })

    logger.info(f"Loaded {len(prompts)} prompts from {excel_path}")
    return prompts


def main():
    if len(sys.argv) != 3:
        print("Usage: python excel_to_json.py <input.xlsx> <output.json>")
        sys.exit(1)

    excel_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])

    prompts = load_prompts(excel_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts}, f, indent=2, ensure_ascii=False)

    logger.info(f"JSON written to {json_path} ({len(prompts)} prompts)")


if __name__ == "__main__":
    main()
