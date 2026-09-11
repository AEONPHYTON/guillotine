"""
rescore.py

Recomputes the 4 automatic metrics (exact_match, same_root,
levenshtein_normalized, is_similar_lev) from the current value of
predicted_word. Useful after a manual correction of predicted_word - e.g.
answers with extended reasoning where automatic extraction failed and the
word was identified by hand from raw_response.

Does not touch raw_response or any other column: only the 4 scoring ones.

Usage:
    python rescore.py D:\\guillotine\\guillotine\\source\\ghigliottina_results.xlsx
    python rescore.py ghigliottina_results.xlsx --sheets T_0.0,T_0.3
"""
import argparse
import logging
import sys
from pathlib import Path

from openpyxl import load_workbook

import config
from scoring import is_clue_echo, score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SHEETS = ["T_0.0", "T_0.3", "T_0.7"]
REQUIRED_COLUMNS = [
    "predicted_word", "expected_solution", "exact_match", "same_root",
    "levenshtein_normalized", "is_similar_lev", "word1", "word2", "word3",
    "word4", "word5",
]


def rescore_sheet(ws) -> int:
    headers = [c.value for c in ws[1]]
    idx = {name: i + 1 for i, name in enumerate(headers)}

    missing = [c for c in REQUIRED_COLUMNS if c not in idx]
    if missing:
        raise ValueError(f"Missing columns in sheet '{ws.title}': {missing}")

    # is_clue_echo is new: if the sheet predates it, append it at the end
    # instead of requiring it like the others.
    if "is_clue_echo" not in idx:
        ws.cell(row=1, column=len(headers) + 1, value="is_clue_echo")
        headers.append("is_clue_echo")
        idx["is_clue_echo"] = len(headers)

    updated = 0
    for row_idx in range(2, ws.max_row + 1):
        predicted = ws.cell(row=row_idx, column=idx["predicted_word"]).value
        expected = ws.cell(row=row_idx, column=idx["expected_solution"]).value
        if predicted is None or expected is None or str(predicted).strip() == "":
            continue

        result = score(str(predicted), str(expected), config.LEVENSHTEIN_SIMILARITY_THRESHOLD)
        clues = [
            ws.cell(row=row_idx, column=idx[f"word{i}"]).value or "" for i in range(1, 6)
        ]
        echo = is_clue_echo(str(predicted), clues)

        ws.cell(row=row_idx, column=idx["exact_match"], value=result.exact_match)
        ws.cell(row=row_idx, column=idx["same_root"], value=result.same_root)
        ws.cell(row=row_idx, column=idx["levenshtein_normalized"], value=result.levenshtein_normalized)
        ws.cell(row=row_idx, column=idx["is_similar_lev"], value=result.is_similar_lev)
        ws.cell(row=row_idx, column=idx["is_clue_echo"], value=echo)
        updated += 1

    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recomputes the automatic metrics from predicted_word (after a manual correction)"
    )
    parser.add_argument("xlsx_path", type=Path)
    parser.add_argument(
        "--sheets", type=str, default=None,
        help="Sheets to process, comma-separated (default: T_0.0,T_0.3,T_0.7)",
    )
    args = parser.parse_args()

    if not args.xlsx_path.exists():
        logger.error(f"File not found: {args.xlsx_path}")
        sys.exit(1)

    sheets = args.sheets.split(",") if args.sheets else DEFAULT_SHEETS

    wb = load_workbook(args.xlsx_path)
    total_updated = 0
    for sheet_name in sheets:
        if sheet_name not in wb.sheetnames:
            logger.warning(f"Sheet not found, skipped: {sheet_name}")
            continue
        n = rescore_sheet(wb[sheet_name])
        logger.info(f"{sheet_name}: {n} rows recomputed")
        total_updated += n

    wb.save(args.xlsx_path)
    logger.info(f"Saved. Total rows recomputed: {total_updated}")


if __name__ == "__main__":
    main()
