"""
openrouter_runner.py

CLI batch runner for generation via OpenRouter - same Excel schema and
same scoring/resume logic as batch_runner.py, kept as separate
orchestration because OpenRouter doesn't have the Ollama-specific
concepts (loading into VRAM, unload, keep_alive).

Writes to the same T_0.0/T_0.3/T_0.7 sheets as the local models, not to a
separate sheet: since temperature is controlled here (it is), these
models share the same experimental design as the local ones - they are
not "commercial" in the sense reserved for the 10 with uncontrollable
parameters - see the session document on this distinction.

Usage:
    python openrouter_runner.py
    python openrouter_runner.py --models dots-studio/dots3-note-preview:free --limit-prompts 3
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import config
from excel_writer import ResultsWriter
from openrouter_client import OpenRouterClient
from response_parser import extract_predicted_word
from scoring import is_clue_echo, score

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
SESSION_START = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.DATA_DIR / "openrouter_runner.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_json_path: Path) -> list[dict]:
    try:
        with open(prompts_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(
            f"{prompts_json_path} not found. Generate the JSON first with excel_to_json.py"
        )
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {prompts_json_path}: {e}")
        sys.exit(1)
    return data["prompts"]


def resolve_requested_models(client: OpenRouterClient, requested_models: list[str]) -> list[str]:
    """Cross-checks the requested models against OpenRouter's real catalog.

    A wrong ID is flagged and skipped instead of failing the whole
    batch - same logic as resolve_available_models in batch_runner.py,
    adapted to the remote catalog instead of 'ollama list'.
    """
    available = client.get_available_models()
    if not available:
        logger.warning(
            "OpenRouter catalog not retrieved (network error or invalid key) - "
            "proceeding anyway, a wrong ID will fail explicitly on the first call."
        )
        return requested_models

    resolved = []
    for model in requested_models:
        if model in available:
            resolved.append(model)
        else:
            logger.warning(
                f"Requested model not found in the current OpenRouter catalog "
                f"(wrong ID, or model removed?): {model}"
            )
    return resolved


def run_model(
        client: OpenRouterClient,
        writer: ResultsWriter,
        model_name: str,
        prompts: list[dict],
        temperatures: list[float],
) -> None:
    for temperature in temperatures:
        sheet_name = config.SHEET_BY_TEMPERATURE[temperature]
        completed = writer.completed_pairs(sheet_name)

        for prompt in prompts:
            prompt_id = prompt["prompt_id"]

            if (model_name, prompt_id) in completed:
                logger.info(f"[skip] {model_name} | T={temperature} | {prompt_id} already present")
                continue

            prompt_text = config.PROMPT_TEMPLATE.format(
                word1=prompt["clues"][0],
                word2=prompt["clues"][1],
                word3=prompt["clues"][2],
                word4=prompt["clues"][3],
                word5=prompt["clues"][4],
            )

            result = client.generate(
                model=model_name,
                prompt=prompt_text,
                temperature=temperature,
                top_p=config.TOP_P,
                top_k=config.TOP_K,
                prompt_id=prompt_id,
            )

            if not result.success:
                logger.error(f"[fail] {model_name} | T={temperature} | {prompt_id}: {result.error}")
                row = {
                    "model_name": model_name,
                    "prompt_id": prompt_id,
                    "word1": prompt["clues"][0],
                    "word2": prompt["clues"][1],
                    "word3": prompt["clues"][2],
                    "word4": prompt["clues"][3],
                    "word5": prompt["clues"][4],
                    "expected_solution": prompt["solution"],
                    "raw_response": f"[{(result.error or 'ERROR').upper()}]",
                    "predicted_word": "",
                    "format_compliant": 0,
                    "exact_match": 0,
                    "same_root": 0,
                    "levenshtein_normalized": 1.0,
                    "is_similar_lev": 0,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "difficulty_note": prompt["difficulty"],
                    "duration_s": round(result.duration_s, 1),
                    "is_first_call": 0,
                    "status": "timeout" if result.error == "timeout" else "error",
                    "is_clue_echo": 0,
                }
                writer.append_row(sheet_name, row)
                continue

            parsed = extract_predicted_word(result.raw_text)
            metrics = score(
                parsed.predicted_word, prompt["solution"], config.LEVENSHTEIN_SIMILARITY_THRESHOLD
            )
            clue_echo = is_clue_echo(parsed.predicted_word, prompt["clues"])

            row = {
                "model_name": model_name,
                "prompt_id": prompt_id,
                "word1": prompt["clues"][0],
                "word2": prompt["clues"][1],
                "word3": prompt["clues"][2],
                "word4": prompt["clues"][3],
                "word5": prompt["clues"][4],
                "expected_solution": prompt["solution"],
                "raw_response": result.raw_text,
                "predicted_word": parsed.predicted_word,
                "format_compliant": int(parsed.format_compliant),
                "exact_match": metrics.exact_match,
                "same_root": metrics.same_root,
                "levenshtein_normalized": metrics.levenshtein_normalized,
                "is_similar_lev": metrics.is_similar_lev,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "difficulty_note": prompt["difficulty"],
                "duration_s": round(result.duration_s, 1),
                "is_first_call": 0,
                "status": "ok",
                "is_clue_echo": clue_echo,
            }
            writer.append_row(sheet_name, row)

            logger.info(
                f"[ok] {model_name} | T={temperature} | {prompt_id} -> "
                f"'{parsed.predicted_word}' (expected '{prompt['solution']}', "
                f"exact={metrics.exact_match}, compliant={parsed.format_compliant}, "
                f"{result.duration_s:.1f}s"
                + (f", {result.completion_tokens} completion_tokens" if result.completion_tokens else "")
                + ")"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guillotine batch runner - models via OpenRouter")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of OpenRouter model IDs (default: config.OPENROUTER_MODELS)",
    )
    parser.add_argument(
        "--limit-prompts", type=int, default=None,
        help="Run only the first N prompts (smoke test on a new model)",
    )
    args = parser.parse_args()

    try:
        client = OpenRouterClient(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            timeout_s=config.OPENROUTER_TIMEOUT_S,
            site_url=config.OPENROUTER_SITE_URL,
            site_name=config.OPENROUTER_SITE_NAME,
            jsonl_backup_path=config.JSON_BACKUP_DIR / f"openrouter_{SESSION_START}.jsonl",
            min_interval_s=config.OPENROUTER_MIN_INTERVAL_S,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if not client.check_service():
        logger.error("OpenRouter unreachable or invalid key. Check OPENROUTER_API_KEY in .env.")
        sys.exit(1)

    requested_models = (
        [m.strip() for m in args.models.split(",")] if args.models else config.OPENROUTER_MODELS
    )
    if not requested_models:
        logger.error("No model specified (neither --models nor config.OPENROUTER_MODELS).")
        sys.exit(1)

    models = resolve_requested_models(client, requested_models)
    if not models:
        logger.error("None of the requested models turned out to be available on OpenRouter.")
        sys.exit(1)

    logger.info(f"Models to run ({len(models)}/{len(requested_models)} requested): {models}")

    prompts = load_prompts(config.PROMPTS_JSON)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
        logger.info(f"Smoke test: limited to the first {len(prompts)} prompts")

    writer = ResultsWriter(config.RESULTS_XLSX, config.RESULT_COLUMNS)

    for idx, model_name in enumerate(models, start=1):
        logger.info(f"=== [{idx}/{len(models)}] Model: {model_name} ===")
        try:
            run_model(client, writer, model_name, prompts, config.TEMPERATURES)
        except Exception as e:
            logger.error(f"Unhandled error on {model_name}, moving to the next model: {e}")
            continue

    logger.info("Batch complete.")


if __name__ == "__main__":
    main()
