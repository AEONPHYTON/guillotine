"""
commercial_runner.py

Batch runner for the "true" commercial models (high cost per call: Claude
Opus, GPT-5.x, Grok, Kimi K3...) - a single run per model/prompt, without
setting temperature/top_p/top_k: the provider uses its own default value,
same principle as an answer collected by hand via copy-paste in chat. RQ4
(sensitivity to temperature) doesn't apply to this data, by construction -
the reason is cost, not a technical impossibility of controlling the
parameters (it could be done with OpenRouter, here it's a deliberate
choice not to).

Writes to the "Commercial" sheet, not to T_0.0/T_0.3/T_0.7 - a sheet kept
separate both from the controlled-temperature ones and from any earlier
structure meant for manual collection.

Usage:
    python commercial_runner.py
    python commercial_runner.py --models anthropic/claude-opus-5 --limit-prompts 3
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
        logging.FileHandler(config.DATA_DIR / "commercial_runner.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_json_path: Path) -> list[dict]:
    try:
        with open(prompts_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"{prompts_json_path} not found. Generate the JSON first with excel_to_json.py")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {prompts_json_path}: {e}")
        sys.exit(1)
    return data["prompts"]


def resolve_requested_models(client: OpenRouterClient, requested_models: list[str]) -> list[str]:
    available = client.get_available_models()
    if not available:
        logger.warning(
            "OpenRouter catalog not retrieved - proceeding anyway, "
            "a wrong ID will fail explicitly on the first call."
        )
        return requested_models

    resolved = []
    for model in requested_models:
        if model in available:
            resolved.append(model)
        else:
            logger.warning(f"Requested model not found in the OpenRouter catalog: {model}")
    return resolved


def run_model(
    client: OpenRouterClient,
    writer: ResultsWriter,
    model_name: str,
    prompts: list[dict],
) -> None:
    completed = writer.completed_pairs(config.COMMERCIAL_SHEET_NAME)

    for prompt in prompts:
        prompt_id = prompt["prompt_id"]

        if (model_name, prompt_id) in completed:
            logger.info(f"[skip] {model_name} | {prompt_id} already present")
            continue

        prompt_text = config.PROMPT_TEMPLATE.format(
            word1=prompt["clues"][0],
            word2=prompt["clues"][1],
            word3=prompt["clues"][2],
            word4=prompt["clues"][3],
            word5=prompt["clues"][4],
        )

        # No temperature/top_p/top_k passed: the client doesn't include
        # them in the request, the provider uses its own default.
        result = client.generate(model=model_name, prompt=prompt_text, prompt_id=prompt_id)

        base_row = {
            "model_name": model_name,
            "prompt_id": prompt_id,
            "word1": prompt["clues"][0],
            "word2": prompt["clues"][1],
            "word3": prompt["clues"][2],
            "word4": prompt["clues"][3],
            "word5": prompt["clues"][4],
            "expected_solution": prompt["solution"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "difficulty_note": prompt["difficulty"],
            "duration_s": round(result.duration_s, 1),
            "is_first_call": 0,
        }

        if not result.success:
            logger.error(f"[fail] {model_name} | {prompt_id}: {result.error}")
            row = {
                **base_row,
                "raw_response": f"[{(result.error or 'ERROR').upper()}]",
                "predicted_word": "",
                "format_compliant": 0,
                "exact_match": 0,
                "same_root": 0,
                "levenshtein_normalized": 1.0,
                "is_similar_lev": 0,
                "is_clue_echo": 0,
                "status": "timeout" if result.error == "timeout" else "error",
            }
            writer.append_row(config.COMMERCIAL_SHEET_NAME, row)
            continue

        parsed = extract_predicted_word(result.raw_text)
        metrics = score(parsed.predicted_word, prompt["solution"], config.LEVENSHTEIN_SIMILARITY_THRESHOLD)
        clue_echo = is_clue_echo(parsed.predicted_word, prompt["clues"])

        row = {
            **base_row,
            "raw_response": result.raw_text,
            "predicted_word": parsed.predicted_word,
            "format_compliant": int(parsed.format_compliant),
            "exact_match": metrics.exact_match,
            "same_root": metrics.same_root,
            "levenshtein_normalized": metrics.levenshtein_normalized,
            "is_similar_lev": metrics.is_similar_lev,
            "is_clue_echo": clue_echo,
            "status": "ok",
        }
        writer.append_row(config.COMMERCIAL_SHEET_NAME, row)

        logger.info(
            f"[ok] {model_name} | {prompt_id} -> '{parsed.predicted_word}' "
            f"(expected '{prompt['solution']}', exact={metrics.exact_match}, "
            f"{result.duration_s:.1f}s"
            + (f", {result.completion_tokens} completion_tokens" if result.completion_tokens else "")
            + ")"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guillotine batch runner - commercial models, one run with no parameters")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of OpenRouter model IDs (default: config.OPENROUTER_COMMERCIAL)",
    )
    parser.add_argument("--limit-prompts", type=int, default=None, help="Run only the first N prompts")
    args = parser.parse_args()

    try:
        client = OpenRouterClient(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            timeout_s=config.OPENROUTER_TIMEOUT_S,
            site_url=config.OPENROUTER_SITE_URL,
            site_name=config.OPENROUTER_SITE_NAME,
            jsonl_backup_path=config.JSON_BACKUP_DIR / f"commercial_{SESSION_START}.jsonl",
            min_interval_s=config.OPENROUTER_MIN_INTERVAL_S,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if not client.check_service():
        logger.error("OpenRouter unreachable or invalid key.")
        sys.exit(1)

    requested_models = (
        [m.strip() for m in args.models.split(",")] if args.models else config.OPENROUTER_COMMERCIAL
    )
    if not requested_models:
        logger.error("No model specified (neither --models nor config.OPENROUTER_COMMERCIAL).")
        sys.exit(1)

    models = resolve_requested_models(client, requested_models)
    if not models:
        logger.error("None of the requested models turned out to be available on OpenRouter.")
        sys.exit(1)

    logger.info(f"Commercial models to run ({len(models)}/{len(requested_models)} requested): {models}")
    logger.info("No sampling parameters set - each provider uses its own default.")

    prompts = load_prompts(config.PROMPTS_JSON)
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
        logger.info(f"Smoke test: limited to the first {len(prompts)} prompts")

    writer = ResultsWriter(config.RESULTS_XLSX, config.RESULT_COLUMNS)

    for idx, model_name in enumerate(models, start=1):
        logger.info(f"=== [{idx}/{len(models)}] Model: {model_name} ===")
        try:
            run_model(client, writer, model_name, prompts)
        except Exception as e:
            logger.error(f"Unhandled error on {model_name}, moving to the next model: {e}")
            continue

    logger.info("Batch complete.")


if __name__ == "__main__":
    main()
