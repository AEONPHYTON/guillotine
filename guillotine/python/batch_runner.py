"""
batch_runner.py

CLI batch runner for automatic generation on the local models via Ollama.
For each model: load it once, iterate over 3 temperatures x 30 prompts,
save each response incrementally, unload the model from VRAM, move on to
the next one. Resumable: on restart it skips (model, prompt_id)
combinations already present in the current temperature's sheet - the
workbook itself is the checkpoint, see excel_writer.py.

Usage:
    python batch_runner.py
    python batch_runner.py --models mistral-small:22b --limit-prompts 3   (smoke test)

    # Session 1 - fast models (everything except deepseek-r1:* and gpt-oss:20b)
python batch_runner.py --models nemotron-3-nano:4b,tinyllama:1.1b,phi3:mini,phi3.5:latest,nous-hermes2:latest,zephyr:7b,mistral:7b,vicuna:13b-q4_K_M,qwen2.5:7b,qwen3.5:9b,llama3.2:3b,llama3.1:8b,gemma2:2b,gemma2:9b,gemma4:e4b,mistral-small:22b,qwen2.5:14b,starcoder2:15b,granite3-dense:8b,gemma3:12b,gemma4:12b,qwen3:14b

# Session 2 - the 4 reasoning models, launch when it can run for hours (overnight?)
python batch_runner.py --models deepseek-r1:1.5b,deepseek-r1:7b,deepseek-r1:14b,gpt-oss:20b

to redo...
python batch_runner.py --models gemma4:12b,qwen3.5:9b,deepseek-r1:7b,gpt-oss:20b,deepseek-r1:1.5b,deepseek-r1:14b,starcoder2:15b-instruct
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import config
from excel_writer import ResultsWriter
from ollama_client import OllamaClient
from response_parser import extract_predicted_word
from scoring import is_clue_echo, score

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
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


def is_reasoning_model(model_name: str) -> bool:
    return any(marker in model_name.lower() for marker in config.REASONING_MODEL_MARKERS)


def resolve_available_models(client: OllamaClient, requested_models: list[str]) -> list[str]:
    """Cross-checks the configured list against the models actually present on Ollama.

    A requested model that isn't found (wrong tag, missing pull) is
    skipped with a warning instead of failing the whole batch.
    """
    available = client.get_available_models()
    resolved = []
    for model in requested_models:
        if model in available:
            resolved.append(model)
        else:
            logger.warning(
                f"Model configured but not present on Ollama (missing pull or "
                f"different tag - check with 'ollama list'): {model}"
            )
    return resolved


def run_model(
        client: OllamaClient,
        writer: ResultsWriter,
        model_name: str,
        prompts: list[dict],
        temperatures: list[float],
) -> None:
    is_reasoning = is_reasoning_model(model_name)
    if is_reasoning:
        logger.info(
            f"{model_name} is a known reasoning model - expect much "
            f"slower generations (seen 15-95s/call on gpt-oss:20b vs 2-3s "
            f"for non-reasoning models)"
        )
    model_loaded = False

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
                top_k=config.TOP_K,
                top_p=config.TOP_P,
                is_first_call_for_model=not model_loaded,
                keep_alive=config.OLLAMA_KEEP_ALIVE_DURING_MODEL,
            )

            if not result.success:
                logger.error(f"[fail] {model_name} | T={temperature} | {prompt_id}: {result.error}")
                # A failure also becomes an explicit row (status=timeout/error)
                # instead of a silent gap - it still remains retryable on the
                # next restart: completed_pairs() only treats it as "already
                # done" if status is 'ok', see ResultsWriter.
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

            was_first_call = not model_loaded
            model_loaded = True  # the long timeout is only needed until a valid response comes in

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
                "is_first_call": int(was_first_call),
                "status": "ok",
                "is_clue_echo": clue_echo,
            }
            writer.append_row(sheet_name, row)

            logger.info(
                f"[ok] {model_name} | T={temperature} | {prompt_id} -> "
                f"'{parsed.predicted_word}' (expected '{prompt['solution']}', "
                f"exact={metrics.exact_match}, compliant={parsed.format_compliant}, "
                f"{result.duration_s:.1f}s)"
            )

    client.unload(model_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guillotine batch runner - local models via Ollama"
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models to run (default: list in config.py)",
    )
    parser.add_argument(
        "--limit-prompts", type=int, default=None,
        help="Run only the first N prompts (smoke test on a new model)",
    )
    args = parser.parse_args()

    client = OllamaClient(
        host=config.OLLAMA_HOST,
        generate_timeout_s=config.OLLAMA_GENERATE_TIMEOUT_S,
        load_timeout_s=config.OLLAMA_LOAD_TIMEOUT_S,
    )

    if not client.check_service():
        logger.error(
            f"Ollama unreachable at {config.OLLAMA_HOST}. "
            f"Start it (ollama serve, or the Ollama app) and try again."
        )
        sys.exit(1)

    requested_models = (
        [m.strip() for m in args.models.split(",")] if args.models else config.LOCAL_MODELS
    )
    models = resolve_available_models(client, requested_models)

    if not models:
        logger.error("None of the requested models are available on Ollama (run 'ollama list' to check).")
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
