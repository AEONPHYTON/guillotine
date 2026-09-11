"""
openrouter_client.py

Minimal wrapper around OpenRouter's /chat/completions endpoint (OpenAI-
compatible format). Same return shape as ollama_client.py
(GenerationResult with success/raw_text/error/duration_s) so
response_parser/scoring/excel_writer can be reused unchanged - only the
transport changes, not the downstream processing.

Differences from Ollama, beyond just transport:
- No concept of "loading into VRAM": there's no is_first_call_for_model
  nor a differentiated timeout for the first call.
- No unload(): it's not our hardware, there's nothing to unload.
- Sampling parameters aren't guaranteed to be uniform across the
  different providers behind OpenRouter (e.g. some don't expose top_k) -
  see the note already taken in session about this.
- Every full response is also appended (JSONL) to a file, before being
  reduced to a GenerationResult - these calls cost money, a raw backup
  means not having to pay for them again if a detail we're discarding
  today (finish_reason, the actual provider behind the routing, the
  response id) needs to be revisited in the future.
"""
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    success: bool
    raw_text: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0
    # From the response's usage.completion_tokens field. Not yet written
    # to Excel (RESULT_COLUMNS doesn't include it until the same update is
    # made on the Ollama side too, for a fair comparison) - for now only
    # logged, see "next steps" in the session document.
    completion_tokens: Optional[int] = None


class OpenRouterClient:
    """HTTP client for the OpenRouter API (chat/completions format)."""

    def __init__(
            self,
            api_key: str,
            base_url: str,
            timeout_s: int,
            site_url: str = "",
            site_name: str = "",
            jsonl_backup_path: Optional[Path] = None,
            min_interval_s: float = 3.5,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY missing - set it in the .env file")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.site_url = site_url
        self.site_name = site_name
        self.jsonl_backup_path = jsonl_backup_path
        if self.jsonl_backup_path is not None:
            self.jsonl_backup_path.parent.mkdir(parents=True, exist_ok=True)
        # The free tier cap is 20 requests/minute (1 every 3s) - 3.5s of
        # margin for network/clock jitter. Enforced BEFORE each call, not
        # after already exceeding it: a model that responds in under a
        # second would otherwise fire requests much faster than the limit
        # allows (seen with z-ai/glm-5.2, ~2/second).
        self.min_interval_s = min_interval_s
        self._last_request_started_at: Optional[float] = None

    def _wait_for_rate_limit(self) -> None:
        if self._last_request_started_at is None:
            return
        elapsed = time.monotonic() - self._last_request_started_at
        remaining = self.min_interval_s - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _headers(self) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
        return headers

    def _append_backup(self, record: dict) -> None:
        """Appends a JSONL row with the full raw response. A failure here
        must never interrupt the batch - it's a backup, not the primary
        data (that remains the Excel file written by excel_writer)."""
        if self.jsonl_backup_path is None:
            return
        try:
            with open(self.jsonl_backup_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning(f"JSONL backup not written ({self.jsonl_backup_path}): {e}")

    def check_service(self) -> bool:
        """Checks that the API key is valid by querying /models."""
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=10)
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter unreachable or invalid key: {e}")
            return False

    def get_available_models(self) -> set[str]:
        """Returns the model IDs currently present in the OpenRouter catalog."""
        try:
            resp = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return {m["id"] for m in data.get("data", [])}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving the model catalog: {e}")
            return set()
        except (KeyError, ValueError) as e:
            logger.error(f"Malformed /models response: {e}")
            return set()

    def generate(
            self,
            model: str,
            prompt: str,
            temperature: Optional[float] = None,
            top_p: Optional[float] = None,
            top_k: Optional[int] = None,
            prompt_id: Optional[str] = None,
    ) -> GenerationResult:
        """
        Runs a single non-streaming chat generation.

        temperature/top_p/top_k are optional: if omitted (None), they are
        not included in the request and the underlying provider uses its
        own default value - used by commercial_runner.py to not impose
        any sampling control, same principle as answers collected by hand
        via copy-paste in chat.

        top_k is only sent if explicitly requested - many providers behind
        OpenRouter silently ignore it if they don't support it, it isn't
        guaranteed to be applied the way it is for Ollama.

        prompt_id is only used to label the row in the JSONL backup, it
        doesn't affect the call itself.
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k

        return self._generate_with_retry(model, payload, prompt_id, request_timestamp=None, attempt=1)

    def _generate_with_retry(
            self, model: str, payload: dict, prompt_id: Optional[str], request_timestamp, attempt: int
    ) -> GenerationResult:
        max_attempts = 4  # 1 initial attempt + up to 3 retries on 429
        self._wait_for_rate_limit()
        self._last_request_started_at = time.monotonic()
        request_timestamp = request_timestamp or datetime.now().isoformat()

        start = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            )
            duration = time.monotonic() - start

            if resp.status_code == 401:
                logger.error("OPENROUTER_API_KEY invalid or missing")
                return GenerationResult(success=False, error="unauthorized", duration_s=duration)
            if resp.status_code == 402:
                logger.error("OpenRouter credit exhausted")
                return GenerationResult(success=False, error="insufficient_credit", duration_s=duration)
            if resp.status_code == 429:
                self._log_rate_limit_headers(model, resp)
                if attempt >= max_attempts:
                    logger.error(
                        f"Persistent OpenRouter rate limit for {model} after {attempt} attempts"
                    )
                    return GenerationResult(success=False, error="rate_limited", duration_s=duration)
                # Exponential backoff (5s, 10s, 20s) if the server doesn't
                # give a precise indication - a single fixed 5s retry
                # wasn't enough in practice (seen on 2026-08-20).
                wait_s = self._parse_retry_after(resp) or (5.0 * (2 ** (attempt - 1)))
                logger.warning(
                    f"OpenRouter rate limit for {model} - attempt {attempt}/{max_attempts}, "
                    f"waiting {wait_s:.1f}s"
                )
                time.sleep(wait_s)
                return self._generate_with_retry(model, payload, prompt_id, request_timestamp, attempt + 1)

            resp.raise_for_status()
            data = resp.json()

            # Back up the full raw response exactly as it arrived, before
            # extracting only the fields we use - independent of how the
            # parsing below turns out, so that even a "no_choices" or
            # malformed response stays archived for later inspection.
            self._append_backup({
                "timestamp": request_timestamp,
                "model": model,
                "prompt_id": prompt_id,
                "temperature": payload.get("temperature"),
                "duration_s": round(duration, 2),
                "request": payload,
                "response": data,
            })

            choices = data.get("choices", [])
            if not choices:
                error_info = data.get("error", {})
                logger.warning(f"Response with no choices for {model}: {error_info}")
                return GenerationResult(success=False, error="no_choices", duration_s=duration)

            message = choices[0].get("message", {}) or {}
            text = message.get("content") or ""

            # Some reasoning models expose their thinking in a separate
            # field (not always present): we fold it back into raw_text
            # with the same <think> tag already handled by
            # response_parser.py for local models with inline reasoning,
            # so downstream extraction works identically.
            reasoning = message.get("reasoning")
            if reasoning:
                text = f"<think>{reasoning}</think>{text}"

            usage = data.get("usage", {}) or {}
            completion_tokens = usage.get("completion_tokens")

            return GenerationResult(
                success=True,
                raw_text=text,
                duration_s=duration,
                completion_tokens=completion_tokens,
            )
        except requests.exceptions.Timeout:
            duration = time.monotonic() - start
            logger.error(f"Timeout ({self.timeout_s}s) generating with {model}")
            return GenerationResult(success=False, error="timeout", duration_s=duration)
        except requests.exceptions.RequestException as e:
            duration = time.monotonic() - start
            logger.error(f"Error calling OpenRouter for {model}: {e}")
            return GenerationResult(success=False, error=str(e), duration_s=duration)
        except ValueError as e:
            duration = time.monotonic() - start
            logger.error(f"Invalid JSON response from OpenRouter for {model}: {e}")
            return GenerationResult(success=False, error="invalid_json", duration_s=duration)

    @staticmethod
    def _parse_retry_after(resp: "requests.Response") -> Optional[float]:
        """Reads Retry-After or X-RateLimit-Reset from the 429 response, if present."""
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return None

    @staticmethod
    def _log_rate_limit_headers(model: str, resp: "requests.Response") -> None:
        """Logs the diagnostic rate-limit headers, if present.

        Useful for telling apart a limit on your own account (clears
        quickly, X-RateLimit-Reset is close) from a saturated shared
        capacity on the model (no useful header, or a far-off reset) -
        see the discussion on 2026-08-20 about z-ai/glm-5.2.
        """
        headers_of_interest = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"]
        found = {h: resp.headers.get(h) for h in headers_of_interest if resp.headers.get(h) is not None}
        if found:
            logger.info(f"Rate limit headers for {model}: {found}")
        else:
            logger.info(
                f"No rate-limit headers in the 429 response for {model} - "
                f"likely saturated shared capacity, not a limit on your own account"
            )
