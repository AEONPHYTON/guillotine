"""
ollama_client.py

Minimal wrapper around Ollama's /api/generate, non-streaming: for this
task only the final text is needed, not the on-screen streaming output.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    success: bool
    raw_text: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0


class OllamaClient:
    """HTTP client for a local Ollama server."""

    def __init__(self, host: str, generate_timeout_s: int, load_timeout_s: int):
        self.host = host.rstrip("/")
        self.generate_timeout_s = generate_timeout_s
        self.load_timeout_s = load_timeout_s

    def check_service(self) -> bool:
        """Checks that the Ollama service is responding."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama service unreachable at {self.host}: {e}")
            return False

    def get_available_models(self) -> set[str]:
        """Returns the tags of the models actually present locally."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return {m["name"] for m in data.get("models", [])}
        except requests.exceptions.RequestException as e:
            logger.error(f"Error retrieving available models: {e}")
            return set()
        except (KeyError, ValueError) as e:
            logger.error(f"Malformed /api/tags response: {e}")
            return set()

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_k: int,
        top_p: float,
        is_first_call_for_model: bool = False,
        keep_alive: str = "5m",
    ) -> GenerationResult:
        """
        Runs a single non-streaming generation.

        is_first_call_for_model=True uses a longer timeout because the
        first call loads the model into VRAM (can take minutes for the
        larger models like mistral-small:22b).
        """
        timeout = self.load_timeout_s if is_first_call_for_model else self.generate_timeout_s
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
        }

        start = time.monotonic()
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return GenerationResult(
                success=True,
                raw_text=data.get("response", ""),
                duration_s=time.monotonic() - start,
            )
        except requests.exceptions.Timeout:
            logger.error(f"Timeout ({timeout}s) generating with {model}")
            return GenerationResult(success=False, error="timeout", duration_s=time.monotonic() - start)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama for {model}: {e}")
            return GenerationResult(success=False, error=str(e), duration_s=time.monotonic() - start)
        except ValueError as e:
            logger.error(f"Invalid JSON response from Ollama for {model}: {e}")
            return GenerationResult(success=False, error="invalid_json", duration_s=time.monotonic() - start)

    def unload(self, model: str) -> None:
        """Forces immediate unloading of the model from VRAM (keep_alive=0).

        Without this, with 22 models run in sequence Ollama can keep
        several models in VRAM at once if there's room, up to running out
        of memory (OOM) on a later model that would have fit on its own.
        """
        try:
            requests.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": 0},
                timeout=30,
            )
            logger.info(f"Model {model} unloaded from VRAM")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Could not force-unload {model}: {e}")
