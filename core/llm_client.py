from __future__ import annotations
import json
import requests
from typing import Iterator

from utils.config import get_config
from utils.logger import log_step, log_error


class OllamaClient:

    def __init__(self) -> None:
        cfg = get_config().llm
        self.model       = cfg.model
        self.base_url    = cfg.base_url.rstrip("/")
        self.temperature = cfg.temperature
        self.max_tokens  = cfg.max_tokens

    def _endpoint(self) -> str:
        return f"{self.base_url}/api/generate"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            return False

    def generate(self, prompt: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                "Ollama is not running.\n"
                "Open a new terminal and run: ollama serve\n"
                f"Then make sure model is pulled: ollama pull {self.model}"
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        log_step("LLM", f"Generating with {self.model}...")

        try:
            response = requests.post(
                self._endpoint(),
                json=payload,
                stream=True,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.Timeout:
            log_error("Ollama timed out. Try a shorter prompt or smaller model.")
            raise
        except requests.exceptions.HTTPError as e:
            log_error(f"Ollama HTTP error: {e}")
            raise

    def stream(self, prompt: str) -> Iterator[str]:
        if not self.is_available():
            raise RuntimeError("Ollama is not running.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        with requests.post(
            self._endpoint(),
            json=payload,
            stream=True,
            timeout=300
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done", False):
                        break