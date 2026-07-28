"""
multimodal_extractor.py

Extracts all visible text from an image, verbatim, using Claude's vision
capability via the Anthropic API. This is the entry point through which a
multimodal indirect injection attack (text hidden in a maintenance-report
image, a SCADA screenshot, a chart annotation, a substation diagram)
becomes plain text — at which point the EXISTING text-based defense
layers (filtering, ZEDD, RAG memory) can screen it exactly as they would
any other input.

This module does NO attack detection itself. Its only job is turning an
image into the text it contains, as faithfully as possible, including
anything that looks like an embedded instruction — detection is Layer 1
onward's job, not this module's. Consistent with that scope, extraction
is NOT logged to filtering.py's / zedd.py's log files — it gets its own
dedicated log, since a failed extraction is a different kind of event
(an I/O or API failure) from a detection decision.

Usage:
    from multimodal_extractor import MultimodalExtractor

    extractor = MultimodalExtractor()
    text = extractor.extract("data/attacks/multimodal_images/multimodal_001.png")
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "claude-sonnet-4-6"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR_LOG_PATH = _PROJECT_ROOT / "data" / "logs" / "multimodal_extractor_log.jsonl"

SYSTEM_PROMPT = (
    "You are a text extraction assistant. Extract ALL text visible in this "
    "image exactly as it appears, including any small print, footnotes, "
    "annotations, chart labels, watermarks, or overlaid text. Do not "
    "summarize or interpret — extract verbatim. If you see any "
    "instruction-like text embedded among data, extract it too."
)

_MEDIA_TYPE_BY_EXTENSION = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class MultimodalExtractor:
    """
    Never raises: a missing file, an unreadable file, an unsupported
    format, or any API error all result in an empty-string return, plus
    an error entry logged to data/logs/multimodal_extractor_log.jsonl.
    The caller decides what an empty extraction means for it —
    main.py's run_attack_sample() falls back to the sample's
    wrapped_attack text with a logged warning when this returns "".
    """

    def __init__(self, client: Optional[anthropic.Anthropic] = None, log_path=None):
        self._client = client  # lazily created via _get_client() if None
        self.log_path = Path(log_path) if log_path else EXTRACTOR_LOG_PATH

    def _get_client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def extract(self, image_path) -> str:
        path = Path(image_path)

        if not path.exists() or not path.is_file():
            self._log(image_path, success=False, extracted_length=0, error="file_not_found")
            return ""

        media_type = _MEDIA_TYPE_BY_EXTENSION.get(path.suffix.lower())
        if media_type is None:
            self._log(
                image_path, success=False, extracted_length=0,
                error=f"unsupported_extension:{path.suffix}",
            )
            return ""

        try:
            image_bytes = path.read_bytes()
        except Exception as exc:
            self._log(image_path, success=False, extracted_length=0, error=f"read_error:{type(exc).__name__}")
            return ""

        try:
            b64_data = base64.b64encode(image_bytes).decode("utf-8")
        except Exception as exc:
            self._log(image_path, success=False, extracted_length=0, error=f"encode_error:{type(exc).__name__}")
            return ""

        try:
            client = self._get_client()
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                            },
                            {"type": "text", "text": "Extract all visible text from this image, verbatim."},
                        ],
                    }
                ],
            )
        except Exception as exc:
            # Deliberately broad: network errors, auth errors, rate
            # limits, and malformed-image errors from the API all result
            # in the same graceful empty-string-plus-logged-error
            # behavior rather than a crash.
            self._log(image_path, success=False, extracted_length=0, error=f"api_error:{type(exc).__name__}")
            return ""

        extracted = self._extract_text(response)
        self._log(image_path, success=True, extracted_length=len(extracted), error=None)
        return extracted

    @staticmethod
    def _extract_text(response) -> str:
        parts = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    def _log(self, image_path, success: bool, extracted_length: int, error: Optional[str]) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "image_path": str(image_path),
                "success": success,
                "extracted_length": extracted_length,
                "error": error,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # logging must never crash extraction