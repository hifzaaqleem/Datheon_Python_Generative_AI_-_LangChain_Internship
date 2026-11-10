"""
Gemini-only client (google-genai SDK). Free-tier models.
Text, images, PDFs and audio via multimodal input.
Soft validation + repair retry on hard structural errors only.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional, Tuple

from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from prompts import build_prompt, SYSTEM_INSTRUCTION
from registry import get_spec
from schemas import SoftValidationResult

# Verify against your account:
#   python -c "from google import genai; c=genai.Client(); print([m.name for m in c.models.list()])"
FREE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
]
DEFAULT_MODEL = "gemini-3.5-flash-lite"

IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}
PDF_MIMES = {"application/pdf"}
AUDIO_MIMES = {
    "audio/wav", "audio/mpeg", "audio/aac", "audio/ogg", "audio/flac", "audio/aiff",
    # Common phone/browser formats. Not in Gemini's documented list as far as I know:
    # if the API rejects them, convert first (ffmpeg -i in.m4a out.wav).
    "audio/mp4", "audio/webm",
}
SUPPORTED_MIMES = IMAGE_MIMES | PDF_MIMES | AUDIO_MIMES

MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "audio/mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/m4a": "audio/mp4",
    "audio/x-m4a": "audio/mp4",
}
MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".heic": "image/heic", ".heif": "image/heif", ".pdf": "application/pdf",
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".webm": "audio/webm", ".ogg": "audio/ogg", ".flac": "audio/flac", ".aiff": "audio/aiff",
}

# Inline uploads share the request size limit (~20 MB total). Larger files need the Files API.
MAX_INLINE_BYTES = 18 * 1024 * 1024

AUDIO_NOTE = (
    "\n\nThe primary source is the attached audio. Listen to the speech, then extract the fields. "
    "If the speaker corrects themselves, use the final corrected value. "
    "Do not invent values that were not said."
)
DOC_NOTE = (
    "\n\nThe primary document is the attached image or PDF. Read all visible text, tables and labels. "
    "Do not invent values that are not visible."
)


def _extract_json_from_text(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not extract valid JSON from model response")


def _load_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if key:
        return key.strip()
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
            key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if key:
                return key.strip()
        except Exception:
            pass
    raise ValueError(
        "Gemini API key not found.\n"
        "Set GOOGLE_API_KEY in the environment or in a .env file.\n"
        "Get a free key: https://aistudio.google.com/apikey"
    )


def _guess_mime(filename: str, fallback: str = "application/octet-stream") -> str:
    ext = os.path.splitext((filename or "").lower())[1]
    return MIME_BY_EXT.get(ext, fallback)


def _is_transient(exc: BaseException) -> bool:
    """Rate limits and server-side errors are worth retrying; nothing else is."""
    try:
        from google.genai import errors
    except ImportError:
        return False
    if isinstance(exc, errors.ServerError):
        return True
    return isinstance(exc, errors.APIError) and getattr(exc, "code", None) in (429, 503)


class GeminiClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError(
                "Package 'google-genai' is not installed in THIS Python environment.\n"
                "Run: python -m pip install google-genai\n"
                "and start the app with: python -m streamlit run app.py"
            ) from e

        self.api_key = _load_api_key()
        self.model_name = model
        self._types = types
        self.client = genai.Client(api_key=self.api_key)
        self._native_schema = True  # flips to False if the API rejects the response schema

    # ------------------------------------------------------------------ helpers
    def _media_part(self, file_bytes: bytes, file_mime: Optional[str], file_name: Optional[str]):
        mime = (file_mime or _guess_mime(file_name or "")).lower()
        mime = MIME_ALIASES.get(mime, mime)
        if mime not in SUPPORTED_MIMES:
            raise ValueError(
                f"Unsupported file type: {mime}. Use PNG, JPEG, WebP, HEIC, PDF, or audio (wav/mp3/aac/ogg/flac)."
            )
        if len(file_bytes) > MAX_INLINE_BYTES:
            raise ValueError(
                f"File is {len(file_bytes) / 1e6:.1f} MB; inline limit is about {MAX_INLINE_BYTES // (1024 * 1024)} MB. "
                "Use a smaller file or upload via the Files API."
            )
        return self._types.Part.from_bytes(data=file_bytes, mime_type=mime), mime

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=4, max=30),
        retry=retry_if_exception(_is_transient),
        reraise=True,
    )
    def _call(self, contents: list, temperature: float, schema_cls) -> str:
        cfg = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "system_instruction": SYSTEM_INSTRUCTION,
        }
        if schema_cls is not None:
            cfg["response_schema"] = schema_cls
        resp = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=self._types.GenerateContentConfig(**cfg),
        )
        return resp.text or ""

    def _generate(self, contents: list, temperature: float, schema_cls) -> str:
        if self._native_schema:
            try:
                return self._call(contents, temperature, schema_cls)
            except Exception as e:  # noqa: BLE001
                from google.genai import errors

                # Schema rejected (unsupported keyword etc.): fall back to prompt-only JSON.
                if isinstance(e, errors.ClientError) and getattr(e, "code", None) == 400:
                    self._native_schema = False
                else:
                    raise
        return self._call(contents, temperature, None)

    # ------------------------------------------------------------------ public API
    def extract(
        self,
        doc_type: str,
        text: str = "",
        temperature: float = 0.0,
        max_repairs: int = 2,
        file_bytes: Optional[bytes] = None,
        file_mime: Optional[str] = None,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None,  # alias for file_mime (used by run_eval_media.py)
    ) -> Tuple[SoftValidationResult, str]:
        """Extract from text and/or an image / PDF / audio file."""
        text = (text or "").strip()
        if not file_bytes and not text:
            raise ValueError("Provide text and/or a PDF, image, or audio file.")

        spec = get_spec(doc_type)
        base = build_prompt(doc_type, text or "(see attached document image/PDF/audio)")

        parts: list = []
        extra = ""
        if file_bytes:
            part, mime = self._media_part(file_bytes, file_mime or mime_type, file_name)
            parts = [part]
            extra = AUDIO_NOTE if mime in AUDIO_MIMES else DOC_NOTE

        def build_contents(note: str = "") -> list:
            return [base + extra + note] + parts

        note, last_raw = "", ""
        last_err: Optional[Exception] = None

        for attempt in range(max_repairs + 1):
            try:
                last_raw = self._generate(build_contents(note), temperature, spec.schema)
                data = _extract_json_from_text(last_raw)
                model = spec.schema.model_validate(data)
                soft = spec.soft_validate(model)
                return soft, json.dumps(soft.data, indent=2, ensure_ascii=False)
            except (ValidationError, ValueError) as e:  # ValidationError is a ValueError in pydantic v2
                last_err = e
                if attempt >= max_repairs:
                    break
                note = (
                    f"\n\nYour previous JSON output was:\n{last_raw[:4000]}\n\n"
                    f"It failed structural validation:\n{e}\n\n"
                    "Return corrected JSON matching the schema types. "
                    "Do not invent missing facts; use null. "
                    "Do not change numeric values to make arithmetic work."
                )

        raise ValueError(
            f"Extraction failed after {max_repairs} repair attempt(s): {last_err}\n"
            f"Last output:\n{last_raw[:2000]}"
        ) from last_err


def create_client(model: Optional[str] = None) -> GeminiClient:
    return GeminiClient(model=model or DEFAULT_MODEL)