"""Pipeline to run scenario QA through OpenRouter inference and judge models."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import mimetypes
import os
import random
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, TypeVar

try:
    from openai import APIError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - dependency not installed yet
    OpenAI = None
    APIError = None
    RateLimitError = None

try:
    from PIL import Image, PngImagePlugin
except ImportError:  # pragma: no cover - dependency not installed yet
    Image = None
    PngImagePlugin = None


# ---------------------------- Data Structures ---------------------------------


@dataclass
class Scenario:
    number: int
    folder_name: str
    scenario: str
    qa_th: str
    qa_en: str
    expected_th: str
    expected_en: str
    rubric_th: str

    @property
    def prefixed_folder(self) -> str:
        return f"{self.number:02d}_{self.folder_name}"


@dataclass
class InferenceResult:
    text: str
    raw_response: Dict[str, object]


@dataclass
class JudgeResult:
    verdict: Dict[str, object]
    raw_response: Dict[str, object]


@dataclass
class RetryConfig:
    max_attempts: int = 5
    initial_delay: float = 1.0


T = TypeVar("T")


# ----------------------------- Utilities ---------------------------------------


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise SystemExit(f"Missing required prompt file: {path}")


def read_scenarios(csv_path: Path) -> List[Scenario]:
    scenarios: List[Scenario] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenarios.append(
                Scenario(
                    number=int(row["Number"]),
                    folder_name=row["Folder Name"].strip(),
                    scenario=row["Scenario"].strip(),
                    qa_th=row["QA_TH"].strip(),
                    qa_en=row["QA_EN"].strip(),
                    expected_th=row["Expected_TH"].strip(),
                    expected_en=row["Expected_EN"].strip(),
                    rubric_th=row["Rubric_TH"].strip(),
                )
            )
    return scenarios


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.as_posix())
    if mime_type:
        return mime_type
    # Default to PNG which matches the dataset
    return "image/png"


def read_image_bytes_without_metadata(path: Path) -> bytes:
    if Image is None:
        raise SystemExit(
            "Pillow is required to strip image metadata. Install with `pip install Pillow`."
        )
    with Image.open(path) as img:
        img_format = (img.format or "PNG").upper()
        output = BytesIO()
        img_no_metadata = img.copy()
        if img_format in {"JPEG", "JPG"}:
            img_no_metadata.save(output, format="JPEG", quality=95, optimize=True)
        elif img_format == "PNG":
            pnginfo = PngImagePlugin.PngInfo() if PngImagePlugin else None
            save_kwargs = {"format": "PNG", "optimize": True}
            if pnginfo is not None:
                save_kwargs["pnginfo"] = pnginfo
            img_no_metadata.save(output, **save_kwargs)
        else:
            # Convert other formats (e.g., WebP) to PNG to ensure metadata removal
            img_no_metadata.convert(
                "RGBA" if img_no_metadata.mode == "P" else "RGB"
            ).save(output, format="PNG", optimize=True)
        return output.getvalue()


def encode_image(path: Path) -> Dict[str, object]:
    image_bytes = read_image_bytes_without_metadata(path)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    mime = guess_mime_type(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
    }


def find_scenario_images(images_root: Path, scenario: Scenario) -> List[Path]:
    folder = images_root / scenario.prefixed_folder
    if not folder.exists():
        raise SystemExit(
            f"Image folder not found for scenario {scenario.number}: {folder}"
        )
    candidates = sorted(folder.glob(f"{scenario.number:02d}_*.png"))
    if len(candidates) < 2:
        raise SystemExit(
            f"Expected at least two images in {folder}, found {len(candidates)}."
        )
    # Use the first two sorted images per requirement ScenarioNumber_01 + ScenarioNumber_02
    return candidates[:2]


def detect_appearance(image_path: Path) -> str:
    """
    Detect appearance type (Thai/Foreign) from image filename.
    Convention:
      - ScenarioNumber_01.png or ScenarioNumber_1.png = Thai
      - ScenarioNumber_02.png or ScenarioNumber_2.png = Foreign
    """
    filename = image_path.stem  # Get filename without extension

    # Check for _01 or _1 pattern (Thai)
    if filename.endswith("_01") or filename.endswith("_1"):
        return "Thai"
    # Check for _02 or _2 pattern (Foreign)
    elif filename.endswith("_02") or filename.endswith("_2"):
        return "Foreign"
    else:
        # Fallback: try to extract the last number
        match = re.search(r"_(\d+)$", filename)
        if match:
            num = int(match.group(1))
            if num == 1:
                return "Thai"
            elif num == 2:
                return "Foreign"

        logging.warning(
            f"Could not detect appearance from filename: {filename}, defaulting to 'Unknown'"
        )
        return "Unknown"


# ----------------------------- Retry Helpers -----------------------------------


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def is_retryable_error(error: Exception) -> bool:
    if RateLimitError and isinstance(error, RateLimitError):
        return True
    if APIError and isinstance(error, APIError):
        status = getattr(error, "status_code", None) or getattr(error, "status", None)
        if status in RETRYABLE_STATUS_CODES:
            return True
    status = getattr(error, "status_code", None)
    if status in RETRYABLE_STATUS_CODES:
        return True
    # Some exceptions hold HTTP response detail
    response = getattr(error, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status in RETRYABLE_STATUS_CODES:
            return True
        body = getattr(response, "body", None)
        if isinstance(body, dict):
            code = (
                body.get("error", {}).get("code")
                if isinstance(body.get("error"), dict)
                else body.get("code")
            )
            if isinstance(code, int) and code in RETRYABLE_STATUS_CODES:
                return True
            if (
                isinstance(code, str)
                and code.isdigit()
                and int(code) in RETRYABLE_STATUS_CODES
            ):
                return True
    # String fallback
    message = str(error)
    if any(
        token in message for token in ("429", "rate limit", "temporarily rate-limited")
    ):
        return True
    return False


def request_with_retry(
    callable_fn: Callable[[], T], retry_config: RetryConfig, label: str
) -> T:
    max_attempts = max(retry_config.max_attempts, 1)
    base_delay = max(retry_config.initial_delay, 0.1)
    for attempt in range(max_attempts):
        try:
            return callable_fn()
        except Exception as exc:  # pragma: no cover - network interactions
            if not is_retryable_error(exc) or attempt + 1 >= max_attempts:
                raise
            sleep_seconds = base_delay * (2**attempt)
            jitter = random.uniform(0, base_delay)
            total_sleep = sleep_seconds + jitter
            print(
                f"[{label}] Retryable error ({exc}); sleeping {total_sleep:.2f}s before retry {attempt + 1}/{max_attempts - 1}",
                file=sys.stderr,
            )
            time.sleep(total_sleep)
    raise RuntimeError("Exceeded retry attempts")  # Should be unreachable


# ----------------------------- OpenRouter Client -------------------------------


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://openrouter.ai/api/v1",
        referer: Optional[str] = None,
        app_title: Optional[str] = None,
        request_timeout: float = 60.0,
    ) -> None:
        if OpenAI is None:
            raise SystemExit(
                "The 'openai' package is required. Install with `pip install openai`."
            )
        self._client = OpenAI(
            base_url=base_url, api_key=api_key, timeout=request_timeout
        )
        self._referer = referer
        self._app_title = app_title

    def _inject_headers(self, kwargs: Dict[str, object]) -> Dict[str, object]:
        extra_headers = {}
        if self._referer:
            extra_headers["HTTP-Referer"] = self._referer
        if self._app_title:
            extra_headers["X-Title"] = self._app_title
        if extra_headers:
            kwargs.setdefault("extra_headers", extra_headers)
        return kwargs

    def chat_completions_create(
        self,
        *,
        model: str,
        messages: List[Dict[str, object]],
        **kwargs,
    ) -> Dict[str, object]:
        kwargs = self._inject_headers(kwargs)
        completion = self._client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        if hasattr(completion, "model_dump"):
            return completion.model_dump()
        if isinstance(completion, dict):
            return completion
        if isinstance(completion, str):
            try:
                return json.loads(completion)
            except json.JSONDecodeError as exc:  # pragma: no cover - unexpected shape
                raise ValueError(
                    f"Unexpected string response: {completion[:200]}"
                ) from exc
        raise TypeError(
            f"Unsupported response type from OpenRouter client: {type(completion)!r}"
        )


def build_messages(system_prompt: str, user_content: object) -> List[Dict[str, object]]:
    messages: List[Dict[str, object]] = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    return messages


def extract_output_text(response_payload: Dict[str, object]) -> str:
    # Attempt to extract the unified output text from the responses payload
    output = response_payload.get("output")
    if isinstance(output, list):
        chunks: List[str] = []
        for item in output:
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"}:
                    chunks.append(content.get("text", ""))
        if chunks:
            return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    # Fallback to old-style output_text field
    if "output_text" in response_payload:
        text = response_payload["output_text"]
        if isinstance(text, str):
            return text.strip()
    # Fallback to choices (chat completions compatibility)
    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks = [c.get("text", "") for c in content if isinstance(c, dict)]
            return "\n".join(chunks).strip()
    raise ValueError("Unable to extract text from OpenRouter response payload")


def parse_json_response(text: str) -> Dict[str, object]:
    cleaned = text.strip()

    # Handle markdown code blocks
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            # Drop opening fence (e.g., ```json)
            lines = lines[1:]
        # Remove closing fence if present
        while lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()

    # If still no JSON found, try to extract JSON object from the text
    if not cleaned or not (cleaned.startswith("{") or cleaned.startswith("[")):
        # Search for JSON object in the text
        # Look for the last JSON object in the text (usually at the end)
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
        else:
            # Try to find JSON after common markers
            markers = ["```json", "```", "output:", "result:", "json:"]
            for marker in markers:
                if marker in cleaned.lower():
                    parts = cleaned.lower().split(marker)
                    if len(parts) > 1:
                        # Get everything after the marker
                        potential_json = cleaned[
                            cleaned.lower().index(marker) + len(marker) :
                        ].strip()
                        # Clean up closing markers
                        if "```" in potential_json:
                            potential_json = potential_json[
                                : potential_json.index("```")
                            ].strip()
                        if potential_json.startswith("{"):
                            cleaned = potential_json
                            break

    if not cleaned:
        raise ValueError("Empty JSON response")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: try to find any JSON-like structure
        # Find the last complete JSON object
        matches = list(re.finditer(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", text, re.DOTALL))
        if matches:
            # Try parsing from the last match backwards
            for match in reversed(matches):
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"Could not parse JSON from response")


def call_inference(
    client: OpenRouterClient,
    *,
    model: str,
    system_prompt: str,
    question: str,
    image_paths: Optional[Iterable[Path]] = None,
    max_output_tokens: Optional[int] = None,
    retry_config: RetryConfig,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> InferenceResult:
    if image_paths:
        user_parts: List[Dict[str, object]] = [{"type": "text", "text": question}]
        for image_path in image_paths:
            user_parts.append(encode_image(image_path))
        user_content: object = user_parts
    else:
        user_content = question
    messages = build_messages(system_prompt, user_content)
    kwargs = {}
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if seed is not None:
        kwargs["seed"] = seed
    response_payload = request_with_retry(
        lambda: client.chat_completions_create(
            model=model, messages=messages, **kwargs
        ),
        retry_config=retry_config,
        label="inference",
    )
    answer_text = extract_output_text(response_payload)
    return InferenceResult(text=answer_text, raw_response=response_payload)


def call_inference_multiple(
    client: OpenRouterClient,
    *,
    model: str,
    system_prompt: str,
    question: str,
    image_paths: Optional[Iterable[Path]] = None,
    max_output_tokens: Optional[int] = None,
    retry_config: RetryConfig,
    num_inferences: int = 20,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    sleep_between_inference: float = 0.0,
    max_workers: Optional[int] = None,
) -> List[InferenceResult]:
    """
    Call inference model multiple times with the same input.

    Args:
        max_workers: Maximum number of concurrent API calls. If None, runs sequentially.
    """
    if max_workers is None or max_workers <= 1:
        # Sequential execution
        results: List[InferenceResult] = []
        for i in range(num_inferences):
            result = call_inference(
                client,
                model=model,
                system_prompt=system_prompt,
                question=question,
                image_paths=image_paths,
                max_output_tokens=max_output_tokens,
                retry_config=retry_config,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            results.append(result)
            if sleep_between_inference > 0 and i < num_inferences - 1:
                time.sleep(sleep_between_inference)
        return results

    # Concurrent execution
    def run_single_inference(idx: int) -> tuple[int, InferenceResult]:
        """Wrapper to track index for ordering results."""
        if sleep_between_inference > 0 and idx > 0:
            # Stagger requests to avoid overwhelming the API
            time.sleep(sleep_between_inference * (idx % max_workers))

        result = call_inference(
            client,
            model=model,
            system_prompt=system_prompt,
            question=question,
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            retry_config=retry_config,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return idx, result

    # Submit all tasks
    results_dict: Dict[int, InferenceResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_inference, i) for i in range(num_inferences)
        ]

        for future in as_completed(futures):
            idx, result = future.result()
            results_dict[idx] = result

    # Return results in original order
    return [results_dict[i] for i in range(num_inferences)]


def load_rubrics(rubric_json_path: Path) -> Dict[int, Dict[str, object]]:
    """Load rubrics from JSON file and index by scenario number."""
    with rubric_json_path.open("r", encoding="utf-8") as f:
        rubrics_list = json.load(f)

    rubrics_by_number = {}
    for rubric in rubrics_list:
        number = rubric.get("number")
        if number is not None:
            rubrics_by_number[number] = rubric

    return rubrics_by_number


def build_rubric_payload(
    scenario: Scenario, rubrics_by_number: Dict[int, Dict[str, object]]
) -> Dict[str, object]:
    """Build rubric payload from loaded JSON rubrics, fallback to scenario data if missing."""
    rubric = rubrics_by_number.get(scenario.number)

    if rubric:
        # Use the comprehensive rubric from JSON
        return rubric
    else:
        # Fallback to old placeholder method if rubric not found
        logging.warning(
            f"Rubric not found for scenario {scenario.number}, using placeholder"
        )
        expected_actions = [
            item.strip() for item in scenario.expected_en.split(",") if item.strip()
        ]
        acceptable_variants = expected_actions
        return {
            "number": scenario.number,
            "scenario_id": scenario.folder_name,
            "norm_title": scenario.scenario,
            "description": scenario.rubric_th,
            "expected_actions": expected_actions,
            "acceptable_variants": acceptable_variants,
            "red_flags": [],
            "visual_cues": [],
            "reasoning_keys": expected_actions,
            "exceptions": [],
            "notes": [],
            "severity": {},
        }


def call_judge(
    client: OpenRouterClient,
    *,
    model: str,
    system_prompt: str,
    question: str,
    answer: str,
    rubric: Dict[str, object],
    max_output_tokens: Optional[int] = None,
    retry_config: RetryConfig,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> JudgeResult:
    payload = {
        "question": question,
        "answer": answer,
        "rubric": rubric,
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    messages = build_messages(system_prompt, user_content)
    kwargs = {}
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if seed is not None:
        kwargs["seed"] = seed
    response_payload = request_with_retry(
        lambda: client.chat_completions_create(
            model=model, messages=messages, **kwargs
        ),
        retry_config=retry_config,
        label="judge",
    )
    output_text = extract_output_text(response_payload)
    try:
        verdict = parse_json_response(output_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Judge model did not return valid JSON. Received:\n{output_text}"
        ) from exc
    return JudgeResult(verdict=verdict, raw_response=response_payload)


def call_judge_multiple_answers(
    client: OpenRouterClient,
    *,
    model: str,
    system_prompt: str,
    question: str,
    answers: List[str],
    rubric: Dict[str, object],
    max_output_tokens: Optional[int] = None,
    retry_config: RetryConfig,
) -> JudgeResult:
    """Call judge model with multiple inference answers."""
    payload = {
        "question": question,
        "answers": answers,  # Changed from "answer" to "answers" (list)
        "rubric": rubric,
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    messages = build_messages(system_prompt, user_content)
    kwargs = {}
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    response_payload = request_with_retry(
        lambda: client.chat_completions_create(
            model=model, messages=messages, **kwargs
        ),
        retry_config=retry_config,
        label="judge",
    )
    output_text = extract_output_text(response_payload)
    try:
        verdict = parse_json_response(output_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Judge model did not return valid JSON. Received:\n{output_text}"
        ) from exc
    return JudgeResult(verdict=verdict, raw_response=response_payload)


def call_judge_batch(
    client: OpenRouterClient,
    *,
    model: str,
    system_prompt: str,
    question: str,
    answers: List[str],
    rubric: Dict[str, object],
    max_output_tokens: Optional[int] = None,
    retry_config: RetryConfig,
    sleep_between_requests: float = 0.0,
    max_workers: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[JudgeResult]:
    """
    Call judge model for multiple answers, optionally in parallel.

    Args:
        answers: List of inference answers to judge
        max_workers: Maximum number of concurrent judge calls. If None, runs sequentially.
    """
    if max_workers is None or max_workers <= 1:
        # Sequential execution
        results: List[JudgeResult] = []
        for i, answer in enumerate(answers):
            result = call_judge(
                client,
                model=model,
                system_prompt=system_prompt,
                question=question,
                answer=answer,
                rubric=rubric,
                max_output_tokens=max_output_tokens,
                retry_config=retry_config,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )
            results.append(result)
            if sleep_between_requests > 0 and i < len(answers) - 1:
                time.sleep(sleep_between_requests)
        return results

    # Concurrent execution
    def run_single_judge(idx: int, answer: str) -> tuple[int, JudgeResult]:
        """Wrapper to track index for ordering results."""
        if sleep_between_requests > 0 and idx > 0:
            # Stagger requests to avoid overwhelming the API
            time.sleep(sleep_between_requests * (idx % max_workers))

        result = call_judge(
            client,
            model=model,
            system_prompt=system_prompt,
            question=question,
            answer=answer,
            rubric=rubric,
            max_output_tokens=max_output_tokens,
            retry_config=retry_config,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return idx, result

    # Submit all tasks
    results_dict: Dict[int, JudgeResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(run_single_judge, i, answer)
            for i, answer in enumerate(answers)
        ]

        for future in as_completed(futures):
            idx, result = future.result()
            results_dict[idx] = result

    # Return results in original order
    return [results_dict[i] for i in range(len(answers))]


# ----------------------------- CSV persistence ---------------------------------


def build_result_header(num_inferences: int) -> list:
    """Build CSV header dynamically based on the number of inferences."""
    header = [
        "run_id",
        "timestamp_utc",
        "inference_model",
        "judge_model",
        "number",
        "scenario_id",
        "scenario_text",
        "language",
        "appearance",
        "image_file",
        "question",
        "temperature",
        "top_p",
        "seed",
    ]

    # Add inference answer columns
    for i in range(1, num_inferences + 1):
        header.append(f"inference_answer_{i}")

    # Add judge result columns for each inference
    for i in range(1, num_inferences + 1):
        header.extend(
            [
                f"judge_{i}_C0_final_action_or_verdict_correct",
                f"judge_{i}_C1_cultural_correctness_in_thought_process",
                f"judge_{i}_C2_presence_of_cultural_reasoning",
                f"judge_{i}_D1_visual_grounding",
                f"judge_{i}_D2_overclaiming",
                f"judge_{i}_D3_politeness_nonstereotype",
                f"judge_{i}_D4_clarity_usefulness",
                f"judge_{i}_rationale0",
                f"judge_{i}_rationale1",
                f"judge_{i}_rationale2",
                f"judge_{i}_flags",
            ]
        )

    # Add raw response columns
    for i in range(1, num_inferences + 1):
        header.append(f"inference_raw_{i}")

    for i in range(1, num_inferences + 1):
        header.append(f"judge_raw_{i}")

    return header


class CsvResultWriter:
    def __init__(self, output_path: Path, num_inferences: int) -> None:
        self.fieldnames = build_result_header(num_inferences)
        self.num_inferences = num_inferences
        self._handle = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def write_row(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "CsvResultWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class LongFormatCsvWriter:
    """Writer for long-format CSV where each inference is a separate row."""

    FIELDNAMES = [
        "run_id",
        "timestamp_utc",
        "inference_model",
        "judge_model",
        "number",
        "scenario_id",
        "scenario_text",
        "language",
        "appearance",
        "image_file",
        "question",
        "temperature",
        "top_p",
        "seed",
        "replicate_idx",
        "inference_answer",
        "C0_final_action_or_verdict_correct",
        "C1_cultural_correctness_in_thought_process",
        "C2_presence_of_cultural_reasoning",
        "D1_visual_grounding",
        "D2_overclaiming",
        "D3_politeness_nonstereotype",
        "D4_clarity_usefulness",
        "rationale0",
        "rationale1",
        "rationale2",
        "flags",
        "inference_raw",
        "judge_raw",
    ]

    def __init__(self, output_path: Path) -> None:
        self._handle = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def write_rows(
        self,
        scenario: Scenario,
        language: str,
        image_path: Path,
        question: str,
        inferences: List[InferenceResult],
        judges: List[JudgeResult],
        run_id: str,
        inference_model: str,
        judge_model: str,
        temperature: Optional[float],
        top_p: Optional[float],
        seed: Optional[int],
    ) -> None:
        """Write one row per inference result."""
        appearance = detect_appearance(image_path)
        timestamp_utc = datetime.now(timezone.utc).isoformat()

        for idx, (inference, judge) in enumerate(zip(inferences, judges), start=1):
            verdict = judge.verdict
            row = {
                "run_id": run_id,
                "timestamp_utc": timestamp_utc,
                "inference_model": inference_model,
                "judge_model": judge_model,
                "number": scenario.number,
                "scenario_id": scenario.folder_name,
                "scenario_text": scenario.scenario,
                "language": language,
                "appearance": appearance,
                "image_file": image_path.name,
                "question": question,
                "temperature": temperature if temperature is not None else "",
                "top_p": top_p if top_p is not None else "",
                "seed": seed if seed is not None else "",
                "replicate_idx": idx,
                "inference_answer": inference.text,
                "C0_final_action_or_verdict_correct": verdict.get(
                    "C0_final_action_or_verdict_correct"
                ),
                "C1_cultural_correctness_in_thought_process": verdict.get(
                    "C1_cultural_correctness_in_thought_process"
                ),
                "C2_presence_of_cultural_reasoning": verdict.get(
                    "C2_presence_of_cultural_reasoning"
                ),
                "D1_visual_grounding": verdict.get("D1_visual_grounding"),
                "D2_overclaiming": verdict.get("D2_overclaiming"),
                "D3_politeness_nonstereotype": verdict.get(
                    "D3_politeness_nonstereotype"
                ),
                "D4_clarity_usefulness": verdict.get("D4_clarity_usefulness"),
                "rationale0": verdict.get("rationale0"),
                "rationale1": verdict.get("rationale1"),
                "rationale2": verdict.get("rationale2"),
                "flags": json.dumps(verdict.get("flags", []), ensure_ascii=False),
                "inference_raw": json.dumps(inference.raw_response, ensure_ascii=False),
                "judge_raw": json.dumps(judge.raw_response, ensure_ascii=False),
            }
            self._writer.writerow(row)

        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "LongFormatCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def result_row(
    scenario: Scenario,
    language: str,
    image_path: Path,
    question: str,
    inferences: List[InferenceResult],
    judges: List[JudgeResult],
    run_id: str,
    inference_model: str,
    judge_model: str,
    temperature: Optional[float],
    top_p: Optional[float],
    seed: Optional[int],
) -> Dict[str, object]:
    appearance = detect_appearance(image_path)
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    row = {
        "run_id": run_id,
        "timestamp_utc": timestamp_utc,
        "inference_model": inference_model,
        "judge_model": judge_model,
        "number": scenario.number,
        "scenario_id": scenario.folder_name,
        "scenario_text": scenario.scenario,
        "language": language,
        "appearance": appearance,
        "image_file": image_path.name,
        "question": question,
        "temperature": temperature if temperature is not None else "",
        "top_p": top_p if top_p is not None else "",
        "seed": seed if seed is not None else "",
    }

    # Add all inference answers
    for i, inference in enumerate(inferences, start=1):
        row[f"inference_answer_{i}"] = inference.text

    # Add all judge scores (one per inference)
    for i, judge in enumerate(judges, start=1):
        verdict = judge.verdict
        row[f"judge_{i}_C0_final_action_or_verdict_correct"] = verdict.get(
            "C0_final_action_or_verdict_correct"
        )
        row[f"judge_{i}_C1_cultural_correctness_in_thought_process"] = verdict.get(
            "C1_cultural_correctness_in_thought_process"
        )
        row[f"judge_{i}_C2_presence_of_cultural_reasoning"] = verdict.get(
            "C2_presence_of_cultural_reasoning"
        )
        row[f"judge_{i}_D1_visual_grounding"] = verdict.get("D1_visual_grounding")
        row[f"judge_{i}_D2_overclaiming"] = verdict.get("D2_overclaiming")
        row[f"judge_{i}_D3_politeness_nonstereotype"] = verdict.get(
            "D3_politeness_nonstereotype"
        )
        row[f"judge_{i}_D4_clarity_usefulness"] = verdict.get("D4_clarity_usefulness")
        row[f"judge_{i}_rationale0"] = verdict.get("rationale0")
        row[f"judge_{i}_rationale1"] = verdict.get("rationale1")
        row[f"judge_{i}_rationale2"] = verdict.get("rationale2")
        row[f"judge_{i}_flags"] = json.dumps(
            verdict.get("flags", []), ensure_ascii=False
        )

    # Add all inference raw responses
    for i, inference in enumerate(inferences, start=1):
        row[f"inference_raw_{i}"] = json.dumps(
            inference.raw_response, ensure_ascii=False
        )

    # Add all judge raw responses
    for i, judge in enumerate(judges, start=1):
        row[f"judge_raw_{i}"] = json.dumps(judge.raw_response, ensure_ascii=False)

    return row


# ----------------------------- Runner ------------------------------------------

LANGUAGE_ORDER = ("TH", "EN")


def setup_logging(output_dir: Path) -> None:
    """Set up logging to both file and console."""
    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging to file: {log_file}")


def run_pipeline(args: argparse.Namespace) -> None:
    # Create results directory with timestamp
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Generate timestamped filename and run_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())
    output_csv = results_dir / f"pipeline_results_wide_{timestamp}.csv"
    output_long_csv = results_dir / f"pipeline_results_long_{timestamp}.csv"

    # Setup logging
    setup_logging(results_dir)

    logging.info("=" * 80)
    logging.info("Pipeline execution started")
    logging.info(f"Run ID: {run_id}")
    logging.info(f"Output CSV (wide): {output_csv}")
    logging.info(f"Output CSV (long): {output_long_csv}")
    logging.info("=" * 80)

    input_csv = Path(args.input_csv)
    images_dir = Path(args.images_root)
    inference_prompt_path = Path(args.inference_prompt)
    judge_prompt_path = Path(args.judge_prompt)
    rubric_json_path = Path(args.rubric_json)

    scenarios = read_scenarios(input_csv)
    logging.info(f"Loaded {len(scenarios)} scenarios from {input_csv}")

    # Load rubrics from JSON
    rubrics_by_number = load_rubrics(rubric_json_path)
    logging.info(f"Loaded {len(rubrics_by_number)} rubrics from {rubric_json_path}")

    inference_system_prompt = load_text(inference_prompt_path)
    judge_system_prompt = load_text(judge_prompt_path)
    logging.info(
        f"Loaded prompts: inference={inference_prompt_path}, judge={judge_prompt_path}"
    )

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "OpenRouter API key is required. Set OPENROUTER_API_KEY or use --api-key."
        )

    client = OpenRouterClient(
        api_key=api_key,
        base_url=args.base_url,
        referer=args.referer,
        app_title=args.app_title,
        request_timeout=args.request_timeout,
    )
    logging.info(f"OpenRouter client initialized with base_url={args.base_url}")

    retry_config = RetryConfig(
        max_attempts=args.max_retries,
        initial_delay=args.retry_initial_delay,
    )

    # Calculate total operations
    total_scenarios = len(scenarios)
    total_operations = (
        total_scenarios * 2 * 2 * args.num_inferences
    )  # scenarios * languages * images * inferences
    total_api_calls = total_operations * 2  # Each operation has 1 inference + 1 judge

    logging.info(f"Configuration:")
    logging.info(f"  - Inferences per image: {args.num_inferences}")
    logging.info(f"  - Inference model: {args.inference_model}")
    logging.info(
        f"    • Temperature: {args.temperature if args.temperature is not None else 'default'}"
    )
    logging.info(f"    • Top-p: {args.top_p if args.top_p is not None else 'default'}")
    logging.info(f"    • Seed: {args.seed if args.seed is not None else 'default'}")
    logging.info(f"  - Judge model: {args.judge_model}")
    logging.info(
        f"    • Temperature: {args.judge_temperature if args.judge_temperature is not None else 'default'}"
    )
    logging.info(
        f"    • Top-p: {args.judge_top_p if args.judge_top_p is not None else 'default'}"
    )
    logging.info(
        f"    • Seed: {args.judge_seed if args.judge_seed is not None else 'default'}"
    )
    logging.info(
        f"  - Max concurrent inferences: {args.max_concurrent_inferences or 'Sequential'}"
    )
    logging.info(
        f"  - Max concurrent judges: {args.max_concurrent_judges or 'Sequential'}"
    )
    logging.info(f"Total scenarios: {total_scenarios}")
    logging.info(
        f"Total API calls planned: {total_api_calls} ({total_operations} inferences + {total_operations} judges)"
    )
    logging.info("")

    with (
        CsvResultWriter(output_csv, args.num_inferences) as wide_writer,
        LongFormatCsvWriter(output_long_csv) as long_writer,
    ):
        completed_operations = 0
        start_time = time.time()

        for scenario_idx, scenario in enumerate(scenarios, 1):
            question_map = {"TH": scenario.qa_th, "EN": scenario.qa_en}
            scenario_images = find_scenario_images(images_dir, scenario)

            logging.info(f"\n{'=' * 80}")
            logging.info(
                f"Processing Scenario {scenario_idx}/{total_scenarios}: #{scenario.number} - {scenario.folder_name}"
            )
            logging.info(f"{'=' * 80}")

            for lang_idx, language in enumerate(LANGUAGE_ORDER, 1):
                question = question_map[language]
                logging.info(f"\n  Language {lang_idx}/2: {language}")

                for img_idx, image_path in enumerate(scenario_images, 1):
                    logging.info(
                        f"    Image {img_idx}/2: {image_path.name} (appearance: {detect_appearance(image_path)})"
                    )

                    # Call inference model multiple times (default: 20)
                    if args.max_concurrent_inferences:
                        logging.info(
                            f"      Running {args.num_inferences} inferences (concurrent: {args.max_concurrent_inferences})..."
                        )
                    else:
                        logging.info(
                            f"      Running {args.num_inferences} inferences (sequential)..."
                        )

                    inference_start = time.time()
                    inference_results = call_inference_multiple(
                        client,
                        model=args.inference_model,
                        system_prompt=inference_system_prompt,
                        question=question,
                        image_paths=[image_path],
                        max_output_tokens=args.inference_max_output_tokens,
                        retry_config=retry_config,
                        num_inferences=args.num_inferences,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                        sleep_between_inference=args.sleep_between_inference,
                        max_workers=args.max_concurrent_inferences,
                    )
                    inference_elapsed = time.time() - inference_start
                    completed_operations += args.num_inferences
                    logging.info(
                        f"      ✓ Completed {args.num_inferences} inferences in {inference_elapsed:.1f}s"
                    )

                    # Call judge once per inference (1:1 mapping)
                    rubric_payload = build_rubric_payload(scenario, rubrics_by_number)

                    if args.max_concurrent_judges:
                        logging.info(
                            f"      Running {args.num_inferences} judge evaluations (concurrent: {args.max_concurrent_judges})..."
                        )
                    else:
                        logging.info(
                            f"      Running {args.num_inferences} judge evaluations (sequential)..."
                        )

                    judge_start = time.time()
                    inference_answers = [inf.text for inf in inference_results]
                    judge_results = call_judge_batch(
                        client,
                        model=args.judge_model,
                        system_prompt=judge_system_prompt,
                        question=question,
                        answers=inference_answers,
                        rubric=rubric_payload,
                        max_output_tokens=args.judge_max_output_tokens,
                        retry_config=retry_config,
                        sleep_between_requests=args.sleep_between_requests,
                        max_workers=args.max_concurrent_judges,
                        temperature=args.judge_temperature,
                        top_p=args.judge_top_p,
                        seed=args.judge_seed,
                    )
                    judge_elapsed = time.time() - judge_start
                    completed_operations += args.num_inferences

                    # Progress update
                    progress_pct = (completed_operations / total_api_calls) * 100
                    elapsed = time.time() - start_time
                    eta = (
                        (elapsed / completed_operations * total_api_calls) - elapsed
                        if completed_operations > 0
                        else 0
                    )
                    logging.info(
                        f"      ✓ Completed {args.num_inferences} judge evaluations in {judge_elapsed:.1f}s"
                    )
                    logging.info(
                        f"      Overall progress: {progress_pct:.1f}% | ETA: {eta / 60:.1f}m"
                    )

                    # Write to both wide and long format CSVs
                    wide_writer.write_row(
                        result_row(
                            scenario,
                            language=language,
                            image_path=image_path,
                            question=question,
                            inferences=inference_results,
                            judges=judge_results,
                            run_id=run_id,
                            inference_model=args.inference_model,
                            judge_model=args.judge_model,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            seed=args.seed,
                        )
                    )

                    long_writer.write_rows(
                        scenario=scenario,
                        language=language,
                        image_path=image_path,
                        question=question,
                        inferences=inference_results,
                        judges=judge_results,
                        run_id=run_id,
                        inference_model=args.inference_model,
                        judge_model=args.judge_model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                    )

                    logging.debug(f"      ✓ Written results to both CSV formats")

        total_elapsed = time.time() - start_time
        logging.info(f"\n{'=' * 80}")
        logging.info(f"Pipeline execution completed!")
        logging.info(f"Run ID: {run_id}")
        logging.info(f"Total time: {total_elapsed / 60:.2f} minutes")
        logging.info(f"Total API calls: {completed_operations}")
        logging.info(f"Results saved to:")
        logging.info(f"  - Wide format: {output_csv}")
        logging.info(f"  - Long format: {output_long_csv}")
        logging.info(f"{'=' * 80}")


# ----------------------------- CLI ---------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scenario QA through OpenRouter inference and judge models.",
    )

    # Input files
    parser.add_argument(
        "--input-csv", default="Scenario_QA.csv", help="Path to Scenario_QA.csv"
    )
    parser.add_argument(
        "--images-root",
        default="Images",
        help="Root directory that contains scenario subfolders with images",
    )
    parser.add_argument(
        "--inference-prompt",
        default="inference_model_prompt.txt",
        help="System prompt file for the inference model",
    )
    parser.add_argument(
        "--judge-prompt",
        default="judge_prompt.txt",
        help="System prompt file for the judge model",
    )
    parser.add_argument(
        "--rubric-json",
        default="vlm_thai_social_norms_rubrics.json",
        help="Path to rubrics JSON file",
    )

    # API configuration
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (overrides OPENROUTER_API_KEY environment variable)",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="Base URL for OpenRouter API",
    )
    parser.add_argument(
        "--referer",
        help="Optional HTTP-Referer header for OpenRouter requests",
    )
    parser.add_argument(
        "--app-title",
        help="Optional X-Title header identifying your application",
    )

    # Model configuration
    parser.add_argument(
        "--inference-model",
        required=True,
        help="Model ID to use for the inference step",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="Model ID to use for the judge step",
    )
    parser.add_argument(
        "--inference-max-output-tokens",
        type=int,
        default=None,
        help="Optional cap on inference model output tokens",
    )
    parser.add_argument(
        "--judge-max-output-tokens",
        type=int,
        default=None,
        help="Optional cap on judge model output tokens",
    )

    # Sampling parameters (for stochasticity control)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for inference model (e.g., 0.7 for diversity, 0.0 for deterministic). If not set, uses model default.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus sampling) for inference model (e.g., 0.9). If not set, uses model default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for inference model (if supported by provider). Helps with reproducibility.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=None,
        help="Temperature for judge model (e.g., 0.0 for deterministic evaluation). If not set, uses model default.",
    )
    parser.add_argument(
        "--judge-top-p",
        type=float,
        default=None,
        help="Top-p (nucleus sampling) for judge model. If not set, uses model default.",
    )
    parser.add_argument(
        "--judge-seed",
        type=int,
        default=None,
        help="Random seed for judge model (if supported by provider). Helps with reproducibility.",
    )

    # Rate limiting and retry
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Sleep seconds between successive judge API calls (helps with rate limiting)",
    )
    parser.add_argument(
        "--sleep-between-inference",
        type=float,
        default=0.0,
        help="Sleep seconds between successive inference API calls (helps with rate limiting and reduces cache hits)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retry attempts for retryable API errors (e.g., 429)",
    )
    parser.add_argument(
        "--retry-initial-delay",
        type=float,
        default=1.0,
        help="Initial delay in seconds for exponential backoff on retries",
    )

    # Experiment configuration
    parser.add_argument(
        "--num-inferences",
        type=int,
        default=20,
        help="Number of inference calls to make per image (default: 20). CSV schema will adapt automatically.",
    )

    # Concurrency configuration
    parser.add_argument(
        "--max-concurrent-inferences",
        type=int,
        default=None,
        help="Maximum number of concurrent inference API calls. Default: None (sequential). Recommended: 5-10 for faster execution.",
    )
    parser.add_argument(
        "--max-concurrent-judges",
        type=int,
        default=None,
        help="Maximum number of concurrent judge API calls. Default: None (sequential). Recommended: 5-10 for faster execution.",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
