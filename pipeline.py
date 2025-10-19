"""Pipeline to run scenario QA through OpenRouter inference and judge models."""
from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import random
import sys
import time
from dataclasses import dataclass
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
            img_no_metadata.convert("RGBA" if img_no_metadata.mode == "P" else "RGB").save(
                output, format="PNG", optimize=True
            )
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
        raise SystemExit(f"Image folder not found for scenario {scenario.number}: {folder}")
    candidates = sorted(folder.glob(f"{scenario.number:02d}_*.png"))
    if len(candidates) < 2:
        raise SystemExit(
            f"Expected at least two images in {folder}, found {len(candidates)}."
        )
    # Use the first two sorted images per requirement ScenarioNumber_01 + ScenarioNumber_02
    return candidates[:2]


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
            code = body.get("error", {}).get("code") if isinstance(body.get("error"), dict) else body.get("code")
            if isinstance(code, int) and code in RETRYABLE_STATUS_CODES:
                return True
            if isinstance(code, str) and code.isdigit() and int(code) in RETRYABLE_STATUS_CODES:
                return True
    # String fallback
    message = str(error)
    if any(token in message for token in ("429", "rate limit", "temporarily rate-limited")):
        return True
    return False


def request_with_retry(callable_fn: Callable[[], T], retry_config: RetryConfig, label: str) -> T:
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
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=request_timeout)
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
        completion = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
        if hasattr(completion, "model_dump"):
            return completion.model_dump()
        if isinstance(completion, dict):
            return completion
        if isinstance(completion, str):
            try:
                return json.loads(completion)
            except json.JSONDecodeError as exc:  # pragma: no cover - unexpected shape
                raise ValueError(f"Unexpected string response: {completion[:200]}") from exc
        raise TypeError(
            f"Unsupported response type from OpenRouter client: {type(completion)!r}"
        )


def build_messages(
    system_prompt: str, user_content: object
) -> List[Dict[str, object]]:
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
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            # Drop opening fence (e.g., ```json)
            lines = lines[1:]
        # Remove closing fence if present
        while lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    if not cleaned:
        raise ValueError("Empty JSON response")
    return json.loads(cleaned)


def call_inference(
    client: OpenRouterClient,
    *,
    model: str,
    system_prompt: str,
    question: str,
    image_paths: Optional[Iterable[Path]] = None,
    max_output_tokens: Optional[int] = None,
    retry_config: RetryConfig,
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
    response_payload = request_with_retry(
        lambda: client.chat_completions_create(model=model, messages=messages, **kwargs),
        retry_config=retry_config,
        label="inference",
    )
    answer_text = extract_output_text(response_payload)
    return InferenceResult(text=answer_text, raw_response=response_payload)


def build_rubric_payload(scenario: Scenario) -> Dict[str, object]:
    expected_actions = [item.strip() for item in scenario.expected_en.split(",") if item.strip()]
    acceptable_variants = expected_actions  # placeholder fallback
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
    response_payload = request_with_retry(
        lambda: client.chat_completions_create(model=model, messages=messages, **kwargs),
        retry_config=retry_config,
        label="judge",
    )
    output_text = extract_output_text(response_payload)
    try:
        verdict = parse_json_response(output_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "Judge model did not return valid JSON. Received:\n"
            f"{output_text}"
        ) from exc
    return JudgeResult(verdict=verdict, raw_response=response_payload)


# ----------------------------- CSV persistence ---------------------------------

RESULT_HEADER = [
    "number",
    "scenario_id",
    "language",
    "image_file",
    "question",
    "inference_answer",
    "judge_C0_final_action_or_verdict_correct",
    "judge_C1_cultural_correctness_in_thought_process",
    "judge_C2_presence_of_cultural_reasoning",
    "judge_D1_visual_grounding",
    "judge_D2_overclaiming",
    "judge_D3_politeness_nonstereotype",
    "judge_D4_clarity_usefulness",
    "judge_rationale0",
    "judge_rationale1",
    "judge_rationale2",
    "judge_flags",
    "inference_raw",
    "judge_raw",
]


class CsvResultWriter:
    def __init__(self, output_path: Path) -> None:
        self._handle = output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=RESULT_HEADER)
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


def result_row(
    scenario: Scenario,
    language: str,
    image_file: str,
    question: str,
    inference: InferenceResult,
    judge: JudgeResult,
) -> Dict[str, object]:
    verdict = judge.verdict
    return {
        "number": scenario.number,
        "scenario_id": scenario.folder_name,
        "language": language,
        "image_file": image_file,
        "question": question,
        "inference_answer": inference.text,
        "judge_C0_final_action_or_verdict_correct": verdict.get("C0_final_action_or_verdict_correct"),
        "judge_C1_cultural_correctness_in_thought_process": verdict.get(
            "C1_cultural_correctness_in_thought_process"
        ),
        "judge_C2_presence_of_cultural_reasoning": verdict.get(
            "C2_presence_of_cultural_reasoning"
        ),
        "judge_D1_visual_grounding": verdict.get("D1_visual_grounding"),
        "judge_D2_overclaiming": verdict.get("D2_overclaiming"),
        "judge_D3_politeness_nonstereotype": verdict.get("D3_politeness_nonstereotype"),
        "judge_D4_clarity_usefulness": verdict.get("D4_clarity_usefulness"),
        "judge_rationale0": verdict.get("rationale0"),
        "judge_rationale1": verdict.get("rationale1"),
        "judge_rationale2": verdict.get("rationale2"),
        "judge_flags": json.dumps(verdict.get("flags", []), ensure_ascii=False),
        "inference_raw": json.dumps(inference.raw_response, ensure_ascii=False),
        "judge_raw": json.dumps(judge.raw_response, ensure_ascii=False),
}

# ----------------------------- Runner ------------------------------------------

LANGUAGE_ORDER = ("TH", "EN")


def run_pipeline(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    images_dir = Path(args.images_root)
    inference_prompt_path = Path(args.inference_prompt)
    judge_prompt_path = Path(args.judge_prompt)

    scenarios = read_scenarios(input_csv)
    inference_system_prompt = load_text(inference_prompt_path)
    judge_system_prompt = load_text(judge_prompt_path)

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

    retry_config = RetryConfig(
        max_attempts=args.max_retries,
        initial_delay=args.retry_initial_delay,
    )

    rubric_cache: Dict[int, Dict[str, object]] = {}

    with CsvResultWriter(Path(args.output_csv)) as writer:
        for scenario in scenarios:
            rubric_cache[scenario.number] = build_rubric_payload(scenario)
            question_map = {"TH": scenario.qa_th, "EN": scenario.qa_en}
            scenario_images = find_scenario_images(images_dir, scenario)
            for language in LANGUAGE_ORDER:
                question = question_map[language]
                for image_path in scenario_images:
                    inference_result = call_inference(
                        client,
                        model=args.inference_model,
                        system_prompt=inference_system_prompt,
                        question=question,
                        image_paths=[image_path],
                        max_output_tokens=args.inference_max_output_tokens,
                        retry_config=retry_config,
                    )
                    rubric_payload = rubric_cache[scenario.number]
                    judge_result = call_judge(
                        client,
                        model=args.judge_model,
                        system_prompt=judge_system_prompt,
                        question=question,
                        answer=inference_result.text,
                        rubric=rubric_payload,
                        max_output_tokens=args.judge_max_output_tokens,
                        retry_config=retry_config,
                    )
                    writer.write_row(
                        result_row(
                            scenario,
                            language=language,
                            image_file=image_path.name,
                            question=question,
                            inference=inference_result,
                            judge=judge_result,
                        )
                    )
                    if args.sleep_between_requests:
                        time.sleep(args.sleep_between_requests)


# ----------------------------- CLI ---------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run scenario QA through OpenRouter inference and judge models.",
    )
    parser.add_argument("--input-csv", default="Scenario_QA.csv", help="Path to Scenario_QA.csv")
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
        "--output-csv",
        default="pipeline_results.csv",
        help="Destination CSV file for aggregated outputs",
    )
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
        help="Sleep seconds between successive API calls (helps with rate limiting)",
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
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
