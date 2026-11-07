"""Render prompt và gọi OpenRouter hoặc Ollama."""

import json
from pathlib import Path
import warnings

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from langchain_ollama import ChatOllama
from openai import OpenAI

from src.core.config import Settings, get_settings


_prompts = Environment(
    loader=FileSystemLoader(Path(__file__).parent / "prompts"),
    undefined=StrictUndefined,
    autoescape=False,
)
_prompts.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}


def check_api_key(settings: Settings | None = None) -> bool:
    """Kiểm tra OPENROUTER_API_KEY hợp lệ cho Cloud model."""
    settings = settings or get_settings()
    key = getattr(settings, "openrouter_api_key", None)
    return bool(key and key.get_secret_value().strip())


check_openrouter_key = check_api_key


def render_prompt(template_name: str, **values: object) -> str:
    return _prompts.get_template(template_name).render(**values).strip()


class GenerationError(ValueError):
    """Lỗi generation vẫn giữ phần answer model đã sinh."""

    def __init__(self, message: str, answer: str):
        super().__init__(message)
        self.generation = {"answer": answer}


def build_answer(answer: str, kb_contexts: list[dict]) -> dict:
    """Gắn metadata nguồn backend vào answer và loại parent trùng."""
    sources = {}
    for context in kb_contexts:
        parent_id = context.get("parent_chunk_id")
        title = context.get("source")
        valid_id = isinstance(parent_id, str) and parent_id.strip()
        valid_title = isinstance(title, str) and title.strip()
        if not valid_id or not valid_title:
            raise ValueError("Source metadata requires nonblank parent_chunk_id and source")
        if parent_id in sources and sources[parent_id]["title"] != title:
            raise ValueError("Conflicting source titles for the same parent_chunk_id")
        sources[parent_id] = {"parent_chunk_id": parent_id, "title": title}
    return {"answer": answer, "sources": list(sources.values())}


def _validate_request(
    prompt: str,
    system_prompt: str | None,
    response_schema: dict | None,
    num_ctx: int | None,
    max_tokens: int,
) -> None:
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if num_ctx is not None:
        input_size = len(
            ((system_prompt or "") + prompt + json.dumps(response_schema or {})).encode("utf-8")
        )
        if num_ctx <= input_size + max_tokens + 256:
            raise ValueError("Context có thể không đủ; tăng num_ctx để giữ nguyên evidence.")


def _openrouter_generate(
    prompt: str,
    settings,
    model_name: str | None,
    system_prompt: str | None,
    response_schema: dict | None,
    seed: int | None,
    max_tokens: int,
    diagnostics: dict | None,
) -> str:
    key = getattr(settings, "openrouter_api_key", None)
    if not key or not key.get_secret_value().strip():
        raise ValueError("OPENROUTER_API_KEY is not configured")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    model = model_name or getattr(settings, "openrouter_model", "mistralai/mistral-small-24b-instruct-2501")
    base_url = getattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")

    client = OpenAI(
        base_url=base_url,
        api_key=key.get_secret_value().strip(),
        timeout=settings.llm_timeout_seconds,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Agent IT Helpdesk",
        },
    )

    request: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0 if response_schema else 0.2,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        request["seed"] = seed
    if response_schema:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "groundedness",
                "schema": response_schema,
            },
        }

    try:
        completion = client.chat.completions.create(**request)
    except Exception as exc:
        raise ValueError(f"OpenRouter API error: {exc}") from exc

    choice = completion.choices[0]
    result = choice.message.content or ""
    usage = getattr(completion, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None

    if diagnostics is not None:
        diagnostics.update(
            raw_response=result,
            finish_reason=choice.finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    print("=" * 50)
    print(f"[OpenRouter (Cloud) - {model}]")
    print("FINISH REASON: ", choice.finish_reason)
    print("INPUT TOKENS: ", input_tokens if input_tokens is not None else "N/A")
    print("OUTPUT TOKENS: ", output_tokens if output_tokens is not None else "N/A")

    if choice.finish_reason != "stop":
        partial_answer = result if isinstance(result, str) else ""
        raise GenerationError(f"OpenRouter response incomplete (finish_reason: {choice.finish_reason})", partial_answer)

    return result


def _ollama_generate(
    prompt: str,
    settings,
    model_name: str | None,
    system_prompt: str | None,
    num_ctx: int | None,
    response_schema: dict | None,
    seed: int | None,
    max_tokens: int,
    diagnostics: dict | None,
):
    client = ChatOllama(
        model=model_name or settings.ollama_model,
        base_url=settings.ollama_host,
        reasoning=False,
        num_ctx=num_ctx,
        format=response_schema,
        temperature=0 if response_schema else 0.2,
        num_predict=max_tokens,
        seed=seed,
        client_kwargs={"timeout": settings.llm_timeout_seconds},
    )
    messages = []
    if system_prompt:
        messages.append(("system", system_prompt))
    messages.append(("human", prompt))
    response = client.invoke(messages)
    metadata = response.response_metadata

    print("=" * 50)
    print(f"[Ollama (Local) - {model_name or settings.ollama_model}]")
    print("DONE REASON: ", metadata.get("done_reason"))
    print("INPUT TOKENS: ", metadata.get("prompt_eval_count", "N/A"))
    print("OUTPUT TOKENS: ", metadata.get("eval_count"))

    if diagnostics is not None:
        diagnostics.update(
            raw_response=response.content,
            finish_reason=metadata.get("done_reason"),
            input_tokens=metadata.get("prompt_eval_count"),
            output_tokens=metadata.get("eval_count"),
        )
    if metadata.get("done_reason") == "length":
        partial_answer = response.content if isinstance(response.content, str) else ""
        raise GenerationError("Đầu ra LLM bị cắt do đạt giới hạn token.", partial_answer)
    return response.content


def generate(
    prompt: str, *, system_prompt: str | None = None, model_name: str | None = None,
    num_ctx: int | None = None, response_schema: dict | None = None,
    seed: int | None = None, max_tokens: int = 2048, use_cloud_model: bool = False,
    diagnostics: dict | None = None,
) -> str:
    """Sinh text bằng LLM.

    Mặc định chạy qua Ollama (local model).
    Nếu `use_cloud_model=True`:
      - Nếu có OPENROUTER_API_KEY trong .env: chạy Cloud model (OpenRouter).
      - Nếu không có API key trong .env: hiện cảnh báo và fallback chạy Ollama local model.
    """
    if diagnostics is not None:
        diagnostics.clear()

    _validate_request(prompt, system_prompt, response_schema, num_ctx, max_tokens)

    if model_name is not None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be blank")

    settings = get_settings()

    run_cloud = False
    if use_cloud_model:
        if check_api_key(settings):
            run_cloud = True
        else:
            warning_msg = (
                "use_cloud_model=True nhưng OPENROUTER_API_KEY chưa được cấu hình trong .env. "
                "Fallback sang Ollama model."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            print(f"CẢNH BÁO: {warning_msg}")

    if run_cloud:
        model = model_name
        if not model or "qwen" in model.lower():
            model = getattr(settings, "openrouter_model", "mistralai/mistral-small-24b-instruct-2501")
        result = _openrouter_generate(
            prompt, settings, model, system_prompt, response_schema,
            seed, max_tokens, diagnostics,
        )
    else:
        result = _ollama_generate(
            prompt, settings, model_name, system_prompt, num_ctx,
            response_schema, seed, max_tokens, diagnostics,
        )

    if not isinstance(result, str) or not result.strip():
        raise ValueError("The model returned an empty response")
    return result.strip()
