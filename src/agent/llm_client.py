"""Generate text with OpenAI when OPENAI_API_KEY is set, else local Ollama."""

from pathlib import Path
import re

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
    """Check whether OPENAI_API_KEY has a nonblank value in the settings."""
    settings = settings or get_settings()
    return bool(settings.openai_api_key and settings.openai_api_key.get_secret_value().strip())


def render_prompt(template_name: str, **values: object) -> str:
    """Fill one of the Jinja templates in the prompts folder."""
    return _prompts.get_template(template_name).render(**values).strip()


def build_answer(answer: str, kb_contexts: list[dict]) -> dict:
    """Tạo kết quả answer/citations, giữ thứ tự nguồn và loại trùng."""
    sources = {context["source"] for context in kb_contexts}
    # Prompt quy định mỗi cặp ngoặc vuông chứa đúng một nhãn nguồn.
    citations = list(dict.fromkeys(re.findall(r"\[([^\[\]\r\n]+)\]", answer)))
    unknown = [label for label in citations if label not in sources]
    if unknown:
        raise ValueError(f"Nhãn trích dẫn không có trong evidence: {unknown}")
    # Không tự thêm nguồn khi thiếu trích dẫn; groundedness sẽ đánh giá tiếp.
    return {"answer": answer, "citations": citations}


def generate(
    prompt: str, *, system_prompt: str | None = None, model_name: str | None = None,
    num_ctx: int | None = None, response_schema: dict | None = None,
) -> str:
    """Sinh câu trả lời với một client mới cho mỗi lần gọi.

    Truyền model_name để dùng Ollama cụ thể cho generation hoặc judge.
    Không truyền model_name thì vẫn chọn OpenAI/Ollama theo .env như trước.
    num_ctx và response_schema là tùy chọn cho đánh giá JSON trên Ollama.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if model_name is not None:
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name must not be blank")

    settings = get_settings()
    if model_name is None and check_api_key(settings):
        client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value().strip(),
            timeout=settings.llm_timeout_seconds,
        )
        request = {"model": settings.openai_model, "input": prompt}
        if system_prompt:
            request["instructions"] = system_prompt
        result = client.responses.create(**request).output_text
    else:
        client = ChatOllama(
            model=model_name or settings.ollama_model,
            base_url=settings.ollama_host,
            reasoning=False,
            num_ctx=num_ctx,
            format=response_schema,
            temperature=0 if response_schema else 0.2,
            num_predict=2048,
            client_kwargs={"timeout": settings.llm_timeout_seconds},
        )
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))
        response = client.invoke(messages)

        print("DONE REASONN: ", response.response_metadata.get("done_reason"))
        print("EVAL COUNT: ", response.response_metadata.get("eval_count"))

        result = response.content
        print("RESPONSE: ", result)
        # Text bị cắt vẫn đọc được nhưng chưa phải câu trả lời hoàn chỉnh.
        if response.response_metadata.get("done_reason") == "length":
            raise ValueError("Đầu ra LLM bị cắt do đạt giới hạn token.")

    if not isinstance(result, str) or not result.strip():
        raise ValueError("The model returned an empty response")
    return result.strip()
