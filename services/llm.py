import json
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def call_llm(prompt: str, temperature: float = 0.2) -> str:
    from groq import Groq

    # Read fresh from env every call so Streamlit sidebar key works
    api_key = os.environ.get("GROQ_API_KEY", "")
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Enter it in the sidebar or add to .env\n"
            "Get a free key at: https://console.groq.com"
        )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    if match:
        fixed_block = re.sub(r",\s*([}\]])", r"\1", match.group())
        try:
            return json.loads(fixed_block)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM output:\n{raw[:600]}")


def call_llm_json(prompt: str, temperature: float = 0.1) -> dict:
    full_prompt = (
        prompt
        + "\n\nCRITICAL: Your response must be ONLY a valid JSON object. "
        "Do not include any text before or after the JSON. "
        "Do not use markdown code fences. Do not add comments. "
        "Start your response with { and end with }."
    )
    raw = call_llm(full_prompt, temperature=temperature)
    return _extract_json(raw)