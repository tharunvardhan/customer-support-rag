import requests

from . import config

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant. Answer the customer's question using ONLY "
    "the numbered context passages provided. If the answer is not in the context, reply only: "
    "\"I couldn't find that in the support documents.\" Do not suggest contacting anyone. "
    "Be clear and concise, use short steps when explaining a process, and mention the passage "
    "numbers you relied on, like [1] or [2]. Never invent policies, prices or contact details."
)

def build_prompt(question, chunks):
    blocks = [
        f"[{i}] Source: {c['source']}, page {c['page']}\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

def _ollama(messages):
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    try:
        r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not reach Ollama at{config.OLLAMA_URL}. Is it running? "
            f"Start it, then run: ollama pull {config.OLLAMA_MODEL} "
        )
    return r.json()["message"]["content"].strip()

def _anthropic(messages):
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "Anthropic API key not set. Please set the ANTHROPIC_API_KEY environment variable."
        )
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()

def generate(question, chunks, history=None):
    history = list(history or [])[-6:]
    while history and history[0]["role"] != "user":
        history.pop(0)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": build_prompt(question, chunks)})

    if config.LLM_BACKEND == "anthropic":
        return _anthropic(messages)
    if config.LLM_BACKEND == "ollama":
        return _ollama(messages)
    raise RuntimeError(
        f"Unknown LLM_Backend '{config.LLM_BACKEND}'. Use 'ollama' or 'anthropic'."
    )