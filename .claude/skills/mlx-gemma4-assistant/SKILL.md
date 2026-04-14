---
name: mlx-gemma4-assistant
description: >
  Complete integration guide for running mlx-openai-server with a locally
  stored Gemma 4 model and wiring it to a pydantic-ai agent on Apple Silicon.
  Use this skill whenever the user wants to: start mlx-openai-server with a
  local Gemma 4 model file (which is a multimodal model), launch it as a
  subprocess inside a FastAPI lifespan, connect pydantic-ai to the local
  server endpoint, understand how SSE streaming chunks differ for reasoning
  vs content vs tool calls, access reasoning_content from Gemma 4 responses,
  use Gemma 4's vision/image capabilities, or debug Gemma 4 specific
  behaviour. Also trigger on `--reasoning-parser gemma4`,
  `--tool-call-parser gemma4`, `OpenAIModel` with localhost base URL, or any
  combination of mlx-openai-server + pydantic-ai.
---

# mlx-openai-server × Gemma 4 × pydantic-ai — Integration Skill

Covers the exact configuration to run mlx-openai-server with a **local**
Gemma 4 model and use it as the backend for a pydantic-ai agent. All details
are derived from the server source code.

> **Important:** Gemma 4 is a **multimodal** model (text + vision). Use
> `--model-type multimodal` so the server loads it through `mlx-vlm`, which
> enables image inputs and correct chat-template handling.

---

## 1. Launching the server — exact CLI flags

```bash
mlx-openai-server launch \
  --model-path /path/to/models/gemma-4-27b-it-4bit \
  --model-type multimodal \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --enable-auto-tool-choice \
  --served-model-name gemma4 \
  --host 127.0.0.1 \
  --port 8000 \
  --queue-timeout 300
```

**Key flags explained:**

| Flag | Why it matters |
|------|---------------|
| `--model-path /path/to/…` | Absolute local path to the MLX model directory |
| `--model-type multimodal` | Gemma 4 is multimodal; loads via `mlx-vlm` for image + text support |
| `--reasoning-parser gemma4` | Strips `<|channel>thought…<channel|>` and routes to `reasoning_content` |
| `--tool-call-parser gemma4` | Parses Gemma 4's `<|tool_call>call:func{args}<tool_call|>` blocks |
| `--enable-auto-tool-choice` | Required to activate tool routing |
| `--served-model-name gemma4` | Short alias; use this name in pydantic-ai (see §3) |
| `--host 127.0.0.1` | Loopback only — never `0.0.0.0` for a local assistant |

Optionally useful:
- `--context-length 32768` — cap context to avoid Metal paging on long sessions
- `--temperature 1.0` — good server-side default for Gemma 4 with reasoning (see §6)
- `--debug` — logs raw model output *before* parsing; invaluable when tool calls misbehave

---

## 2. Subprocess integration inside a FastAPI lifespan

Start the server as a child process, poll `/health`, kill on shutdown.
Do **not** import `MLXVLMHandler` directly into the parent process — loading
the model in the same Metal context causes GPU semaphore leaks (the reason
the server itself uses `multiprocessing spawn` internally).

```python
import asyncio, subprocess, httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI

MLX_SERVER_CMD = [
    "mlx-openai-server", "launch",
    "--model-path",        "/path/to/models/gemma-4-27b-it-4bit",
    "--model-type",        "multimodal",
    "--reasoning-parser",  "gemma4",
    "--tool-call-parser",  "gemma4",
    "--enable-auto-tool-choice",
    "--served-model-name", "gemma4",
    "--host",              "127.0.0.1",
    "--port",              "8000",
    "--no-log-file",
]


async def _wait_until_ready(url: str, timeout: float = 180.0) -> None:
    """Poll /health until the model is loaded."""
    deadline = asyncio.get_event_loop().time() + timeout
    async with httpx.AsyncClient() as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                r = await client.get(url, timeout=2.0)
                if r.status_code == 200:
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(1.5)
    raise TimeoutError(f"mlx-openai-server not ready after {timeout:.0f}s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    proc = subprocess.Popen(MLX_SERVER_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    app.state.mlx_proc = proc
    try:
        await _wait_until_ready("http://127.0.0.1:8000/health")
    except TimeoutError:
        proc.terminate()
        raise

    yield  # ── application runs here ──

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
```

> **Load time:** a 27B 4-bit model on an M4 Pro (48 GB) takes 60–90 s.
> 180 s timeout is safe; tighten it once you know your hardware baseline.

---

## 3. Connecting pydantic-ai to the local server

pydantic-ai uses the `openai` provider under the hood. Point it at the local
endpoint with a custom `AsyncOpenAI` client:

```python
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

_local_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="not-needed",          # server ignores the key
)

# Model name MUST match --served-model-name (or --model-path if omitted)
local_model = OpenAIModel("gemma4", openai_client=_local_client)

agent = Agent(
    model=local_model,
    instructions="You are a helpful local assistant.",
)
```

### Tools — nothing special for Gemma 4

pydantic-ai sends tools as the standard OpenAI schema; the server's
`--tool-call-parser gemma4` handles the model-side conversion transparently.
Declare tools normally and pydantic-ai manages the full call/result loop:

```python
from pydantic_ai import Agent

@agent.tool_plain
def read_file(path: str) -> str:
    """Read a local file and return its contents."""
    with open(path) as f:
        return f.read()
```

### Streaming

```python
async def stream_response(user_message: str) -> None:
    async with agent.run_stream(user_message) as stream:
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end="", flush=True)
    print()
```

### Full result with usage

```python
result = await agent.run("Explain how MLX uses unified memory.")
print(result.output)
print(result.usage())   # RunUsage(input_tokens=…, output_tokens=…)
```

---

## 4. Accessing `reasoning_content`

Gemma 4 wraps chain-of-thought in `<|channel>thought … <channel|>`.
The server parser strips these tokens and makes the reasoning available as
structured data — it is **not** limited to server logs.

### Via the Responses API (`/v1/responses`) — recommended

The Responses API returns reasoning as a `ResponseReasoningItem` in the
`output` list, clearly separated from the final answer:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

response = client.responses.create(
    model="gemma4",
    input="Solve step by step: what is 17 × 23?",
)

for item in response.output:
    if item.type == "reasoning":
        print("REASONING:", item.summary[0].text)
    elif item.type == "message":
        for part in item.content:
            if hasattr(part, "text"):
                print("ANSWER:", part.text)
```

For **streaming** with the Responses API, reasoning arrives as
`response.reasoning_summary_text.delta` SSE events, before the answer:

```python
response = client.responses.create(
    model="gemma4",
    input="Solve step by step: what is 17 × 23?",
    stream=True,
)

for event in response:
    if event.type == "response.reasoning_summary_text.delta":
        print(event.delta, end="", flush=True)   # reasoning tokens
    elif event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)   # answer tokens
```

The full SSE sequence for a reasoning response is:
`response.created` → `response.in_progress` →
`response.output_item.added` (reasoning item) →
`response.reasoning_summary_text.delta` × N →
`response.reasoning_summary_text.done` →
`response.output_item.done` (reasoning) →
`response.output_item.added` (message) →
`response.output_text.delta` × N →
`response.output_text.done` → … →
`response.completed`

### Via the Chat Completions API (`/v1/chat/completions`)

Reasoning is exposed as a **non-standard** `reasoning_content` field on
the message/delta objects. It is returned in real-time in the SSE stream:

```python
# Non-streaming
response = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "Think carefully: what is 17 × 23?"}],
)
choice = response.choices[0].message
reasoning = getattr(choice, "reasoning_content", None)
print("REASONING:", reasoning)
print("ANSWER:", choice.content)

# Streaming — reasoning_content arrives first, then content
stream = client.chat.completions.create(
    model="gemma4",
    messages=[{"role": "user", "content": "Think carefully: what is 17 × 23?"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    rc = getattr(delta, "reasoning_content", None)
    if rc:
        print(rc, end="", flush=True)   # reasoning token
    if delta.content:
        print(delta.content, end="", flush=True)   # answer token
```

### pydantic-ai and reasoning

pydantic-ai reads only `delta.content`, so reasoning tokens are silently
discarded when routing through a pydantic-ai agent. This is expected and
correct — the agent sees only the final answer. To capture reasoning
alongside an agent run, consume the raw OpenAI client directly or use the
Responses API in a separate call.

---

## 5. Vision — using Gemma 4's multimodal capabilities

Because `--model-type multimodal` loads the model via `mlx-vlm`, you can pass
images alongside text. Use the standard OpenAI image format:

```python
import base64

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="gemma4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe what you see in this image."},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image('/path/to/image.jpg')}"
                },
            },
        ],
    }],
)
print(response.choices[0].message.content)
```

You can also pass a public URL instead of base64:

```python
{"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
```

The server's `ImageProcessor` resizes to 448 px on the longest edge by default
and caches processed images. Use `--disable-auto-resize` on the server if
you need full resolution.

### Vision via the Responses API

```python
response = client.responses.create(
    model="gemma4",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What is in this image?"},
            {"type": "input_image", "image_url": "https://example.com/photo.jpg"},
        ],
    }],
)
```

---

## 6. Gemma 4 quirks and limits

### Reasoning and tool calls are mutually exclusive per turn
The model emits a reasoning block OR tool call blocks in a single turn, never
both simultaneously. pydantic-ai handles multi-turn tool loops transparently.

### Temperature — stay above 0.7 with reasoning enabled
Gemma 4 degrades visibly below 0.7 when reasoning is active. Use `0.9–1.0`
for agentic loops. Use `0.3–0.6` only for pure extraction tasks where you
want low variance. Set it via the server default (`--temperature 1.0`) or
per-request:

```python
from pydantic_ai.settings import ModelSettings

result = await agent.run(
    "Summarise this document.",
    model_settings=ModelSettings(temperature=0.95),
)
```

### Custom argument serialisation (transparent)
Gemma 4 uses its own value format internally (`<|"|>string<|"|>`, bare
numbers, nested `{key:value}`). The `gemma4` parser converts this to standard
JSON before the response reaches pydantic-ai. `--debug` will expose it in
server logs if a tool call is misbehaving.

### Context budget on Apple Silicon
At 4-bit quantisation a 27B model uses ~16 GB of GPU RAM. On an M4 Pro with
48 GB, set `--context-length 32768` explicitly. Without it the model default
may push the system into Metal paging mid-session, causing a sharp drop in
tokens-per-second.

### No speculative decoding
`--draft-model-path` does not work with the `gemma4` parser. No Gemma 4 draft
model is available on mlx-community as of mid-2025.

---

## 7. Quick-reference checklist

- [ ] `mlx-openai-server` installed in the active virtualenv
- [ ] Model directory exists at the path given to `--model-path`
- [ ] `--model-type multimodal` (not `lm`) — Gemma 4 is a multimodal model
- [ ] Both `--reasoning-parser gemma4` AND `--tool-call-parser gemma4` set
- [ ] `--enable-auto-tool-choice` present when the agent has tools
- [ ] `--served-model-name gemma4` matches `OpenAIModel("gemma4", …)`
- [ ] `/health` returns `{"status":"ok"}` before the agent makes its first call
- [ ] `--context-length 32768` set if sessions are long or documents are large
- [ ] Use `/v1/responses` to get structured reasoning output (not just logs)
