---
name: mlx-vlm-inference
description: >
  Use this skill whenever the user is writing, debugging, or reviewing Python code that uses
  mlx-vlm (mlx_vlm) for local inference on Apple Silicon. Triggers include any mention of
  mlx_vlm, mlx-vlm, VisionFeatureCache, stream_generate, load() from mlx_vlm, Mistral3/Gemma4
  via MLX, or any question about integrating a local VLM into a pydantic-ai agent.
  Also use when the user reports MLX-VLM errors, unexpected outputs, or memory issues on Mac.
  Always consult this skill before writing any mlx_vlm code — do not rely on training memory
  for API signatures, as they evolve quickly.
---

# MLX-VLM Inference Skill

Reference: mlx-vlm v0.4.4+ (last checked April 2026, commit 3472132f).  
Target user: JB — Mac M4 Pro 48GB, Python, models: Mistral3 / Gemma4 family at 4-bit quant,
building a local-first agentic assistant with pydantic-ai.

---

## 1. Canonical Import Surface

```python
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.utils import load_config
# batch:
from mlx_vlm.generate import batch_generate
# vision cache for multi-turn:
from mlx_vlm.utils import VisionFeatureCache
# prompt formatting (always use this, never hand-roll):
from mlx_vlm.prompt_utils import apply_chat_template
```

Do **not** import from internal sub-modules (e.g. `mlx_vlm.models.gemma3`): they are unstable.

---

## 2. Standard Three-Step Workflow

```python
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config
from mlx_vlm.prompt_utils import apply_chat_template

# Step 1 — Load
model, processor = load("mlx-community/mistral-small-3.1-24b-instruct-4bit")
config = load_config("mlx-community/mistral-small-3.1-24b-instruct-4bit")

# Step 2 — Format prompt (ALWAYS use apply_chat_template, never format manually)
messages = [{"role": "user", "content": "What do you see in this image?"}]
prompt = apply_chat_template(
    processor, config, messages,
    images=["path/to/image.jpg"],   # list[str|PIL.Image] or None
    num_images=1,
)

# Step 3 — Generate
output = generate(
    model, processor,
    prompt=prompt,
    image="path/to/image.jpg",   # str, URL, or PIL.Image; omit if text-only
    max_tokens=512,
    temperature=0.0,             # 0.0 = greedy (recommended for structured tasks)
    verbose=False,
)
print(output)   # output is a str when verbose=False
```

### GenerationResult fields (stream_generate yields these)

| Field | Type | Notes |
|---|---|---|
| `text` | `str` | Cumulative generated text |
| `token` | `Optional[int]` | Last token ID |
| `prompt_tokens` | `int` | |
| `generation_tokens` | `int` | |
| `generation_tps` | `float` | Tokens/sec — use to benchmark |
| `peak_memory` | `float` | GB — watch this on 48GB machine |

---

## 3. Streaming

```python
from mlx_vlm import stream_generate

for result in stream_generate(
    model, processor,
    prompt=prompt,
    image="path/to/image.jpg",
    max_tokens=512,
    temperature=0.7,
):
    print(result.text, end="", flush=True)
print()  # newline at end
```

---

## 4. Multi-Turn Efficiency — VisionFeatureCache

In a conversational agent, **never re-encode the same image on every turn**.
The vision tower is the expensive step; `VisionFeatureCache` memoizes it keyed on image path.

```python
from mlx_vlm.utils import VisionFeatureCache

cache = VisionFeatureCache(maxsize=4)  # LRU, tune to your RAM budget

# Pass cache into generate / stream_generate:
output = generate(
    model, processor,
    prompt=prompt,
    image="path/to/image.jpg",
    vision_feature_cache=cache,
    max_tokens=256,
)
# Second call with the same image path = cache hit, vision tower skipped entirely
```

---

## 5. Model-Specific Notes

### Mistral3 / Pixtral family
- HF repo pattern: `mlx-community/mistral-small-3.1-24b-instruct-4bit` etc.
- Standard `apply_chat_template` works. No special flags needed.
- Does **not** support audio input.

### Gemma4 family
- HF repo pattern: `mlx-community/gemma-4-*-it-4bit`
- **Known limitation**: default `sliding_window=1024` caps effective context at ~10K tokens.
  For longer contexts patch `~/.../site-packages/mlx_vlm/models/gemma3/language.py`
  (increase sliding window) — see references/gemma4-context-patch.md.
- Supports `enable_thinking=True` for reasoning-style tasks.
- Does not support audio.
- `gemma-3n` variants support audio (Omni model).

### Text-only calls on a VLM
Pass `image=None` and omit images from `apply_chat_template`. Fully supported, but check
that you actually need a VLM for the task — `mlx-lm` is faster for text-only.

---

## 6. Batch Processing

```python
from mlx_vlm.generate import batch_generate

results = batch_generate(
    model, processor,
    prompts=[prompt1, prompt2],
    images=["img1.jpg", "img2.jpg"],
    max_tokens=256,
)
# returns list[str]
```

Useful for offline pipelines (e.g. batch annotation of images). Not suited for interactive
agents — use `stream_generate` there instead.

---

## 7. Pydantic-AI Integration Pattern

### Architecture overview

**Pydantic-AI has no native mlx-vlm provider.** Its built-in providers are all cloud APIs.
The `pydantic-ai-mlx` community package (pip) only supports `mlx-lm` (text-only), not mlx-vlm.

**The correct full-local architecture** — VLM as the agent's main model:

```
pydantic-ai  (tools, structured output, agent loop)
     │  OpenAI-compatible HTTP  (localhost only, zero exfil)
mlx-serve    (thin server wrapping mlx-vlm)
     │  mlx-vlm Python API
Mistral3 / Gemma4  (Metal, Apple unified memory)
```

### Step 1 — Start the local server

```bash
pip install mlx-serve[all]

# Starts an OpenAI-compatible server at localhost:8000
mlx-serve start --model mlx-community/mistral-small-3.1-24b-instruct-4bit --model-type vlm
```

`--model-type vlm` is required for vision support; omit for text-only.

### Step 2 — Point pydantic-ai at it

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
import httpx

# Zero data leaves the machine: base_url points to localhost
local_model = OpenAIChatModel(
    model_name="mistral-small-3.1-24b-instruct-4bit",  # must match --model arg
    base_url="http://localhost:8000/v1",
    api_key="not-used",                                  # required by client, ignored by server
    http_client=httpx.AsyncClient(timeout=120.0),        # increase for slow first-token
)

agent = Agent(local_model, system_prompt="Tu es un assistant local souverain.")
```

### Step 3 — Use exactly like any pydantic-ai agent

```python
from pydantic_ai import Agent, RunContext, ImageUrl
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    description: str
    objects_detected: list[str]

agent = Agent(local_model, output_type=AnalysisResult)

# Vision call — works because mlx-serve exposes multimodal endpoint
result = await agent.run(
    [
        ImageUrl(url="file:///path/to/local/image.jpg"),
        "Describe this image and list all objects.",
    ]
)
print(result.output.description)
```

### Why this is better than wrapping mlx-vlm directly

| Approach | Tool calls | Structured output | Vision | Complexity |
|---|---|---|---|---|
| mlx-serve + OpenAIChatModel | ✅ full | ✅ full | ✅ | Low |
| Custom pydantic-ai Model subclass | manual impl | manual impl | manual | Very high |
| mlx-vlm as a tool (old pattern) | n/a — VLM is a tool, not the agent | ❌ | ✅ | Medium |

The OpenAI-compatible server layer is the correct abstraction: pydantic-ai gets a real
model interface, mlx-vlm handles inference natively on Apple Silicon.

### Server management tips

```bash
mlx-serve status          # check if running
mlx-serve stop            # shut down
mlx-serve logs            # tail logs

# For Gemma4 (needs --model-type vlm):
mlx-serve start --model mlx-community/gemma-4-12b-it-4bit --model-type vlm --port 8001
```

Multiple models can run on different ports — useful for routing light tasks to smaller models.

---

## 8. Common Errors & Fixes

| Error | Likely cause | Fix |
|---|---|---|
| `KeyError: 'image'` in processor | Calling processor directly instead of `apply_chat_template` | Always use `apply_chat_template` |
| `ValueError: images must be a list` | Passing a bare string to `images=` in `apply_chat_template` | Wrap: `images=[image_path]` |
| Model loads then OOM on first generation | `prefill_step_size` too large | Add `prefill_step_size=512` to `generate()` |
| Very slow second turn in multi-turn | Vision re-encoding every turn | Use `VisionFeatureCache` (see §4) |
| Gemma context truncated unexpectedly | `sliding_window=1024` default | See §5 Gemma4 patch |
| `TypeError: generate() got unexpected keyword argument` | Old mlx-vlm version | `pip install -U mlx-vlm` |
| Empty / cut-off output | `max_tokens` too low | Increase, or check `generation_tokens` in result |
| Chinese/garbled output from Mistral | Wrong chat template applied | Ensure `apply_chat_template` uses the right `processor` |

---

## 9. Generation Parameters Quick Reference

```python
generate(
    model, processor,
    prompt=prompt,
    image=...,
    max_tokens=512,           # default 256 — usually too low
    temperature=0.0,          # 0.0 = deterministic
    top_p=1.0,                # nucleus sampling (use with temperature > 0)
    prefill_step_size=2048,   # lower to reduce peak RAM during prefill
    enable_thinking=False,    # True for Qwen3/Gemma reasoning variants
    thinking_budget=None,     # int, caps reasoning tokens
    verbose=False,            # True prints tps/memory stats to stdout
)
```

---

## 10. Do Not Do

- ❌ `processor(images=[...], text=prompt)` — bypass `apply_chat_template`
- ❌ Hard-code special tokens (`[INST]`, `<start_of_turn>`, etc.) manually
- ❌ Load model inside a loop / on every request — always use a singleton
- ❌ Use `mlx_vlm.generate` CLI as a subprocess from Python — use the Python API directly
- ❌ Use `mlx-vlm` for text-only inference in production — prefer `mlx-lm` (lighter, faster)

---

## References

- `references/gemma4-context-patch.md` — sliding window patch for long-context Gemma4
- DeepWiki: https://deepwiki.com/Blaizzy/mlx-vlm (indexed 2026-04-13, commit 3472132f)
- PyPI: https://pypi.org/project/mlx-vlm/ (v0.4.4)
