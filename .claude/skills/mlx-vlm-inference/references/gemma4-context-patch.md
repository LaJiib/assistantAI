# Gemma4 / Gemma3 Long-Context Patch

## Problem
`mlx_vlm`'s Gemma3/Gemma4 language model implementation ships with
`sliding_window=1024` by default, which limits effective context to ~10K tokens
on Apple Silicon (Metal GPU timeout at higher context).

## Patch (mlx-vlm ≤ 0.4.4)

File: `~/.../site-packages/mlx_vlm/models/gemma3/language.py`

Find the `TextConfig` dataclass and change:
```python
# Before
sliding_window: int = 1024

# After — up to ~50K usable context on M4 Pro 48GB
sliding_window: int = 4096   # or higher; test stability
```

Or override at load time via config dict (if supported in your version):
```python
model, processor = load(
    "mlx-community/gemma-4-12b-it-4bit",
    # config overrides not yet officially supported in utils.load()
    # patch the file directly for now
)
```

## Note on gemma-3n (Omni)
`gemma-3n` variants have a different architecture (MobileNet-style soft token routing)
and different config structure. The sliding window patch location may differ.
Check `mlx_vlm/models/gemma3n/` if it exists in your install.

## Source
Documented in https://github.com/waybarrios/vllm-mlx (January 2026).
