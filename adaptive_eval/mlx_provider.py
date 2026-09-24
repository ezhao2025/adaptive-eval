"""Local vision models through MLX (Apple silicon), behind the same Provider interface.

Same contract as AnthropicProvider: `await answer(model, item_id) -> Response`. The cache,
event log, IRT fit and reports all work unchanged; only the transport differs.

Differences from an API provider, both of which the harness has to respect:
  * ONE MODEL PER PROCESS. Weights sit in memory, so a worker serves a single model and the
    scheduler is pointed at that one model (--models <name>). Swapping means restarting.
  * NO RATE LIMIT and NO COST. Calls are serialised on the GPU, so concurrency is 1 and every
    Response carries cost 0.

Inference blocks, so it runs in a thread: otherwise one call would freeze the worker's event
loop and its Redis heartbeats.

    pip install mlx-vlm
    python -m adaptive_eval.b.worker --id w1 --provider mlx \\
        --mlx-model mlx-community/Qwen2.5-VL-3B-Instruct-4bit --items data/cal_items.json
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import tempfile

from .providers import ProviderConfig, Response
from .real_provider import grade

# No limits to respect locally; the numbers only have to be large enough not to throttle.
MLX_CONFIG = ProviderConfig(rpm=100_000, tpm=100_000_000, latency=(0.0, 0.0), failure_rate=0.0,
                            usd_per_1k_input=0.0, usd_per_1k_output=0.0, est_tokens_per_call=1)


class MLXProvider:
    def __init__(self, cfg: ProviderConfig, items: dict[str, dict], model_id: str, *,
                 max_tokens: int = 512, name: str = "mlx", image_dir: str | None = None):
        self.name, self.cfg, self.items = name, cfg, items
        self.model_id, self.max_tokens = model_id, max_tokens
        self.calls = 0
        self._image_dir = image_dir or tempfile.mkdtemp(prefix="spatial-items-")
        self._loaded = None                      # (model, processor, config), loaded lazily

    # ---- model + images ------------------------------------------------------------
    def _load(self):
        if self._loaded is None:
            from mlx_vlm import load                       # pip install mlx-vlm
            from mlx_vlm.utils import load_config
            model, processor = load(self.model_id)
            self._loaded = (model, processor, load_config(self.model_id))
        return self._loaded

    def _image_path(self, item_id: str, item: dict) -> str:
        """mlx-vlm takes image paths, so each item's PNG is written once and reused."""
        name = hashlib.sha256(item_id.encode()).hexdigest()[:16] + ".png"
        path = os.path.join(self._image_dir, name)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(base64.b64decode(item["image_b64"]))
        return path

    # ---- one call ------------------------------------------------------------------
    def _generate(self, item: dict, image_path: str) -> tuple[str, int, int]:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, processor, config = self._load()
        prompt = f"{item['question']}\nAnswer with one word, nothing else."
        formatted = apply_chat_template(processor, config, prompt, num_images=1)
        out = generate(model, processor, formatted, [image_path],
                       max_tokens=self.max_tokens, verbose=False)
        # mlx-vlm returns a plain string in older versions, a result object in newer ones
        text = getattr(out, "text", out if isinstance(out, str) else str(out))
        usage = (getattr(out, "prompt_tokens", 0) or 0, getattr(out, "generation_tokens", 0) or 0)
        return text, usage[0], usage[1]

    async def answer(self, model: str, item_id: str) -> Response:
        item = self.items[item_id]               # a missing item is our bug: let it raise
        self.calls += 1
        text, tin, tout = await asyncio.to_thread(self._generate, item,
                                                  self._image_path(item_id, item))
        return Response(grade(text, item), tin, tout, 0.0)
