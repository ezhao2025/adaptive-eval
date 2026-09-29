"""Run open VLMs over the spatial items on a rented GPU. Self-contained: copy this file and
the items JSON to the box, nothing else from the repo is needed.

    pip install vllm
    python vllm_batch.py --items cal_items.json --models Qwen/Qwen2.5-VL-7B-Instruct \\
        --limit 3 --out responses.jsonl          # smoke test first
    python vllm_batch.py --items cal_items.json --models-file models.txt --out responses.jsonl

Writes one JSONL line per (model, item): the raw reply, not a score. Grading happens back in
the repo so every model is graded by the same code, and so a grader fix never means renting
the GPU again.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time


def load_items(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def prompt_of(item: dict) -> str:
    return f"{item['question']}\nAnswer with one word, nothing else."


def run_model(model_id: str, items: dict, ids: list[str], out, max_tokens: int,
              max_model_len: int, gpu_frac: float, legacy_prompt: bool = False) -> None:
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(model=model_id, max_model_len=max_model_len, gpu_memory_utilization=gpu_frac,
              limit_mm_per_prompt={"image": 1}, trust_remote_code=True)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    if legacy_prompt:
        outputs = llm.generate(_legacy_requests(llm, items, ids), params)
    else:
        # llm.chat applies the model's own chat template, including templates that live in
        # the processor rather than the tokenizer (LLaVA-OneVision, LLaVA-1.5, ...). Building
        # the prompt from the tokenizer alone silently fell back to a bare prompt for those
        # models, and they answered with an immediate end-of-turn: empty text.
        convs = []
        for iid in ids:
            item = items[iid]
            url = f"data:{item.get('media_type', 'image/png')};base64,{item['image_b64']}"
            convs.append([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": prompt_of(item)}]}])
        outputs = llm.chat(convs, params)

    empty = 0
    for iid, o in zip(ids, outputs):
        text = o.outputs[0].text
        empty += not text.strip()
        out.write(json.dumps({"model": model_id, "item_id": iid, "text": text,
                              "output_tokens": len(o.outputs[0].token_ids)}) + "\n")
    out.flush()
    print(f"[{model_id}] {len(ids)} items in {time.time() - t0:.0f}s, {empty} empty replies",
          flush=True)
    if empty > len(ids) // 2:
        print(f"[{model_id}] WARNING: most replies are empty -- check the prompt format",
              flush=True)


def _legacy_requests(llm, items: dict, ids: list[str]) -> list[dict]:
    """The original prompt path (tokenizer chat template), kept to reproduce earlier runs."""
    from PIL import Image
    tok = llm.get_tokenizer()
    requests = []
    for iid in ids:
        item = items[iid]
        image = Image.open(io.BytesIO(base64.b64decode(item["image_b64"]))).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": prompt_of(item)}]}]
        try:            # most VLMs ship a chat template; fall back to a bare prompt if not
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = f"<image>\n{prompt_of(item)}"
        requests.append({"prompt": text, "multi_modal_data": {"image": image}})
    return requests


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--items", required=True)
    p.add_argument("--models", default="", help="comma-separated Hugging Face ids")
    p.add_argument("--models-file", default="", help="file with one model id per line")
    p.add_argument("--out", default="responses.jsonl")
    p.add_argument("--limit", type=int, default=0, help="only this many items (smoke test)")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-frac", type=float, default=0.90)
    p.add_argument("--legacy-prompt", action="store_true",
                   help="build prompts from the tokenizer's chat template (the original path)")
    a = p.parse_args()

    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.models_file:
        with open(a.models_file) as f:
            models += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not models:
        raise SystemExit("give --models or --models-file")

    items = load_items(a.items)
    ids = list(items)[:a.limit] if a.limit else list(items)
    print(f"{len(models)} models x {len(ids)} items -> {a.out}", flush=True)

    with open(a.out, "a") as out:                  # append: a crashed model does not lose the rest
        for model_id in models:
            try:
                run_model(model_id, items, ids, out, a.max_tokens, a.max_model_len, a.gpu_frac,
                          a.legacy_prompt)
            except Exception as e:                 # one unsupported model must not end the sweep
                print(f"[{model_id}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
