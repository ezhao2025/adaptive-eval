# adaptive-eval

Adaptive evaluation for language models. Instead of running every model on every benchmark
item, this service uses item response theory (IRT) to pick the items that are most informative
for each model, runs many models' evaluations concurrently under per-provider rate limits,
caches every response, and survives crashes mid-evaluation.

The approach follows Fluid Benchmarking (Hofmann et al., COLM 2025), which applies IRT and
adaptive item selection to LM evaluation. This repo reimplements the method as an async
evaluation service and tests it on held-out model families.

## Results on real data

**Setup.** ARC-Challenge per-item results (1,172 items) for 452 pretraining checkpoints of six
LMs, from the Fluid Benchmarking data release. One LM family is held out at a time and the 2PL
IRT model is fit on the rest. For each held-out checkpoint, each method estimates ability (θ)
from a fixed budget of items. The metric is Kendall τ between those estimates and θ estimated
from all 1,172 items. Adaptive selection is deterministic; random selection is reported as
mean ± sd over 20 seeds.

**Headline: on both held-out families, 40 adaptively selected items rank checkpoints better
than 160 random items (4× fewer).**

Pythia-6.9B, 78 checkpoints (both Pythia sizes excluded from the IRT fit):

| Items | Adaptive τ | Random τ (mean ± sd) | Best random seed |
|------:|-----------:|---------------------:|-----------------:|
| 20  | 0.763 | 0.338 ± 0.064 | 0.44 |
| 40  | 0.786 | 0.438 ± 0.074 | 0.59 |
| 80  | 0.681 | 0.565 ± 0.042 | 0.64 |
| 120 | 0.703 | 0.615 ± 0.034 | 0.67 |
| 160 | 0.764 | 0.661 ± 0.038 | 0.73 |

OLMo-2-7B, 94 checkpoints:

| Items | Adaptive τ | Random τ (mean ± sd) | Best random seed |
|------:|-----------:|---------------------:|-----------------:|
| 20  | 0.546 | 0.296 ± 0.046 | 0.40 |
| 40  | 0.633 | 0.373 ± 0.063 | 0.49 |
| 80  | 0.675 | 0.476 ± 0.049 | 0.57 |
| 120 | 0.815 | 0.519 ± 0.039 | 0.61 |
| 160 | 0.836 | 0.560 ± 0.046 | 0.64 |

Adaptive beats the best of 20 random seeds at every budget on both families.

### Finding: correlated checkpoints inflate item discrimination

The first real-data run failed in an instructive way. With the default discrimination prior
(σ_log_a = 0.5), the fitted discriminations reached a = 22.1, with 141 items above 5. Neighboring
checkpoints of the same LM are near-duplicates, but the fit treats them as independent evidence,
which overwhelms the prior. An item with a ≈ 22 behaves like a step function, and max-information
selection kept choosing those items. The result was that adaptive selection ranked OLMo-2's
checkpoints *backwards* (τ ≈ −0.35 at 40–160 items), while random selection behaved normally.

The fix was to choose σ_log_a on a validation LM (Amber-7B, excluded from the fit) before looking
at the test family again. Adaptive / random τ on the validation LM (random here is a single draw):

| σ_log_a | 20 items | 40 items | 80 items |
|--------:|---------:|---------:|---------:|
| 0.5  | −0.33 / 0.21 | −0.34 / 0.24 | −0.34 / 0.36 |
| 0.25 |  0.31 / 0.34 |  0.39 / 0.37 |  0.45 / 0.53 |
| 0.1  |  0.64 / 0.37 |  0.72 / 0.40 |  0.87 / 0.51 |
| 0.05 |  0.75 / 0.35 |  0.77 / 0.40 |  0.83 / 0.46 |
| 0.02 |  0.68 / 0.35 |  0.76 / 0.39 |  0.74 / 0.44 |

σ_log_a = 0.05 was chosen by a rule fixed in advance (best adaptive τ at 20–40 items) and then
reused unchanged for the Pythia holdout. Near-Rasch (σ = 0.02, every a ≈ 1) does worse, so a
modest spread in discrimination helps; unconstrained discrimination is what breaks max-information
selection.

### Caveats

- One benchmark (ARC-Challenge). τ measures ranking *within one LM's training trajectory*, not
  ranking across different models.
- The reference ranking is θ from all 1,172 items under the same IRT model, so τ measures
  agreement with full-benchmark IRT ability, not with an external ground truth. On OLMo-2, that
  reference tracks training step closely (Spearman 0.87).
- OLMo-1-7B, a related model, was in the IRT fit for the OLMo-2 holdout. The Pythia holdout
  excludes both Pythia sizes and is the cleaner test.
- The adaptive curve is not monotone in budget (each checkpoint follows one deterministic item
  path), so treat per-budget values as approximate.
- σ_log_a was selected on a single validation LM.

## System design

Everything runs in one asyncio process with SQLite storage.

| Module | Role |
|---|---|
| `irt.py` | 2PL MAP fit (L-BFGS), Newton ability estimate with standard error, max-information item selection (deterministic tie-breaking) |
| `data.py` | Response-matrix format, synthetic 2PL generator, held-out split |
| `providers.py` | Async provider interface; `ReplayProvider` replays logged answers with simulated latency and transient failures; `AnthropicProvider` skeleton |
| `ratelimit.py` | Per-provider token buckets for requests/min and tokens/min, FIFO so no session starves |
| `storage.py` | Response cache keyed by SHA-256 of everything that can change an answer; append-only event log with idempotent writes |
| `engine.py` | Adaptive session loop with write-ahead logging, retries with exponential backoff and jitter, concurrent scheduler |
| `report.py` | Call reduction, ranking agreement, cache hit rate, adaptive-vs-random curve |

**Crash recovery.** Each step logs `item_selected` before the call and `answer_recorded` after
it, with `UNIQUE(session_id, step, type)` as the idempotency key. On resume, a session is rebuilt
from its log, and θ is always recomputed from the log rather than stored. A run crashed after
six steps and then resumed produces final results identical to an uninterrupted run (all
metrics match; only wall-clock time differs). This is also covered by an automated test, and was checked with a real `kill -9` mid-run.

The guarantee is at-least-once, not exactly-once: if the process dies after a provider
returns but before the response is cached, that call is paid for again on resume.

Synthetic-data runs (call reduction, cache reuse when tightening the stopping rule, crash demo)
are in `results/`.

## Quickstart

Setup uses [uv](https://github.com/astral-sh/uv) with Python 3.12:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest -q
```

Synthetic pipeline:

```bash
python -m adaptive_eval.cli gen-synthetic --models 60 --items 400 --seed 0
python -m adaptive_eval.cli fit
python -m adaptive_eval.cli run --run-name adaptive-v1
python -m adaptive_eval.cli report --run-name adaptive-v1
python -m adaptive_eval.cli curve
```

Real-data pipeline (ARC-Challenge, OLMo-2 holdout):

```bash
uv pip install huggingface_hub
hf download allenai/fluid-benchmarking --repo-type dataset --local-dir data/fluid
python scripts/convert_fluid.py --benchmark arc_challenge --out data/fluid_arc.json --holdout olmo2-7b
python scripts/sigma_sweep.py --data data/fluid_arc.json --val-prefix amber-7b/
python -m adaptive_eval.cli fit --data data/fluid_arc.json --out data/fluid_arc_params.json --sigma-log-a 0.05
python scripts/holdout_eval.py --data data/fluid_arc.json --params data/fluid_arc_params.json --prefix olmo2-7b/
```

For the Pythia holdout, convert with `--holdout pythia-7b,pythia-3b` and evaluate with
`--prefix pythia-7b/`.

Crash demo:

```bash
python -m adaptive_eval.cli run --run-name demo --db data/crash.db --crash-after 6
python -m adaptive_eval.cli run --run-name demo --db data/crash.db
python -m adaptive_eval.cli run --run-name demo --db data/clean.db
diff <(python -m adaptive_eval.cli report --run-name demo --db data/crash.db) \
     <(python -m adaptive_eval.cli report --run-name demo --db data/clean.db)
```

## Roadmap

- **Real provider:** finish `AnthropicProvider` (retry only on 429, 5xx and connection errors),
  with the image content hash added to the cache key for multimodal items.
- **Design B:** distributed workers, shared rate limits in Redis, Postgres storage, speculative
  prefetch of likely next items.
- **Design C:** adaptive ranking across models, and multidimensional IRT for a hierarchical
  spatial-reasoning benchmark for vision-language models.

## Data and attribution

Real-data experiments use the
[Fluid Benchmarking dataset](https://huggingface.co/datasets/allenai/fluid-benchmarking)
(CC BY 4.0) from Hofmann et al., "Fluid Language Model Benchmarking," COLM 2025
([arXiv:2509.11106](https://arxiv.org/abs/2509.11106)). The data is downloaded locally and is not
redistributed in this repo.
