# adaptive-eval

Adaptive evaluation for language models. Instead of running every model on every benchmark
item, this service uses item response theory (IRT) to pick the items that are most informative
for each model, runs many models' evaluations concurrently under per-provider rate limits,
caches every response, and survives crashes mid-evaluation.

The approach follows Fluid Benchmarking (Hofmann et al., COLM 2025), which applies IRT and
adaptive item selection to LM evaluation. This repo reimplements the method as an async
evaluation service and tests it on held-out model families.

It is built in three stages: **A**, a single-process service; **B**, a distributed version
(workers, Redis queues, Postgres event log) that reproduces A exactly under fault
injection; and **C**, adaptive *ranking* of many models at once on a procedurally
generated spatial-reasoning benchmark for vision-language models.

## At a glance

- **Fewer calls, better rankings (A).** On ARC-Challenge, 40 adaptively chosen items rank
  held-out model checkpoints better than 160 random items, on both held-out families
  (Kendall tau 0.79 vs 0.66 on Pythia-6.9B, 0.63 vs 0.56 on OLMo-2-7B).
- **Crash-safe distributed evaluation (B).** Scheduler plus stateless workers over Redis
  Streams and a Postgres event log. It matches the single-process design's ability
  estimates to 9 decimals. Killing workers or the scheduler with `kill -9`, or wiping Redis
  mid-run, still reproduced a clean run exactly; the worst case re-paid 1 call of 2,227.
  Throughput scales 5.7x at 8 workers until the shared rate limit binds.
- **Ranking many models at once (C).** An allocator that spends each call where it most
  reduces expected misordered pairs. On real answers from 7 VLMs it reaches Kendall tau 0.96
  at 80 calls per model vs 0.89 for per-model adaptive testing; on simulated 64-model
  leaderboards it wins all 10 from 10 calls per model on.
- **IRT with too few models.** 8 models cannot fit per-item parameters (held-out log-loss
  0.676 on new items, near a constant's 0.687). Predicting difficulty from the item
  generator's settings gets 0.469.
- **Negative results are reported, not dropped.** An inflated-discrimination failure that
  ranked checkpoints backwards (A); a lookahead flaw in the allocator; two fixes that did not
  help; and a baseline showing that placing new models on a leaderboard needs about 40 calls
  each before any method beats leaving them at the average score (C).
- **Tested:** 95 tests, including exact-equivalence tests between designs and crash/replay
  tests against real Postgres and Redis.

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

## Design B: distributed evaluation

A scheduler process and a pool of stateless workers, coordinated through Redis Streams,
with an atomic Redis rate limiter shared by all workers and Postgres as the source of truth.
On the same data, a distributed run reproduces Design A's results exactly (same theta to
9 decimals, same item counts), including when the scheduler is killed mid-run and restarted.

**The scheduler is a single point of failure by design.** It is recoverable, not highly
available: on restart it rebuilds every session from the Postgres event log, re-enqueues
pending steps (duplicates are harmless because every write is idempotent), and takes over
results it had received but not acknowledged.

**Fault injection** (`scripts/chaos.py`): killing one worker, all workers, or the scheduler with `kill -9`, or wiping Redis with `FLUSHALL` mid-run, with speculation on or off, always produced results identical to a clean run (60 sessions, no orphaned steps).

### Admission under a budget

When the budget is scarce, the order in which ready sessions get calls matters. Three rules,
60 sessions, 5 repeats each (mean; sd at most 0.9 sessions and 0.009 tau):

| Budget | Metric | FIFO | SE reduction per $ (breadth) | Fewest calls to target (depth) |
|---|---|---:|---:|---:|
| $0.80 | Sessions reaching SE target | 9.0 | 5.2 | **25.4** |
| | Kendall tau vs full benchmark | 0.715 | 0.720 | 0.712 |
| $1.10 | Sessions reaching SE target | 24.8 | 20.2 | **37.0** |
| | Kendall tau vs full benchmark | **0.780** | 0.767 | 0.724 |

The rule trades completion against ranking quality. Depth-first admission completed 2.8x
more sessions than FIFO at $0.80 and 49% more at $1.10, but lowered ranking agreement at
$1.10. The SE-reduction-per-dollar rule, the original design, won neither metric. The right
rule depends on whether the goal is finished evaluations or a good leaderboard. Raw tables
are in `results/admission/`.

### Speculative prefetch

While a session waits on an item, the scheduler computes the two items it would pick next,
one per possible outcome, and prefetches them into the cache. Item selection is
deterministic, so these are exact predictions. Speculative jobs only warm the cache and
never write session events, so speculation cannot change results. Two guards protect real
work: a provider must have at least 30% of its rate-limit capacity free, and speculative
spend is capped at 20% of total spend.

60 sessions, 3 repeats each (mean):

| | Wall clock | Median session | Total cost | Spec hit rate | Wasted spec $ |
|---|---:|---:|---:|---:|---:|
| Off | 17.7 s | 5.56 s | $1.385 | - | - |
| On, 30% free-capacity guard | 17.8 s | 5.76 s | $1.397 (+0.9%) | 2% | $0.012 |
| On, no guard | 19.4 s (+9%) | 5.70 s | $1.448 (+4.5%) | 14% | $0.063 |

Speculation never changed results (identical in all 12 runs). In this rate-limited replay
setup it gave no speedup: with the guard it fired rarely, and without the guard it hit 14%
of next steps but made runs 9% slower and 4.5% more expensive by competing with real calls
for rate-limit capacity. Speculation can only reuse spare capacity; it should pay off when
calls are slow and rate limits are loose, which this benchmark does not exercise.

### Scaling

Throughput vs worker count (2 concurrent calls per worker, simulated calls at 4x replay
latency, 60 sessions):

| Workers | Loose limit (calls/s) | Tight limit (calls/s) |
|---:|---:|---:|
| 1 | 12.0 | 12.2 |
| 2 | 23.7 | 24.1 |
| 4 | 44.1 | 29.4 |
| 8 | 68.2 | 35.2 |

Near-linear while workers are the bottleneck (5.7x throughput at 8 workers, wall clock
53.8 s -> 9.7 s), then a plateau once the shared rate limit binds: under the tight limit,
going from 4 to 8 workers adds only 1.2x.

### Speculation across stopping rules

In a latency-bound regime (loose limits, slow calls, 20% spend cap), speculation still did
not pay off: at se_target 0.2 the wall-clock change was within noise (+1.1% cost), and at
0.3 and 0.4 runs got 9% and 13% slower (+5.2% and +7.8% cost) with hit rates of 19-24%.
Speculation stays correctness-safe (identical results in all 18 runs), but its per-step
planning overhead outweighed the savings here.

### Recovery cost

Extra paid calls caused by each fault, vs a clean run of 2,227 paid calls (3 runs each):
killing one worker, the scheduler, or wiping Redis re-paid nothing; killing all workers
re-paid 1 call in one of three runs (the at-least-once window between receiving a response
and caching it). Calls killed mid-flight are not logged, so a real provider could bill for
calls this count misses.

### Door to Design C

B logs every admission decision as an `allocation` event so that Design C's coupled
decisions could be audited and replayed; C uses exactly that. The rest of the plan written
here (reuse the admission heap, extend `irt.py`) did not survive; see
[How Design C departed from B's plan](#how-design-c-departed-from-bs-plan).

## Spatial benchmark (VLM items)

A procedural benchmark in the style of Spatial-IQ (Rim et al.): stacked-cube structures
rendered isometrically in pure Python, with ground truth taken from the scene description,
so items are cheap to make and cannot be mislabelled. Each scene yields counting items
(visible, total, hidden, tallest stack) and relation items (left/right, higher, nearer).
Visibility is measured by re-rendering each cube in its own ID colour and reading back which
survive: a rule-based test is wrong, because a cube can be fully covered by a combination of
neighbours rather than by any single one.

202 items across four difficulty tiers, four models, every model answered every item
(808 calls, $13.37):

| Model | Accuracy |
|---|---:|
| Claude Opus 5.5 | 0.59 |
| Claude Sonnet 5 | 0.55 |
| Claude Opus 5 | 0.49 |
| Claude Haiku 4.5 | 0.45 |

Accuracy by sub-task, pooled over models:

| Sub-task | Accuracy |
|---|---:|
| relation_left_right | 1.00 |
| relation_height | 0.89 |
| relation_near_far | 0.70 |
| tallest_column | 0.63 |
| count_visible | 0.31 |
| count_hidden | 0.18 |
| count_total | 0.17 |

Spatial relations are read off almost perfectly while counting collapses, especially where it
requires inferring cubes that cannot be seen. That is the decomposition's point: perception
is close to solved, and what sits on top of it is not.

**A harness bug produced a fake result first.** With a 16-token reply limit, the models that
reason before answering returned empty text and were scored wrong on every item: Opus 5 came
out at 0.00 and Opus 5.5 at 0.03, "worse" than Haiku, which answers in four tokens. Raising
the limit and extracting the answer from a longer reply (last number, or last of the item's
options) moved them to 0.49 and 0.59. Validate the harness before believing the leaderboard.

**Difficulty saturates with size.** Mean accuracy by tier (base grid and stack height 2
through 5): 0.81, 0.48, 0.42, 0.40. Almost the whole drop happens between tier 2 and tier 3;
tier 5 structures hold roughly five times the cubes of tier 3 and are barely harder. Once
counting collapses, adding cubes stops adding difficulty, so extending the scale upward needs
a different knob (occlusion depth, ambiguity) rather than a bigger grid.

**Item difficulty spans 3.2 logits** across sub-tasks, from b = -1.89 (left/right) to
b = +1.35 (total count), from the same rendered scenes -- the perception/inference gap,
measured rather than asserted.

**The IRT fit here is a pipeline check, not a measurement.** Four models is far below the
usual floor (~15-30), and it shows: fitted discriminations sit on the prior (all a ~ 1.0) and
difficulty is a monotone transform of accuracy (corr(b, 1 - accuracy) = 0.999), so IRT adds
nothing over raw accuracy at this sample size. Item parameters need many more models.

### Five models: weak models reveal which items discriminate

A second item set (116 items, 3 tiers, an occlusion knob, and three-way comparisons) run
against five models: two Claude models through the API and three open VLMs run locally on an
8 GB M1 through MLX, behind the same Provider interface as the API models.

| Model | Accuracy |
|---|---:|
| Claude Opus 5.5 | 0.63 |
| Claude Haiku 4.5 | 0.51 |
| Qwen2.5-VL 3B (4-bit) | 0.39 |
| Qwen2-VL 2B (4-bit) | 0.33 |
| SmolVLM 2B (4-bit) | 0.31 |

Item difficulty spans 3.2 logits, and the ordering is the same decomposition the pilot found:

| Sub-task | b | Accuracy |
|---|---:|---:|
| relation_left_right | -1.97 | 0.91 |
| support_on_ground | -1.18 | 0.74 |
| relation_height | -1.08 | 0.71 |
| triple_leftmost | -1.01 | 0.69 |
| relation_near_far | -0.79 | 0.63 |
| triple_nearest | -0.77 | 0.63 |
| triple_highest | -0.37 | 0.54 |
| count_above_red | +0.32 | 0.34 |
| tallest_column | +0.42 | 0.33 |
| count_visible | +0.64 | 0.28 |
| count_hidden | +1.12 | 0.18 |
| count_ground | +1.15 | 0.18 |
| count_total | +1.21 | 0.17 |

**Adding a weak model changed which items looked useful.** On Claude models alone,
`relation_height` (0.93) and `triple_highest` (0.71) looked like ceiling items worth cutting.
Qwen2.5-VL 3B scores 0.29 and 0.14 on them, so they are among the most discriminating items
in the bank: they separate weak models from strong ones, which is exactly what an item is
for. Two families really are dead: `relation_left_right` is at or near 1.00 for every model,
and `count_ground` is flat at about 0.2 across the whole ability range.

**Local models are cheap but uneven.** Per item: Qwen2-VL 2B about 1.5 s, Qwen2.5-VL 3B about
3 s, SmolVLM 2B about 23 s (its image tiling, not generation: a smaller token budget did not
help). 8 GB of unified memory caps the local set at about 3B, so the locally runnable models
all sit in a narrow 0.31-0.39 band.

**IRT still needs more models than this.** At four models and again at five, fitted
discriminations stay pinned to the prior (a in 0.89-1.10) and difficulty remains a monotone
transform of accuracy (corr(b, 1 - accuracy) = 0.997). The "~15-30 models" floor is a real
requirement, not a conservative hedge: every result reported here is an accuracy result, and
the IRT fit is a pipeline check. Filling the middle of the ability range needs 7B-34B open
VLMs, which need a rented GPU rather than an 8 GB laptop.

## Design C: adaptive ranking across models

Designs A and B measure one model at a time. Design C ranks a set of models, which couples
their evaluations: a call spent on one model is worth more when that model's neighbours on
the leaderboard are close. The allocator spends each call on the (model, item) pair that most
reduces the expected number of misordered pairs, and it runs on B's distributed stack.

**Data.** 746 spatial items (13 sub-tasks, grid sizes 2-4, `data/big_items.json`) and 8
models that answered them: three open VLMs run with vLLM on a rented GPU, two Claude models
through the API, and three 4-bit models through MLX (SmolVLM answered only 103 items and is
left out of the rankings). Five more open VLMs were run afterwards to test placing new models
on the leaderboard (below); two others that were attempted failed to load (a tokenizer
incompatibility with the current vLLM, and a gated repository).

### Eight models is too few for per-item IRT, so item features stand in

Every IRT fit on the spatial data so far sat on the prior: 8 models cannot pin down two
parameters per item. But these items are generated, so their difficulty can be predicted
from the generator's settings (sub-task, share of cubes hidden, cube count, grid size) plus
a small, heavily regularized per-item residual: about 30 free parameters plus residuals
held near zero, instead of ~1,500 free ones. Each sub-task
loads on one of two abilities, fixed in advance: counting/occlusion or spatial relations.

Held-out log-loss, 5-fold (a constant prediction scores 0.687):

| Model | Held-out answers | Held-out scenes (new items) |
|---|---:|---:|
| 2PL, parameters per item (Design A's model) | 0.498 | 0.676 |
| Difficulty per sub-task only | 0.511 | 0.514 |
| Explanatory, 1 ability | 0.468 | 0.472 |
| Explanatory, 1 ability + item residual | 0.454 | 0.472 |
| Explanatory, 2 abilities (counting, relations) | 0.467 | 0.469 |
| Explanatory, 2 abilities + item residual | **0.453** | **0.469** |

Per-item 2PL is barely better than a constant on items it has not seen; the feature model
predicts them almost as well as items it has. Two abilities beat one by a small but
consistent margin (0.002, on all 5 extra seeds for both splits), and the two correlate at
0.78.

**Ranking target.** With two abilities, "rank the models" needs a weighting. The score is
accuracy weighted 50/50 between the counting and relations families, so the 70% of items that
are counting questions (each scene yields more of them) do not set the weight by accident.
Sweeping the counting weight from 0 to 0.7 leaves the order of the seven fully answered
models unchanged; it reshuffles only near 1.0, where a counting-only score drops
Qwen2.5-VL-32B (weakest at counting in the middle of the pack, best at relations after Opus)
below Claude Haiku and Qwen2.5-VL-7B.

### Allocation: expected misordered pairs, with a multi-step lookahead

Each model's score is estimated from its answers plus IRT predictions for items it has not
answered, so the estimate equals the true score once every item is answered. The allocator
asks the model whose next answer is expected to remove the most misordered pairs,
averaging over how that answer could move the score (so an exact tie still shows a gain).

**The first version could stall late in a run.** One answer moves a score by about 1/n, so
once a pair's gap is a few times that, no single answer can flip it: a one-step lookahead
scores the pair near zero even while it is still 20% likely to be misordered, and once every
pair looks like that, allocation degrades to a tie-break. The fix scores each model by its
best gain per call over the next 1, 2, 4, ..., 64 answers. The flaw matters mostly near the
end: on real data the two rules agree at 5, 10 and 40 calls per model and differ at 80
(tau 0.924 one-step, 0.962 with the fix). The range was not tuned: on 16 simulated models,
extending it to 256 answers changes nothing, shortening it to 8 moves tau by at most 0.007,
and even the one-step rule stays within 0.02 (`results/c_lookahead.txt`).

### Ranking from scratch

Real answers, 7 models, the item bank calibrated on half the scenes and the models ranked on
the other half (5 random splits). Truth is each model's actual 50/50 accuracy on the ranking
half. Kendall tau:

| Calls per model | Random | Independent | Coupled |
|---:|---:|---:|---:|
| 5 | 0.27 | **0.61** | 0.52 |
| 10 | 0.42 | 0.58 | **0.64** |
| 20 | 0.52 | 0.77 | **0.83** |
| 40 | 0.69 | 0.79 | **0.87** |
| 80 | 0.79 | 0.89 | **0.96** |

"Independent" is Design A/B behaviour: each model gets the same number of calls and asks the
item that most shrinks its own score's variance. Coupled wins from 10 calls per model (by
0.06-0.08 tau, winning or tying 3-5 of 5 splits), and at 80 calls it misorders 0.4 of 21
pairs against 1.2. It does so by moving calls: at 80 calls per model, the two closest
mid-table models (Claude Haiku and Qwen2-VL-7B) got about 135 calls each and the clear
leader (Opus) 23. One misordered pair is
0.095 tau here, so these are differences of about one pair.

**At scale, simulated.** Leaderboards of 8-64 models drawn from the fitted population, with
answers generated from the full fit and a deliberately imperfect bank (difficulties from item
features only). Coupled minus independent, Kendall tau, 10 leaderboards per size:

| Calls per model | 8 models | 16 | 32 | 64 |
|---:|---:|---:|---:|---:|
| 5 | -0.02 | +0.01 | +0.01 | -0.01 |
| 10 | +0.10 | +0.08 | +0.03 | **+0.05** |
| 20 | +0.14 | +0.08 | **+0.08** | **+0.10** |
| 40 | +0.05 | **+0.08** | **+0.08** | **+0.12** |
| 80 | +0.06 | **+0.11** | **+0.08** | **+0.08** |

Bold: 95% interval excludes zero. The advantage grows more consistent as the leaderboard
gets denser (64 models: 10 of 10 leaderboards won at every budget from 10 calls). Coupled
never helps measurably at 5 calls per model, on real or simulated data: it needs a few
answers per model before it can tell which pairs are close.

### Placing new models on an existing leaderboard

Five more open VLMs, run on all 746 items afterwards (their answers are the ground truth),
placed against the 7 known models with the bank calibrated on the original models only.
True leaderboard, new models starred:

1. Claude Opus 5.5 0.692 · 2. Qwen2.5-VL-32B 0.600 · 3. ★ Pixtral-12B 0.569 · 4. Claude Haiku
4.5 0.563 · 5. Qwen2-VL-7B 0.561 · 6. Qwen2.5-VL-7B 0.547 · 7. ★ LLaVA-OneVision-7B 0.526 ·
8. ★ Idefics3-8B 0.518 · 9. ★ Phi-3.5-vision 0.471 · 10. Qwen2.5-VL-3B 0.457 · 11. ★
LLaVA-1.5-7B 0.347 · 12. Qwen2-VL-2B 0.340

Pairs involving a new model ordered correctly, out of 45 (20 repeats on random 80% subsets
of scenes):

| Calls per new model | Random | Independent | Coupled |
|---:|---:|---:|---:|
| 0 (every new model left at the average score) | 37.9 | 37.9 | 37.9 |
| 5 | 34.9 | 32.8 | 34.0 |
| 20 | 36.6 | 36.4 | 36.4 |
| 40 | 38.5 | 39.8 | 40.2 |
| 80 | 40.5 | **42.2** | 41.8 |

**The zero-call row is the finding.** A new model left at the average score lands mid-table,
and a mid-table guess already orders 38 of 45 pairs right on this leaderboard. No method
beats that until about 40 calls per new model; after that, adaptive beats random by about
1.5 pairs, and coupled ties independent. With the anchors' scores known exactly, coupling
only decides which new model to ask next, and that choice matters less than which item to
ask.

**Two fixes were tested and not adopted.** Both are kept in the code, off by default, so the
results reproduce:
- *Indifference zone:* ignore pairs closer than the score's own sampling noise (about 0.02),
  on the theory that coupled wasted calls separating near-ties (Pixtral got a third of all
  calls). No change: within 0.4 pairs everywhere.
- *Content balancing* (standard in adaptive testing): force each model's calls to follow the
  sub-tasks' share of the score, since variance-minimizing selection took 43 of its first 50
  calls from 2 of the 13 sub-tasks. It looked like a large win in placement at 5-10 calls,
  but that was the zero-call effect: its early items barely move the estimates. Ranking from
  scratch exposed it, dropping tau at 5 calls per model from 0.61 to 0.05.

### Distributed scheduler

`adaptive_eval/c/scheduler.py` runs the allocator on B's workers, Redis queues and Postgres
event log unchanged. Each model has at most one call in flight (a second would be chosen as
if the first had not happened), per-provider windows cap concurrency, and every decision is
logged as an `allocation` event with a global sequence number and the gain it was made on.

**Replay.** With one call in flight, the distributed run makes exactly the offline
allocator's decisions, including across a scheduler crash, and on sp6 reproduces its tau
(0.829 and 0.962 at 20 and 80 calls). With several in flight, the order results arrive in is
not reproducible, so recovery rebuilds state from the log and the tests check invariants
instead: no model with two calls out, no item asked twice, every answer preceded by a logged
allocation, nothing orphaned, and final scores recomputable from the log alone. Deliberately
breaking recovery, or allowing two calls per model, fails those tests.

**Concurrency costs accuracy unless some slots stay empty.** Filling every free slot forces a
call onto each provider's best model even when that call is nearly worthless. Leaving a slot
idle when its best call is worth less than half the best call anywhere (set once, not tuned)
recovers one-at-a-time quality. 7 models on 3 simulated providers:

| Calls per model | Setup | Tau | Wall clock |
|---:|---|---:|---:|
| 20 | one call at a time | 0.83 | 7.7 s |
| 20 | 1 per provider, fill every slot | 0.71 | 2.6 s |
| 20 | 2 per provider, idle rule | 0.81 | 2.3 s |
| 80 | one call at a time | 0.96 | 30.6 s |
| 80 | 2 per provider, fill every slot | 0.92 | 5.6 s |
| 80 | 2 per provider, idle rule | 0.96 | 10.1 s |
| 80 | every model at once | 0.89-0.91 | 5 s |

Useful parallelism is bounded by the number of models: with every model in flight, the
allocator has nothing left to choose.

### How Design C departed from B's plan

B's plan was to reuse its admission heap with a new priority and to give `irt.py` a
multidimensional theta behind the same interface. Neither survived contact. Priorities in a
heap go stale the moment any answer lands, because every answer changes its neighbours'
gains, so the C scheduler re-decides from current state on each result instead of queueing.
And per-item IRT could not be fit at all on 8 models, so C uses a separate explanatory,
two-ability bank (`adaptive_eval/c/ranking.py`) rather than extending `irt.py`.

### Caveats

- Real leaderboards here have 7-12 models; the evidence at 32-64 models is simulated from a
  model fitted on those 12.
- One benchmark (the spatial items), one item generator, and a 50/50 target chosen by hand.
- The truth for every ranking is accuracy on this item pool, which carries its own sampling
  noise (about 0.02). Several true gaps are smaller than that.
- New-model answers were batch-generated and replayed, not served live; at temperature 0 a
  live run would give the same answers, but its latency and failure behaviour are untested.
- Only 5 real splits for ranking from scratch: differences of one pair out of 21 are within
  noise.

### Reproduce

```bash
python scripts/explanatory_cv.py                       # item bank: held-out log-loss
python scripts/ranking_experiment.py                   # ranking from scratch (real)
python scripts/ranking_scaling.py --sizes 8,16,32,64   # simulated scaling (~25 min)
python scripts/placement_experiment.py                 # new models on the leaderboard
python scripts/c_concurrency_experiment.py             # distributed, needs PG_DSN/REDIS_URL
python -m pytest -q tests/test_ranking.py tests/test_c_scheduler.py
```

The distributed path is exercised by `c_concurrency_experiment.py` and the scheduler tests,
which start their own workers. `python -m adaptive_eval.c.scheduler` (with a bank from
`scripts/make_bank2d.py`) is the command-line entry point, but it has not yet been run
against separately started workers.

The data files (`data/sp6_*.json`) are exported from Postgres with
`scripts/spatial_matrix.py`; raw results are in `results/c_*.txt`.

## Known limitations

- **The scheduler is a single process.** It is recoverable (it rebuilds from the Postgres
  event log after a crash), not highly available.
- **Delivery is at-least-once.** A crash between a provider's response and the cache write
  re-pays that call. Measured across 12 fault-injection runs: at most 1 re-paid call out of
  2,227. Calls killed mid-flight are not logged, so a real provider could bill for calls
  this count misses.
- **Redis is a single instance with no persistence guarantees.** That is acceptable only
  because Postgres is the source of truth: wiping Redis mid-run loses work, not results.
- **The benchmark results use simulated providers.** Latency and failure distributions do
  not match real APIs: replay calls take ~50 ms, the live API ~1.2 s per call, which is the
  latency-bound regime where speculation would matter most and the experiments here do not
  reach. The real provider itself is verified end to end (`adaptive_eval/real_provider.py`):
  a 10-call live run cost $0.0005 and exercised auth, grading, retry classification, and
  per-token cost accounting.
- **Rate limits must admit a single call.** `tpm / 60 * burst_s` has to exceed a call's
  estimated tokens, or the limiter rejects the call up front (rather than waiting forever
  for a bucket that can never fill).

## Roadmap

- **A live Design C run:** serve a new model with `vllm serve` and place it on the leaderboard
  through the command-line scheduler, instead of replaying batch answers.
- **More real models:** 12 are fully answered; around 15-30 would let per-item IRT be fitted
  and checked against the feature-based bank.
- **A second benchmark** for Design C, so the ranking results do not rest on one item
  generator.

## Data and attribution

Real-data experiments use the
[Fluid Benchmarking dataset](https://huggingface.co/datasets/allenai/fluid-benchmarking)
(CC BY 4.0) from Hofmann et al., "Fluid Language Model Benchmarking," COLM 2025
([arXiv:2509.11106](https://arxiv.org/abs/2509.11106)). The data is downloaded locally and is not
redistributed in this repo.
