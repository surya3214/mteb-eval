# MTEB Offline Eval Toolkit

Prefetch and evaluate embedding models on **MTEB(Multilingual, v2)** (default), filtered to the **ml16** language preset. Default task types are STS + Retrieval; Classification, Clustering, and Reranking are fully supported via `--task-types`. Also supports the older eng STS+Retrieval 19-task set via `--benchmark "MTEB(eng, v2)" --languages-preset none`.

Minimum versions are listed in [`requirements.txt`](requirements.txt). Pin exact versions in your environment if you need bit-identical scores across machines.

## Quick start

```bash
pip install -r requirements.txt
# Install torch for your CUDA version — see requirements-gpu.txt
```

## Two-machine workflow

### Step 1 — Internet machine (prefetch datasets)

```bash
pip install -r requirements.txt

python -m mteb_eval.prefetch \
  --cache-dir /data/hf_cache \
  --models Qwen/Qwen3-Embedding-4B

# Transfer cache to GPU machine
rsync -av /data/hf_cache/ gpu-host:/data/hf_cache/
```

Optional: freeze the exact environment after install:

```bash
pip freeze > requirements-lock.txt
```

### Step 2 — GPU machine (evaluate, Hub model)

```bash
pip install -r requirements.txt   # same mins as prefetch; pin exact versions for reproducibility

python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache \
  --offline \
  --model Qwen/Qwen3-Embedding-4B \
  --task-types STS Retrieval \
  --output-dir results/qwen3-4b \
  --batch-size 32 \
  --corpus-batch-size 4 \
  --device cuda
```

### Step 2b — GPU machine (evaluate, local fine-tuned model)

Copy the model separately from the HF cache (training output or `huggingface-cli download --local-dir`):

```bash
rsync -av ./harrier-ft/ gpu-host:/data/models/harrier-ft/

python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache \
  --offline \
  --model-path /data/models/harrier-ft \
  --model-type harrier \
  --task-types STS Retrieval \
  --output-dir results/harrier-ft \
  --device cuda
```

### Alternative — same machine, default cache

On a machine with internet access, skip cache relocation:

```bash
python -m mteb_eval.evaluate \
  --default-cache \
  --model microsoft/harrier-oss-v1-0.6b \
  --output-dir results/harrier \
  --device cuda
```

## Cache modes

| Flag | Behavior |
|------|----------|
| `--cache-dir PATH` | Portable cache: sets `HF_HOME`, `HF_HUB_CACHE`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` |
| `--default-cache` | No env override; uses `~/.cache/huggingface` |
| `--offline` | Sets `HF_DATASETS_OFFLINE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` (local `--model-path` still works) |

**Important:** `configure_cache()` runs at CLI startup before any Hugging Face imports. Do not import `mteb`, `datasets`, or `transformers` before calling it in custom scripts.

## Model loading

### Hub models (`--model`)

Uses `mteb.get_model()` / MTEB registry when available (required for correct **Qwen3** scores — see [mteb#2867](https://github.com/embeddings-benchmark/mteb/issues/2867)).

### Local folders (`--model-path`)

| Model family | Example `--model-path` | `--model-type` | Notes |
|--------------|------------------------|----------------|-------|
| Qwen3 | `/data/models/qwen3-4b` | `qwen3` | Pass `--hub-id Qwen/Qwen3-Embedding-4B` for MTEB instruct wrapper |
| Harrier | `/data/models/harrier-ft` | `harrier` | Needs `config_sentence_transformers.json` |
| EuroBERT ST | `/data/models/eurobert-st` | `sentence-transformer` | `trust_remote_code=True` |
| EuroBERT base | `/data/models/eurobert-210m` | `eurobert-base` | Custom mean-pool wrapper; not leaderboard-comparable |
| Generic ST | any `model.save()` folder | `auto` or `sentence-transformer` | |

`--model` and `--model-path` are mutually exclusive. A directory passed to `--model` is treated as a local path.

### Export Hub model to local folder

```bash
huggingface-cli download Qwen/Qwen3-Embedding-4B --local-dir /data/models/qwen3-4b
rsync -av /data/models/qwen3-4b gpu-host:/data/models/qwen3-4b
```

Models and datasets can live in different directories.

## Example commands by model

**Qwen3-4B (Hub, offline):**

```bash
python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache --offline \
  --model Qwen/Qwen3-Embedding-4B \
  --query-batch-size 32 --corpus-batch-size 4 \
  --output-dir results/qwen3-4b --device cuda
```

**Qwen3 local checkpoint:**

```bash
python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache --offline \
  --model-path /data/models/qwen3-4b-local \
  --hub-id Qwen/Qwen3-Embedding-4B \
  --model-type qwen3 \
  --output-dir results/qwen3-local --device cuda
```

**Harrier (Hub):**

```bash
python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache --offline \
  --model microsoft/harrier-oss-v1-0.6b \
  --output-dir results/harrier --device cuda
```

**EuroBERT ST fine-tune (local):**

```bash
python -m mteb_eval.evaluate \
  --default-cache \
  --model-path ./checkpoints/eurobert-st \
  --model-type sentence-transformer \
  --output-dir results/eurobert-local
```

## Prefetch CLI

```bash
python -m mteb_eval.prefetch --cache-dir /data/hf_cache
python -m mteb_eval.prefetch --cache-dir /data/hf_cache --tasks BIOSSES STS12
python -m mteb_eval.prefetch --default-cache --models microsoft/harrier-oss-v1-0.6b
```

Shell wrappers: [`scripts/prefetch.sh`](scripts/prefetch.sh), [`scripts/evaluate.sh`](scripts/evaluate.sh).

## Tasks (19 total)

**Retrieval (10):** ArguAna, CQADupstackGamingRetrieval, CQADupstackUnixRetrieval, ClimateFEVERHardNegatives, FEVERHardNegatives, FiQA2018, HotpotQAHardNegatives, SCIDOCS, TRECCOVID, Touche2020Retrieval.v3

**STS (9):** BIOSSES, SICK-R, STS12, STS13, STS14, STS15, STSBenchmark, STS17, STS22.v2

Manifest: [`mteb_eval/manifests/eng_v2_sts_retrieval.json`](mteb_eval/manifests/eng_v2_sts_retrieval.json) (~1–1.5 GB datasets).

## Evaluate CLI options

```
--model / --model-path     Model source (required, mutually exclusive)
--hub-id                   Canonical Hub id for local Qwen3
--model-type               auto | qwen3 | harrier | sentence-transformer | eurobert-base
--cache-dir / --default-cache
--offline
--benchmark                Default: MTEB(Multilingual, v2)
--task-types               Default: STS Retrieval (also: Classification Clustering Reranking)
--tasks                    Optional explicit task subset
--tasks-preset             retrieval-fast | mteb-eval-all
--languages                Language-script codes (overrides preset)
--languages-preset         Default: ml16 (use none for all languages)
--exclusive-language-filter  Keep only subsets where ALL languages match (default: ANY match)
--output-dir               Results + summary.json + summary.csv + language CSVs (required)
--batch-size / --query-batch-size / --corpus-batch-size
--max-seq-len              Default: 512 (truncation cap for long inputs like STS22)
--query-prefix             Optional query prefix (SentenceTransformer `prompts['query']`)
--document-prefix          Optional document/passage prefix (`prompts['document']`)
--device                   cuda, cpu, mps
--dtype                    Default: bfloat16 (also: auto | float32 | float16)
--attn-implementation      Default: sdpa (also: eager | flash_attention_2)
--overwrite                only-missing | always | never
--continue-on-error        Default: on (use --no-continue-on-error to stop on first error)
```

Throughput tips: defaults use `--dtype bfloat16` and `--attn-implementation sdpa`. Language filtering defaults to `--languages-preset ml16`. Use `--languages-preset none` for all subsets. These flags only affect model loading / task subset selection; MTEB scoring is unchanged.

## Multilingual evaluation (16 languages)

For `MTEB(Multilingual, v2)` (the default benchmark), `--languages-preset ml16` is **on by default** so only your 16 supported languages are evaluated. Tasks with no overlapping subsets are skipped automatically (e.g. `TwitterHjerneRetrieval` for Danish-only). Use `--languages-preset none` to disable.

```bash
python -m mteb_eval.prefetch \
  --cache-dir /data/hf_cache
  # defaults: Multilingual v2 + ml16 (no --no-validate-manifest needed)

python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache --offline \
  --model Qwen/Qwen3-Embedding-4B \
  --output-dir results/qwen3-ml16 \
  --device cuda
  # defaults: Multilingual v2, ml16, bfloat16, sdpa, continue-on-error
```

The `ml16` preset maps to: `eng-Latn`, `kor-Hang`, `ara-Arab`, `zho-Hans`, `fra-Latn`, `deu-Latn`, `hin-Deva`, `ind-Latn`, `ita-Latn`, `jpn-Jpan`, `por-Latn`, `rus-Cyrl`, `spa-Latn`, `vie-Latn`, `tha-Latn`, `pol-Latn`.

Cross-lingual subsets (e.g. `en-de` in STS17) are kept when **any** language in the pair is in your list. Use `--exclusive-language-filter` for strict all-language matching.

## Classification, Clustering, and Reranking

Same offline + ml16 workflow as STS/Retrieval. Pass the three types explicitly (defaults stay STS + Retrieval for backward compatibility). Prefetch auto-validates against the shipped ml16 inventory (`mteb_eval/manifests/multilingual_v2_clf_clust_rerank_ml16.json`) when these types are selected with Multilingual v2 + ml16.

```bash
# Internet machine
python -m mteb_eval.prefetch \
  --cache-dir /data/hf_cache \
  --task-types Classification Clustering Reranking

rsync -av /data/hf_cache/ gpu-host:/data/hf_cache/

# GPU machine
python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache --offline \
  --task-types Classification Clustering Reranking \
  --model Qwen/Qwen3-Embedding-4B \
  --output-dir results/qwen3-clf-clust-rerank-ml16 \
  --device cuda
```

Under ml16 this resolves to **38 tasks** (21 Classification, 12 Clustering, 5 Reranking). Tasks with no overlapping language subsets are skipped automatically. Requesting `Clustering` also expands to `HierarchicalClustering` when that type exists in the MTEB version.

Bi-encoder embedding models work for Reranking the same way as Retrieval (encode + similarity). Wall time on 1× H100 for a ~200M model is typically ~1–1.5 hours; **WebLINXCandidatesReranking** dominates.

## `mteb-eval-all` preset

`--tasks-preset mteb-eval-all` runs a practical full suite on Multilingual v2 (ml16 still applies by default):

- **All** STS, Classification, Clustering
- **Reranking** without `WebLINXCandidatesReranking`
- **Retrieval** = `retrieval-fast` only (12 tasks)

This preset **overrides** `--task-types`. Under ml16 it typically resolves to ~62 tasks.

```bash
python -m mteb_eval.prefetch \
  --cache-dir /data/hf_cache \
  --tasks-preset mteb-eval-all

python -m mteb_eval.evaluate \
  --cache-dir /data/hf_cache --offline \
  --tasks-preset mteb-eval-all \
  --model Qwen/Qwen3-Embedding-4B \
  --output-dir results/qwen3-mteb-eval-all \
  --device cuda
```

On completion, results are written to:

- `summary.csv` / language CSVs (unchanged)
- **`results.xlsx`** with sheets: `Summary`, per-type (`STS`, `Classification`, `Clustering`, `Reranking`, `Retrieval`), `ByLanguage`, `Detail`

## Fast Retrieval preset

`--tasks-preset retrieval-fast` runs the 12 quicker multilingual Retrieval tasks and skips the heavy multi-subset ones.

**Included (12):** ArguAna, SCIDOCS, AILAStatutes, LegalBenchCorporateLobbying, SpartQA, TempReasonL1, WinoGrande, StackOverflowQA, HagridRetrieval, StatcanDialogueDatasetRetrieval, TRECCOVID, LEMBPasskeyRetrieval

**Omitted (6):** BelebeleRetrieval, MIRACLRetrievalHardNegatives, WikipediaRetrievalMultilingual, MLQARetrieval, TwitterHjerneRetrieval, CovidRetrieval

```bash
python -m mteb_eval.evaluate_parallel \
  --cache-dir /data/hf_cache --offline \
  --task-types Retrieval \
  --tasks-preset retrieval-fast \
  --model Qwen/Qwen3-Embedding-4B \
  --gpus 0,1,2,3 \
  --output-dir results/qwen3-retrieval-fast
```

Typical H100 wall time for this preset is much lower than the full 18 Retrieval tasks (often ~30–90 min on 1 GPU for mid-size models, less with multi-GPU).

## Multi-GPU parallel evaluation

Use `evaluate_parallel` to split tasks across GPUs (one model copy per GPU). Wall-clock time drops roughly proportional to GPU count; VRAM usage is per-GPU.

```bash
python -m mteb_eval.evaluate_parallel \
  --cache-dir /data/hf_cache --offline \
  --model Qwen/Qwen3-Embedding-4B \
  --gpus 0,1,2,3 \
  --output-dir results/qwen3-ml16-parallel \
  --corpus-batch-size 4
```

- `--gpus auto` (default): use all visible CUDA devices
- Per-GPU shards are written to `{output_dir}/.shards/gpu{N}/`
- Merged `summary.json` and `summary.csv` are written to `--output-dir`
- Language-wise scores: `summary_by_language.csv` (mean per language) and `summary_by_language_detail.csv` (task × subset × language)

Shell wrapper: [`scripts/evaluate_parallel.sh`](scripts/evaluate_parallel.sh).


## Troubleshooting

- **Score mismatch across machines:** Pin identical `mteb`, `sentence-transformers`, and `transformers` versions (mins are in `requirements.txt`).
- **Dataset not found offline:** Run `prefetch` on the internet machine first; verify `rsync` completed.
- **Wrong FEVER/Hotpot size (~3 GB):** The toolkit uses v2 hard-negative repos (`FEVERHardNegatives`, `HotpotQAHardNegatives`), not full `mteb/fever`.
- **Local folder load fails:** Folder must contain ST artifacts (`modules.json`, `config_sentence_transformers.json`) or transformers weights (`config.json` + `*.safetensors`).
- **Qwen3 local scores differ:** Pass `--hub-id` so the MTEB instruct wrapper is applied.
- **Hub id passed to `--model-path`:** Use `--model <repo_id>` instead; `--model-path` must be an existing directory.
- **One task fails mid-run:** By default `--continue-on-error` is on so remaining tasks finish; failed tasks appear as `FAILED` in the summary and the process exits with code 1. Use `--no-continue-on-error` to stop on the first error.
- **Offline Retrieval `Couldn't find cache for config 'default'`:** Some Hub datasets store qrels under config `qrels` (not `default`). Prefer prefetching with `python -m mteb_eval.prefetch` so all of `corpus` / `queries` / `qrels` (or `default`) land in the cache, and use this toolkit's evaluate entrypoints (they apply an offline qrels compatibility shim). Ensure `rsync` copies both `hub/` and `datasets/` under `--cache-dir`.
- **Custom query/document prefixes:** Use `--query-prefix` / `--document-prefix` for SentenceTransformer-style models (e.g. `query: ` / `document: `). Instruct models (Qwen3, Harrier) use per-task instructions instead; prefixes are printed before each task runs.
- **STS22 OOM on long news articles:** Lower `--batch-size` (4–16 for large models) and/or reduce `--max-seq-len` (default 512). GPU memory is released between tasks via `gc.collect()` and `torch.cuda.empty_cache()`. Try `--dtype bfloat16` on Ampere+ GPUs to free VRAM.
- **bf16 vs fp32 scores:** `--dtype bfloat16` can produce tiny score deltas vs float32; still a valid MTEB run, but not bit-identical.
- **FlashAttention-2 failures:** Only use `--attn-implementation flash_attention_2` when FA2/kernels are installed and the model supports it. Prefer the default `sdpa`.

## Tests

```bash
pytest tests/ -v
```

## Project layout

```
mteb_eval/
  cache.py           HF cache / offline env setup
  offline_compat.py  Offline Retrieval qrels config shim
  languages.py       Language presets (ml16) and CLI resolution
  tasks.py           Task resolution + manifest validation + partitioning
  manifests/         Task inventories (eng STS+Retrieval; ml16 Clf/Clust/Rerank)
  runner.py          Shared evaluation loop
  prefetch.py        Dataset (+ optional model) download CLI
  evaluate.py        Single-GPU evaluation CLI
  evaluate_parallel.py  Multi-GPU parallel evaluation coordinator
  model_loader.py    Hub + local model loading
  prompts.py         Per-task prompt resolution
  summary.py         CSV/JSON summaries + shard merge
scripts/
  prefetch.sh
  evaluate.sh
  evaluate_parallel.sh
tests/
  test_smoke.py
```
