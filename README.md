# bioner-llm

Few-shot biomedical named entity recognition (BioNER) for chemicals and diseases using large language models, evaluated on the BC5CDR corpus.

LLMs are served via [vLLM](https://docs.vllm.ai) on an HPC cluster (SLURM + Singularity) and queried through a local FastAPI middleware. Entities are extracted using prompt strategies ranging from zero-shot to 32-shot, and results are evaluated against the BC5CDR gold standard.

## Results

Experiments were conducted over the full BC5CDR corpus (1,500 articles across all three splits) with 18 models from 9 architectural families (1B–70B parameters) and 7 ICL densities (k ∈ {0, 1, 2, 4, 8, 16, 32}). Evaluation uses exact string matching on normalized mentions; the primary metric is micro-F1 (pooled TP/FP/FN across all articles).

### Peak performance per model

| Model | Scale | Best k | F1 Overall | F1 Chem | F1 Dis |
|-------|-------|--------|-----------|---------|--------|
| `meta-llama-3.1-70b-instruct` | 70B | 8 | **0.637** | 0.780 | 0.505 |
| `meta-llama-3.1-8b-instruct` | 8B | 1 | **0.605** | 0.737 | 0.499 |
| `qwen2.5-14b-instruct` | 14B | 8 | 0.589 | 0.731 | 0.462 |
| `llama-3.2-3b-instruct` | 3B | 1 | 0.550 | 0.714 | 0.397 |
| `yi-1.5-9b-chat` | 9B | 2 | 0.537 | 0.652 | 0.455 |
| `qwen2.5-7b-instruct` | 7B | 2 | 0.500 | 0.670 | 0.349 |
| `internlm2-5-7b-chat` | 7B | 1 | 0.480 | 0.646 | 0.337 |
| `phi-3-mini-4k-instruct` | 3.8B | 8 | 0.482 | 0.595 | 0.418 |
| `mistral-7b-instruct-v0.3` | 7B | 2 | 0.456 | 0.630 | 0.322 |
| `phi-3-mini-128k-instruct` | 3.8B | 16 | 0.440 | 0.560 | 0.367 |
| `qwen2.5-3b-instruct` | 3B | 8 | 0.434 | 0.595 | 0.267 |
| `gemma-1.1-2b-it` | 2B | 8 | 0.352 | 0.530 | 0.230 |
| `h2o-danube3-4b-chat` | 4B | 1 | 0.337 | 0.447 | 0.283 |
| `llama-3.2-1b-instruct` | 1B | 1 | 0.282 | 0.380 | 0.206 |
| `qwen2-1.5b-instruct` | 1.5B | 0 | 0.244 | 0.413 | 0.133 |
| `internlm2-5-1-8b-chat` | 8B | 32 | 0.256 | 0.354 | 0.189 |
| `smollm2-1.7b-instruct` | 1.7B | 2 | 0.220 | 0.331 | 0.172 |
| `llama3-openbiollm-8b` | 8B | 0 | 0.083 | 0.144 | 0.049 |

### Key findings

**Scale vs. F1.** There is a log-linear relationship between parameter count and F1, but parameter count is not the sole determinant. `meta-llama-3.1-8b-instruct` (8B) outperforms models with more parameters, including `qwen2.5-14b-instruct` and `yi-1.5-9b-chat`, suggesting that pre-training data quality and instruction tuning carry weight comparable to scale. The step from 8B to 70B yields only 2–3 F1 points overall, making 8B models the Pareto-optimal choice under hardware constraints.

**Chemical vs. Disease asymmetry.** Every model performs better on Chemicals (F1 range 0.14–0.78) than on Diseases (0.05–0.51). Chemical names follow regular morphological and IUPAC patterns (lexical recognition); disease mentions require semantic abstraction and disambiguation. This asymmetry widens under higher ICL densities and is most severe in smaller architectures.

**ICL saturation.** Few-shot examples improve F1 up to a model-specific threshold, after which performance plateaus and then degrades sharply at k=32 in smaller models. `gemma-1.1-2b-it` loses 74.6% of its peak F1 from k=8 to k=32. Models above 7B generally stay within −6% degradation; models in the 1B–2B range are the most sensitive.

**Stability metric Δ.** Defined as (F1_k32 − F1_peak) / F1_peak × 100%, Δ quantifies context-saturation degradation. `llama3-openbiollm-8b` collapses to Δ = −100% because it abandons JSON formatting under dense context — a formatting failure rather than a loss of extraction capacity. `qwen2.5-14b-instruct` achieves Δ = −0.3%, the highest resilience in the study.

**Error profile.** Across all models, false negatives (omission) outnumber false positives (over-extraction), placing most models above the FP=FN diagonal. Smaller architectures and high k values amplify omission bias, particularly in the Disease class.

### F1 vs. number of examples — all models

![Overall F1 vs k-shot](figures/f1_vs_examples_overall_f1.png)

### F1 vs. model size (log scale)

![F1 vs parameters](figures/f1_vs_parameters.png)

### Efficiency frontier (peak F1 vs. parameter count)

![Pareto frontier](figures/pareto_frontier_efficiency.png)

### Chemical vs. Disease F1 gap

![Dumbbell: Chem vs Dis](figures/dumbbell_chem_vs_dis.png)

### Error analysis: false positives vs. false negatives (overall)

![FP vs FN](figures/fp_vs_fn_overall.png)

### Resilience heatmap (Δ = F1_k32 − F1_peak)

![Resilience heatmap](figures/resilience_heatmap.png)

---

## Project structure

```
bioner-llm/
├── config/                    # Prompt templates and few-shot examples
│   ├── config.yaml            # Runtime configuration (model, port, strategy)
│   ├── prompts_type1.yaml     # Prompt templates — Type 1 (with positions)
│   ├── prompts_type2.yaml     # Prompt templates — Type 2 (no positions) ← used in dissertation
│   ├── examples_type1.yaml    # 32 few-shot examples for Type 1
│   └── examples_type2.yaml    # 32 few-shot examples for Type 2
│
├── src/
│   ├── api/                   # FastAPI middleware
│   │   ├── main.py            # App entry point; exposes /extract endpoint
│   │   ├── storage_endpoints.py
│   │   └── audit_endpoints.py
│   ├── audit/
│   │   └── metrics_auditor.py # Audit logging (used by the API)
│   ├── benchmark/
│   │   └── evaluator.py       # BenchmarkEvaluator (position-based IoU matching)
│   ├── consensus/
│   │   └── consensus_engine.py # Multi-LLM consensus (voting, weighted, cascade)
│   ├── llm/
│   │   ├── llm_manager.py     # Single-LLM client (vLLM OpenAI-compatible API)
│   │   ├── multi_llm_manager.py
│   │   └── huggingface_manager.py
│   ├── models/
│   │   └── schemas.py         # Pydantic schemas (Entity, Metrics, etc.)
│   ├── prompts/
│   │   └── prompt_engine.py   # Builds prompts from YAML templates + examples
│   └── storage/
│       └── response_storage.py # Saves per-article JSON extractions
│
├── preprocessing/             # BC5CDR dataset preparation (run once)
│   ├── text_to_df.py          # Parses PubTator → CSV (used to generate gold + input)
│   ├── create_combined_cdr_dataset.py  # Parses all three BC5CDR splits into one CSV
│   └── create_validation_datasets.py
│
├── pipeline/                  # Experiment execution
│   ├── llm_sender.py          # Sends articles to the API; saves extraction JSONs
│   ├── run_multiple_examples.py  # Runs llm_sender for k ∈ {0,1,2,4,8,16,32}
│   ├── api_launcher.py        # Launches the FastAPI server in a background thread
│   ├── indicios_to_df.py      # Converts extraction JSONs → CSV (one row per article)
│   ├── get_results.py         # Computes P/R/F1 vs gold; writes results.txt
│   ├── check_missing_pmids.py # Reports which PMIDs are missing for a model/strategy
│   ├── remove_duplicates_pmids.py
│   └── verify_and_clean_pmids.py
│
├── figures/                   # Result plots (committed; generated by generate_plots.py)
│
├── scripts/
│   └── token_analysis/        # Token count analysis across models and strategies
│
├── vllm/                      # HPC cluster scripts
│   ├── example_sbatch.sh      # Annotated SBATCH script — copy and adapt per model
│   ├── vllm.def               # Singularity image definition
│   └── llama32_chat_template.jinja
│
├── run_api.py                 # Start the FastAPI middleware locally
├── generate_plots.py          # Generate figures from results.txt files
└── requirements.txt
```

## Examples

The `examples/` folder contains minimal working samples that mirror the real pipeline output — useful for understanding the expected file formats before running the full experiment.

```
examples/
├── indicios_encontrados_exemplo/        # Extraction JSONs (output of llm_sender.py)
│   └── modelo_exemplo/
│       └── type2/
│           ├── zero_shot/               # 3 articles, imperfect predictions (FP/FN present)
│           │   ├── extraction_20250101_120000_2004_0.json
│           │   ├── extraction_20250101_120018_26094_0.json
│           │   └── extraction_20250101_120035_56789_0.json
│           └── examples_1/             # same 3 articles, perfect predictions
│               ├── extraction_20250101_130000_2004_0.json
│               ├── extraction_20250101_130018_26094_0.json
│               └── extraction_20250101_130035_56789_0.json
└── dataset_exemplo/                     # Processed outputs (output of indicios_to_df + get_results)
    └── modelo_exemplo/
        └── type2/
            ├── inferencias/             # One CSV per strategy (one row per article)
            │   ├── modelo_exemplo_zero_shot.csv
            │   └── modelo_exemplo_examples_1.csv
            ├── comparison/              # Gold vs. predicted comparison with TP/FP/FN
            │   └── comparison_zero_shot.csv
            └── results.txt             # P/R/F1 report (macro + micro) across strategies
```

Each extraction JSON contains the predicted entities for one article:

```json
{
  "pmid": "2004",
  "entities": {
    "chemicals": [{"text": "thioridazine", "type": "Chemical"}],
    "diseases":  [{"text": "ventricular tachycardia", "type": "Disease"}]
  },
  "model": "org/modelo-exemplo",
  "prompt_strategy": "zero-shot",
  "num_examples": 0
}
```

## Data files

`data/cdr_gold.csv` is included in this repository — it contains the gold-standard annotations for all 1,500 BC5CDR articles (chemicals and diseases per article) and is the reference used for all evaluation.

The following files must be generated locally before running the pipeline:

| File | Status | Generated by | Description |
|------|--------|-------------|-------------|
| `data/cdr_gold.csv` | **tracked** | `preprocessing/create_combined_cdr_dataset.py` | Gold standard with annotated entities (all 1,500 articles) |
| `data/cdr_ner_dataset.csv` | not tracked | `preprocessing/text_to_df.py` or `create_combined_cdr_dataset.py` | Article texts sent to the API for extraction (all 1,500 articles) |
| `indicios_encontrados/<model>/<type>/<strategy>/` | not tracked | `pipeline/llm_sender.py` | Per-article extraction JSONs |
| `dataset/<model>/<type>/` | not tracked | `pipeline/get_results.py` | Comparison CSVs + results.txt |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the experiment

### 1. Prepare the dataset

`data/cdr_gold.csv` is already included in this repository. To run the full extraction pipeline you also need `data/cdr_ner_dataset.csv`, which contains only the article texts (no annotations) and must be generated from the raw BC5CDR PubTator files. Download them from the [BioCreative V CDR task page](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/).

Two approaches are available depending on how the BC5CDR files are organised locally.

**Option A — single combined file (all three splits at once):**

```bash
python preprocessing/create_combined_cdr_dataset.py \
    /path/to/CDR_TrainingSet.PubTator \
    /path/to/CDR_DevelopmentSet.PubTator \
    /path/to/CDR_TestSet.PubTator \
    data/cdr_gold.csv \
    data/cdr_ner_dataset.csv
```

**Option B — one split at a time, then concatenate:**

```bash
python preprocessing/text_to_df.py /path/to/CDR_TrainingSet.PubTator    data/train.csv
python preprocessing/text_to_df.py /path/to/CDR_DevelopmentSet.PubTator data/dev.csv
python preprocessing/text_to_df.py /path/to/CDR_TestSet.PubTator        data/test.csv
# then concatenate train.csv + dev.csv + test.csv into cdr_gold.csv and cdr_ner_dataset.csv
```

`cdr_gold.csv` contains the gold annotations (chemicals and diseases per article) used for evaluation. `cdr_ner_dataset.csv` contains only the article texts sent to the API for extraction — both cover all 1,500 articles.

### 2. Start the vLLM server (HPC)

Models were served on [NPAD/UFRN](https://npad.ufrn.br/) (Núcleo de Processamento de Alto Desempenho da Universidade Federal do Rio Grande do Norte), a SLURM-managed cluster. vLLM runs inside a Singularity container; each model is a separate SBATCH job on a dedicated port.

Copy `vllm/example_sbatch.sh`, set `MODEL` and `PORT`, and submit:

```bash
#SBATCH --partition=gpu-8-h100
#SBATCH --gres=gpu:1

MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
PORT=8000
SIF=/path/to/vllm.sif

singularity exec --nv "$SIF" \
    vllm serve $MODEL --host 0.0.0.0 --port $PORT \
    --max-model-len 16384 --dtype half
```

```bash
# Submit
sbatch vllm/example_sbatch.sh   # adapt MODEL, PORT and SIF first

# Forward the port to your local machine
autossh -M 0 -N -L <port>:<node-ip>:<port> -p <ssh-port> <user>@<cluster>
```

See [`vllm/README.md`](vllm/README.md) for the full annotated script, GPU memory guidelines, and troubleshooting.

### 3. Configure and start the API middleware

Edit `config/config.yaml` to point to the running model:

```yaml
llm:
  base_url: "http://localhost:<port>/v1"
  model_name: "<huggingface-model-id>"
```

```bash
python run_api.py
```

### 4. Run the extraction

```bash
# Run all k-shot strategies (k = 0, 1, 2, 4, 8, 16, 32) for one model
python pipeline/run_multiple_examples.py \
    --model <model-name> \
    --prompt-type type2 \
    --input data/cdr_ner_dataset.csv
```

Extractions are saved as JSON files under `indicios_encontrados/<model>/type2/<strategy>/`.

### 5. Compute metrics

```bash
python pipeline/get_results.py --model <model-name> --prompt-type type2
```

Outputs per-strategy P/R/F1 to `dataset/<model>/type2/results.txt`.

### 6. Generate plots

```bash
python generate_plots.py
```

Plots are saved to `figures/`.

## Evaluation methodology

Evaluation is performed in `pipeline/get_results.py` using **exact string matching** on normalized entity mentions (lowercased, stripped). This follows the standard mention-level NER evaluation used in BC5CDR benchmarks.

For each article and each entity type (Chemical, Disease):

```
TP = |gold ∩ predicted|    (exact normalized text match)
FP = |predicted \ gold|
FN = |gold \ predicted|

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 · P · R / (P + R)
```

Two aggregation modes are reported in `results.txt`:

- **Macro (Média por Artigo)** — per-article P/R/F1, then mean across articles (macro-average)
- **Micro (Agregado)** — global sum of TP/FP/FN across all articles, then P/R/F1 (micro-average)

The primary result used throughout is **micro-F1 (Agregado)**.

## Prompt strategies

| Strategy | k (examples) | Template key |
|----------|-------------|-------------|
| Zero-shot | 0 | `zero_shot_no_positions` |
| Few-shot | 1, 2, 4, 8, 16, 32 | `few_shot_<k>_no_positions` |

All strategies use **Type 2** templates (`config/prompts_type2.yaml`), which:
- Request only entity text and type (no character positions)
- Include generic drug classes (e.g., "chemotherapy") as Chemical
- Exclude general symptoms and biological processes from Disease
- Are aligned with BC5CDR annotation guidelines

Examples are loaded deterministically from `config/examples_type2.yaml` (32 total, not drawn from BC5CDR).

## Models evaluated

| Model | Parameters | Family |
|-------|-----------|--------|
| SmolLM2-1.7B-Instruct | 1.7 B | SmolLM |
| Llama-3.2-1B-Instruct | 1 B | Llama |
| Qwen2-1.5B-Instruct | 1.5 B | Qwen |
| Qwen2.5-3B-Instruct | 3 B | Qwen |
| Llama-3.2-3B-Instruct | 3 B | Llama |
| H2O-Danube3-4B-Chat | 4 B | H2O Danube |
| Phi-3-Mini-4K-Instruct | 3.8 B | Phi-3 |
| Phi-3-Mini-128K-Instruct | 3.8 B | Phi-3 |
| InternLM2.5-1.8B-Chat | 1.8 B | InternLM |
| InternLM2.5-7B-Chat | 7 B | InternLM |
| Mistral-7B-Instruct-v0.3 | 7 B | Mistral |
| Meta-Llama-3.1-8B-Instruct | 8 B | Llama |
| Llama3-OpenBioLLM-8B | 8 B | Llama (biomedical) |
| Yi-1.5-9B-Chat | 9 B | Yi |
| Qwen2.5-7B-Instruct | 7 B | Qwen |
| Qwen2.5-14B-Instruct | 14 B | Qwen |
| Qwen2.5-32B-Instruct | 32 B | Qwen |
| Meta-Llama-3.1-70B-Instruct | 70 B | Llama |

## Dataset

[BC5CDR](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860626/) (BioCreative V Chemical-Disease Relation) — 1,500 PubMed abstracts annotated with Chemical and Disease mentions, split into training, development, and test sets of 500 articles each. All three splits are used in this experiment.

The gold-standard annotations (`data/cdr_gold.csv`) are included in this repository. The raw PubTator files are not — download them from the BioCreative V CDR task page and follow the instructions in [Prepare the dataset](#1-prepare-the-dataset) to generate `cdr_ner_dataset.csv`.

