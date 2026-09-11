# The Guillotine — a semantic association benchmark for LLMs

A pilot study that puts 30 items in the format of the Italian TV game show "La Ghigliottina" ("The Guillotine": 5 English clue words → a single word connecting all of them) to 127 language models — 111 at controlled temperature (local via Ollama + open-weight via OpenRouter) and 16 top commercial models (Claude, GPT, Gemini, Grok, Kimi, among others) — to compare their semantic pattern-recognition ability against the judgment of a single human author (N=1, explicitly stated as a limitation, not hidden).

The full essay, with the extended discussion of results and the philosophical reflection sections, is in `documentation/` (Italian and English) and will also be published on LessWrong and LinkedIn — links will be added here once available.

## Research questions

1. Does the author's solution match the one found by the models?
2. Quantitatively, how far are the models' answers from the reference solution?
3. In the word association, can a model's answer turn out to be more coherent than the author's own (subjective linguistic judgment)?
4. How much does a model's answer change across the temperature spectrum (T=0.0/0.3/0.7), with top_k/top_p held constant? — *not* a fixed-parameter reproducibility test, which would require an additional run at identical T, not included in this design.
5. Does model size (parameter count) correlate with association accuracy?
6. Which model gives the best answer, in the author's judgment?

## Some results (full details in `documentation/`)

- The most reliable local/open-weight model is `stepfun/step-3.5-flash` (71.1% exact match, 95% Wilson CI [61.0%, 79.5%], over 90 observations) — not the nominally highest-ranked one, which has only 4 valid observations out of 90.
- Among commercial models, `x-ai/grok-4.6` leads with 76.7% (95% CI [59.1%, 88.2%], N=30). Among the top 8 models in each group, **no pair remains significantly different** after FDR correction (McNemar) — the exact ranking position at the top should not be over-interpreted.
- Temperature shifts absolute accuracy by a few points for a subset of models (9 out of the testable ones, without correction for multiple comparisons), but **does not flip the ranking** between models (Spearman between rankings at different T: 0.886–0.930).
- Model size correlates with accuracy (Spearman rho=0.756 on verified total parameters, rho=0.690 on active parameters per token, n=45) — but doesn't explain everything: `gpt-oss:20b` (3.61B active parameters per token, MoE architecture) is among the best in the roster with one of the smallest compute budgets.
- The three independent item-difficulty measures (author-declared label, empirical cluster on accuracy, empirical cluster on generation time for reasoning models) agree only partially with each other (weighted Kappa: label↔accuracy 0.122, label↔time 0.145, accuracy↔time 0.325) — the two measures derived from model behavior agree with each other more than either agrees with human judgment.
- Qualitative behaviors documented in detail: a model (`starcoder2:15b-instruct`) that in half its answers keeps generating Python code unrelated to the task after answering; cases of reasoning models that find the correct answer within their own reasoning and then abandon it (including one case that oscillates more than ten times between two answers before getting it wrong).

## Repository structure

```
guillotine/
├── source/                  # input data (raw results are NOT included, see below)
│   ├── guillotine.xlsx          # dataset of the 30 items (source of truth, hand-editable)
│   └── guillotine_prompts.json  # same dataset, converted to JSON for the scripts
├── python/                  # data collection pipeline
├── notebook/                # statistical analysis (Jupyter)
│   ├── ghigliottina_analisi_locali_13[_eng].ipynb
│   ├── ghigliottina_analisi_commerciali_01[_eng].ipynb
│   └── ghigliottina_tempo_token_ragionamento_02[_eng].ipynb
├── documentation/           # full essay (Italian + English)
├── requirements.txt         # collection pipeline dependencies
└── .env.example             # configuration variables (copy to .env)
```

### Data included and excluded

To keep the repository light and avoid redistributing thousands of raw commercial-model responses (whose terms of service on reuse/redistribution aren't always clear), **only the two input files are included**:

- `source/guillotine.xlsx` — the 30 original items (clues, expected solution, difficulty note)
- `source/guillotine_prompts.json` — the same data, JSON format, used directly by the scripts and notebooks

**Not included**: the raw-results workbooks (`complete_ghigliottina_results.xlsx` and its intermediate versions), the collection logs (`*.log`), the CSV files produced by the notebooks (`analisi_leaderboard.csv`, `analisi_per_item.csv`, `analisi_temperatura.csv` — regenerable by re-running the notebooks), and the raw JSONL logs of the OpenRouter calls (`raw_logs/`, needed only to rebuild the `ghigliottina_tempo_token_ragionamento_02` notebook from scratch — it uses results already computed and saved in the notebook's own cells).

**Practical consequence**: the three notebooks open and read perfectly as they are (the cells already contain the results and figures from the original run) — but to *re-run* them from scratch against new data, `source/complete_ghigliottina_results.xlsx` needs to be regenerated first with the pipeline below, or the full dataset requested from the author.

### Collection pipeline (`python/`)

| Script | What it does |
|---|---|
| `config.py` | Central configuration: paths, list of local models (`LOCAL_MODELS`) and OpenRouter models (`OPENROUTER_MODELS`, `OPENROUTER_COMMERCIAL`), sampling parameters, output column schema. No hardcoded credentials — API keys live only in `.env` |
| `excel_to_json.py` | Converts `guillotine.xlsx` (hand-validated) into `guillotine_prompts.json`, with a sanity check that flags if the solution accidentally appears as a substring in a clue |
| `generate_template.py` | Creates the empty/pre-filled results workbook (4 sheets: `T_0.0`/`T_0.3`/`T_0.7` for the controlled-temperature models, `Commercial` pre-filled for manual collection) |
| `oldgenerate_template.py` | Earlier version of `generate_template.py`, superseded — kept only for historical reference, do not use |
| `ollama_client.py` | HTTP client for Ollama's `/api/generate` (local models) |
| `openrouter_client.py` | HTTP client for OpenRouter's chat endpoint (open-weight models on remote hardware + commercial models); handles rate limiting (5/10/20s backoff on 429), retries, and a raw JSONL backup of every call before parsing |
| `batch_runner.py` | CLI orchestrator for local/OpenRouter models at controlled temperature: for each model, iterates 3 temperatures × 30 prompts, saves incrementally, unloads the model from VRAM. Resumable: on restart it skips (model, prompt) combinations already completed |
| `commercial_runner.py` | Same principle for commercial models: a single run per model/prompt, no sampling parameters set (the provider uses its own default) |
| `response_parser.py` | Extracts the final word from the raw response (handles quoted text, `<think>` reasoning traces, answer-then-explanation and vice versa) |
| `scoring.py` | The three automatic metrics: `exact_match`, `same_root` (stemming), `is_similar_lev` (normalized Levenshtein) |
| `excel_writer.py` | Incremental writing to Excel with anti-duplicate upsert; the workbook itself is the resume checkpoint |
| `merge_log_timings.py` | Utility to reconcile generation times from the logs when missing from the main workbook |
| `rescore.py` | Recomputes the automatic metrics on an already-collected workbook, without re-running the models (used after manual corrections to `predicted_word`) |

### Analysis notebooks (`notebook/`)

Each notebook exists in two versions with identical results: the original Italian one and an English translation (`_eng`, same data, same code, chart titles/labels and comments in English) — also used to generate the essay's English-language figures.

| Notebook | Covers |
|---|---|
| `ghigliottina_analisi_locali_13` | RQ1–RQ6 on the 111 controlled-temperature models: data loading/deduplication, coverage check, leaderboard with Wilson CIs, temperature effect (Cochran's Q), size/accuracy correlation (RQ5, with and without verified parameters), calibration and empirical classification of item difficulty (k-means + silhouette), convergent validity (weighted Kappa), Zipf lexical-rarity analysis |
| `ghigliottina_analisi_commerciali_01` | Same statistical scaffolding on the 16 commercial models (single run, N=30) — without RQ4 (no temperature axis) or RQ5 (flagship proprietary models don't disclose parameter counts) |
| `ghigliottina_tempo_token_ragionamento_02` | Analysis of the raw OpenRouter logs (not the Excel file): native reasoning tokens per provider, inference speed, characters-per-token ratio as a transparency check on the reasoning summaries shown by some providers |

## How to reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # customize GHIGLIOTTINA_DATA_DIR/OLLAMA_HOST if needed
```

Local models require [Ollama](https://ollama.com) running (`ollama serve`); OpenRouter/commercial models require an API key in `.env` (`OPENROUTER_API_KEY=...`, not included in `.env.example` to avoid suggesting a credential format).

```bash
# 1. Dataset -> JSON
python python/excel_to_json.py source/guillotine.xlsx source/guillotine_prompts.json

# 2. Create the empty workbook
python python/generate_template.py source/guillotine_prompts.json source/ghigliottina_results.xlsx

# 3. Collection — local/OpenRouter models at controlled temperature (resumable)
python python/batch_runner.py
python python/batch_runner.py --models mistral-small:22b --limit-prompts 3   # smoke test

# 4. Collection — commercial models (single run, no temperature)
python python/commercial_runner.py
```

For the notebooks (not covered by `requirements.txt`, which only covers the collection pipeline):

```bash
pip install jupyter pandas numpy scipy matplotlib scikit-learn statsmodels nltk rapidfuzz wordfreq
```

The prompt used, identical for every model:

```
You are playing a word-association game. You will be given 5 clue
words. Find the single word that connects all five (through common
phrases, idioms, or shared concepts). Respond with ONLY the final
word, in uppercase, with no explanation. Clues: {word1}, {word2},
{word3}, {word4}, {word5}.
```

## Main limitations

- Solution, expected difficulty, and linguistic-coherence judgment are all set by a single author (N=1), stated as such, not a hidden limitation.
- The pairwise comparison between the top models (McNemar + FDR correction) finds no statistically significant differences at the top of the leaderboard, in either group — the exact ranking position of the top models should not be over-interpreted.
- Commercial models have a single run per item, with no control over sampling parameters (a cost choice, not a technical impossibility) — comparable to the controlled-temperature models only on RQ1/2/3/5/6, not on RQ4.
- Automatic extraction of the final answer from the raw text is heuristic (`format_compliant` flags at-risk cases without discarding them); manual review of non-exact answers (`synonym_manual`, `coherence_manual`) is nonetheless 100% complete.
- 3 of the 30 original items were replaced after an initial methodological review (they measured a spelling pattern or a category label, not a semantic association) — the dataset version in this repo is the final one.

## License

The code in this repository (`python/`, `source/*.py`, notebooks included) is distributed under the **MIT** license — see [LICENSE](LICENSE).

The material in `documentation/` (the essay and its supporting analysis) is distributed under **[Creative Commons Attribution 4.0 (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**: anyone can use, modify, and redistribute it, including commercially, provided the author is credited and/or this repository is linked.

## How to cite

If this work is useful to you, please credit the author and/or link this repository. A more formal citation reference (with a link to the published essay) will be added here once available.
