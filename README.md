# Floating on Main: PinCheck

PinCheck finds Hugging Face model loads in Python code that are not pinned to a fixed revision.

```python
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")                  # floating: follows `main`
model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", revision="3f2a…")  # pinned to a commit
```

A floating load downloads whatever the model's owner most recently pushed. Two builds of
unchanged code can therefore run different models. This repository contains PinCheck and
everything needed to reproduce the paper:

> A. V. Khetawat, R. Sharma, S. Arora, and M. Rai, "Floating on Main: Detecting Unpinned Model
> Dependencies in MLOps," in *Proc. ICon INDIA 2026*.

## Install

PinCheck needs Python 3.9 or newer and has no third-party dependencies.

```bash
pip install "git+https://github.com/Aditya-Khetawat/floating-on-main.git"
# or, from a clone:
pip install -e .
```

## Use

```bash
pincheck src/                      # scan a directory (Python files and notebooks)
pincheck app.py --all              # also list call sites that are fine
pincheck src/ --format json        # machine-readable output
pincheck src/ --fail-on remote-code,default-pipeline
```

Example output:

```
src/encoder.py:48: [unpinned] from_pretrained(<indirect>) revision=none
src/rag.py:12: [unpinned,remote-code] from_pretrained(org/model) revision=none
pincheck: 37 model-loading call site(s), 2 failing [default-pipeline, remote-code, unpinned]
```

PinCheck exits with status 1 if any call site violates a selected rule and with status 0 otherwise.

| Rule | Fails when |
|---|---|
| `unpinned` | a hub load has no `revision` at all |
| `remote-code` | `trust_remote_code=True` is used without a commit-hash revision |
| `default-pipeline` | `pipeline(task)` is called without a model, so the library chooses one |
| `named-ref` | the revision is a branch or tag such as `"main"` (mutable); off by default |

The default is `--fail-on unpinned,remote-code,default-pipeline`. Local paths
(for example `"./checkpoints"`) are never reported. To accept one call site deliberately,
add `# pincheck: ignore` to the line of the call.

### What PinCheck detects

| Interface | Library |
|---|---|
| `from_pretrained` | Transformers, Diffusers, PEFT and other `from_pretrained` APIs |
| `pipeline` | Transformers (only when imported from `transformers`) |
| `SentenceTransformer`, `CrossEncoder` | Sentence-Transformers |
| `hf_hub_download`, `snapshot_download` | huggingface_hub |
| `HuggingFaceEmbeddings` | LangChain (including `model_kwargs={"revision": …}`) |

PinCheck parses each file into an abstract syntax tree, so it ignores comments, docstrings and
function definitions, and it finds `revision=` anywhere in a multi-line call. It resolves
identifiers that are stored in a variable assigned a single string in the same file
(for example `MODEL_ID = "org/model"`).

**Limitations:** it does not follow values across files or into configuration files,
environment variables or command-line arguments; such identifiers are reported as
`<indirect>`. Code embedded inside string literals (templates, `python -c` scripts) is not
analyzed.

### In continuous integration

Copy [`examples/github-actions-pincheck.yml`](examples/github-actions-pincheck.yml) to
`.github/workflows/` in your project. To pin models without hard-coding hashes, see the
lockfile example in [`examples/lockfile/`](examples/lockfile/) (Fig. 5 of the paper):
`lock_models.py` records each model's current commit hash in `models.lock`, and `loader.py` loads
every model at its recorded commit. These two example scripts need `huggingface_hub` and
`transformers` and are not part of the automated tests.

## Reproduce the paper

```bash
pip install -r requirements.txt
pytest tests/                                   # 17 unit tests

python study/04_evaluate_detectors.py --split test   # RQ1, Tables II and III (needs step 3 first)
python study/05_statistics.py                   # RQ2-RQ5 and robustness -> output/summary.json
python study/06_figures.py                      # Figs. 3 and 4
```

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `01_collect_candidates.py` | GitHub Search API | `output/candidates.json` |
| 2 | `02_analyze_repos.py` | candidate list | `output/results.jsonl` |
| 3 | `03_fetch_eval_samples.py` | `data/eval_samples.json` | `output/evalfiles/{dev,test}/` |
| 4 | `04_evaluate_detectors.py` | evaluation files | `output/rq1_*.json`, Tables II-III |
| 5 | `05_statistics.py` | `data/results.jsonl` | `output/summary.json` |
| 6 | `06_figures.py` | `output/summary.json` | Figs. 3 and 4 |

Steps 5 and 6 reproduce every number in Section V exactly from the frozen data in `data/`.
Steps 1-3 query GitHub live. Because repositories keep changing, re-running them gives a new
snapshot rather than the paper's; step 3 reports how many evaluation files still match the
hashes recorded at analysis time. The paper's snapshot was analyzed on 20 October 2026, and the
candidate list was collected on 5 October 2026.

**Note on two PinCheck settings.** The held-out evaluation (RQ1) used the version of PinCheck
that was frozen before evaluation. That version counted any bare `pipeline(...)` call in a file
mentioning Transformers. The evaluation exposed three calls to a project's own `pipeline`
function, so the rule was tightened to require an import from Transformers before RQ2-RQ5 were
computed. `04_evaluate_detectors.py` therefore runs PinCheck with `pipeline_rule="legacy"`;
everything else, and the command-line tool, uses the corrected rule.

### Data (`data/`)

| File | Contents |
|---|---|
| `candidates.json` | 1,150 candidate repositories from GitHub search (5 Oct 2026) |
| `results.jsonl` | one line per analyzed repository (939): call sites, dependency declarations, CI, tests, Docker and lockfile signals |
| `eval_samples.json` | development (seed 2026) and held-out (seed 7331) samples: repositories, file paths and SHA-256 hashes |
| `adjudication_test.csv` | every disagreement between PinCheck and the baseline on the held-out sample, its cause and verdict, and whether it was inspected manually; plus the 30 inspected agreements |

The evaluation source files are third-party code and are not redistributed; step 3 downloads them.

## License

MIT, see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).
