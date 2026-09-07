# Multimodal Running Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, measure, and visualize a real six-operator COCO image--text workload whose optimizer-generated staged and joint plans support Figure 2 and whose operator input sensitivity supports Figure 5.

**Architecture:** A deterministic fixture builder partitions COCO into calibration, pilot, formal, and scaling splits. A Cedar `Feature` composes two CPU text operators, one CPU image operator, and three CUDA-Ray model operators; a separate experiment package calibrates thresholds, runs the declared pilot grid, freezes one configuration, performs round-robin formal execution, audits Cedar/PICO scores, and generates plots solely from archived artifacts.

**Tech Stack:** Python 3, Cedar, PyTorch 2.0.1/CUDA 11.8, Hugging Face Transformers, Pillow/OpenCV, NumPy, pytest, Matplotlib, Bash/nohup, LaTeX.

**Spec:** `docs/superpowers/specs/2026-09-07-multimodal-running-example-design.md`

## Global Constraints

- Run every Python command inside `optimalcedar-torch201-dev` after `cd /workspace/OptimalCedar && source env/bin/activate`.
- Use COCO val2017 records 0--499 for calibration, 500--999 for pilot, 1000--3999 for formal comparison, and 4000--4999 for scaling after sorting by `(image_id, caption_id)`.
- Use one shared ten-second profile per selected configuration, `W=8`, `CPU_BUDGET=64`, one RTX A6000, cache disabled, and three formal round-robin repetitions.
- Use model revisions CLIP `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, BLIP `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`, and aesthetic `684098de3856fa4678bf800efc05635de5b6cde5`.
- The 27-point pilot grid is the only automatic parameter search; do not alter optimizer code, profile statistics, resource budgets, model revisions, or formal data after observing results.
- Generate figures only from archived JSON/CSV/plan files. Do not embed experimental values in plotting code.
- Do not modify Chapter 6 or existing workload results.

---

### Task 1: Deterministic COCO fixture and split manifest

**Files:**
- Create: `evaluation/pipelines/multimodal_running_example/__init__.py`
- Create: `evaluation/pipelines/multimodal_running_example/fixture.py`
- Create: `tests/test_multimodal_running_example_fixture.py`

**Interfaces:**
- Consumes: `datasets/coco/annotations/captions_val2017.json` and `datasets/coco/val2017/`.
- Produces: `build_fixture(annotation_path: Path, image_root: Path, output_dir: Path) -> FixtureManifest` and JSONL files `calibration.jsonl`, `pilot.jsonl`, `formal.jsonl`, `scaling.jsonl`.

- [ ] **Step 1: Write failing split and checksum tests**

```python
def test_partition_boundaries_are_disjoint(tmp_path, mini_coco):
    manifest = build_fixture(*mini_coco, output_dir=tmp_path, split_sizes=(2, 2, 3, 1))
    ids = [set(read_record_ids(tmp_path / f"{name}.jsonl")) for name in manifest.split_names]
    assert [len(x) for x in ids] == [2, 2, 3, 1]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1 :])

def test_fixture_is_byte_reproducible(tmp_path, mini_coco):
    first = build_fixture(*mini_coco, output_dir=tmp_path / "a", split_sizes=(2, 2, 3, 1))
    second = build_fixture(*mini_coco, output_dir=tmp_path / "b", split_sizes=(2, 2, 3, 1))
    assert first.output_sha256 == second.output_sha256
```

- [ ] **Step 2: Run the fixture tests and verify import failure**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_multimodal_running_example_fixture.py'
```

Expected: failure because `fixture.py` and its interfaces do not exist.

- [ ] **Step 3: Implement stable record ordering, partitioning, and hashing**

```python
@dataclass(frozen=True)
class FixtureManifest:
    annotation_sha256: str
    output_sha256: str
    splits: dict[str, int]
    split_names: tuple[str, ...] = ("calibration", "pilot", "formal", "scaling")

def build_fixture(annotation_path: Path, image_root: Path, output_dir: Path,
                  split_sizes: tuple[int, int, int, int] = (500, 500, 3000, 1000)) -> FixtureManifest:
    annotations = json.loads(annotation_path.read_text())["annotations"]
    records = sorted(annotations, key=lambda x: (int(x["image_id"]), int(x["id"])))
    # Materialize exactly sum(split_sizes) records with stable relative image paths.
```

Each JSONL record contains `record_id`, `image_id`, `caption_id`, `image_path`, `caption`, `width`, and `height`. Hash bytes in split-name order and write `manifest.json` atomically.

- [ ] **Step 4: Run focused tests and materialize the real fixture**

Run the pytest command from Step 2, then:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && python -m evaluation.pipelines.multimodal_running_example.fixture --annotations datasets/coco/annotations/captions_val2017.json --image-root datasets/coco/val2017 --output outputs/motivation_multimodal/fixture'
```

Expected: tests pass; `manifest.json` reports split counts `500/500/3000/1000` and every referenced image exists.

- [ ] **Step 5: Commit the fixture unit**

```bash
git add evaluation/pipelines/multimodal_running_example tests/test_multimodal_running_example_fixture.py
git commit -m "feat: add deterministic multimodal COCO fixture"
```

### Task 2: Real CPU and CUDA operator implementations

**Files:**
- Create: `evaluation/pipelines/multimodal_running_example/operators.py`
- Create: `tests/test_multimodal_running_example_operators.py`

**Interfaces:**
- Consumes: fixture records from Task 1 and cached immutable model snapshots.
- Produces: `TextNormalizer`, `PerplexityPredicate`, `SharpnessPredicate`, `AestheticPredicate`, `ClipPredicate`, `BlipPredicate`; every predicate exposes `score(record) -> float` and `__call__(record) -> bool`.

- [ ] **Step 1: Write failing CPU modality-isolation tests**

```python
def test_text_operators_ignore_image_changes(sample_record):
    changed = {**sample_record, "image_path": "missing-but-never-opened.jpg"}
    assert TextNormalizer()(sample_record)["caption"] == TextNormalizer()(changed)["caption"]
    assert PerplexityPredicate(max_score=float("inf")).score(sample_record) == \
           PerplexityPredicate(max_score=float("inf")).score(changed)

def test_sharpness_ignores_caption(sample_record):
    changed = {**sample_record, "caption": "unrelated text"}
    assert SharpnessPredicate(min_score=0).score(sample_record) == \
           SharpnessPredicate(min_score=0).score(changed)
```

- [ ] **Step 2: Run tests and verify missing implementations**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_multimodal_running_example_operators.py -k "text or sharpness"'
```

Expected: failure on missing classes.

- [ ] **Step 3: Implement CPU operators and deterministic score/threshold separation**

Reuse `FixUnicodeMapper`, `PunctuationNormalizationMapper`, and `PerplexityFilter.inner` behavior from `evaluation/pipelines/llava_pretrain/dj_operators.py`. Implement sharpness as grayscale OpenCV `Laplacian(..., CV_64F).var()`. Operators copy the record before adding normalized text or score fields; score functions never filter.

- [ ] **Step 4: Run CPU tests and verify pass**

Run the command from Step 2. Expected: all selected tests pass without initializing CUDA.

- [ ] **Step 5: Write failing offline model and resource tests**

```python
@pytest.mark.gpu
def test_gpu_scores_are_finite(sample_record):
    for op in (AestheticPredicate(0), ClipPredicate(0), BlipPredicate(0)):
        assert math.isfinite(op.score(sample_record))

def test_model_revisions_are_frozen():
    assert MODEL_REVISIONS == {
        "clip": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "blip": "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c",
        "aesthetic": "684098de3856fa4678bf800efc05635de5b6cde5",
    }
```

- [ ] **Step 6: Implement lazy, offline-only CUDA model wrappers**

Resolve snapshots with `huggingface_hub.snapshot_download(..., revision=..., local_files_only=True)`. Load each model lazily inside the executing process, call `.eval().cuda()`, use its matching processor, wrap inference in `torch.inference_mode()`, and return the same normalized score semantics used by Data-Juicer. Do not load a model in the parent process before Ray actors start.

- [ ] **Step 7: Run all operator tests**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_multimodal_running_example_operators.py'
```

Expected: CPU and single-GPU tests pass; no network request appears in the log.

- [ ] **Step 8: Commit operators**

```bash
git add evaluation/pipelines/multimodal_running_example/operators.py tests/test_multimodal_running_example_operators.py
git commit -m "feat: add multimodal curation operators"
```

### Task 3: Cedar feature, dependencies, and semantic-equivalence checks

**Files:**
- Create: `evaluation/pipelines/multimodal_running_example/cedar_dataset.py`
- Create: `tests/test_multimodal_running_example_feature.py`

**Interfaces:**
- Consumes: six operators from Task 2 and a thresholds JSON path supplied through `CedarEvalSpec.kwargs`.
- Produces: `MultimodalRunningExampleFeature`, `get_dataset(spec: CedarEvalSpec) -> DataSet`, and `logical_signature(feature) -> dict[str, object]`.

- [ ] **Step 1: Write failing graph/resource tests**

```python
def test_feature_has_six_operators_and_declared_constraints(feature):
    sig = logical_signature(feature)
    assert sig["tags"] == ["normalize", "perplexity", "sharpness", "aesthetic", "clip", "blip"]
    assert set(sig["dependencies"]) == {("normalize", "perplexity"), ("sharpness", "aesthetic"),
                                        ("perplexity", "clip"), ("aesthetic", "clip"), ("clip", "blip")}
    assert sig["cuda_tags"] == ["aesthetic", "clip", "blip"]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_multimodal_running_example_feature.py'
```

Expected: failure because the feature does not exist.

- [ ] **Step 3: Implement Feature composition and optimizer options**

Compose `MapperPipe(N)`, `FilterPipe(P)`, `FilterPipe(Q)`, `FilterPipe(A)`, `FilterPipe(C)`, `FilterPipe(B)`. Apply `.depends_on()` exactly as specified and mark `A/C/B` with `PipeExecutionResource.CUDA`. Keep cache disabled and pass `W=8`, CPU budget 64, and optimizer selection through the existing `CedarEvalSpec` pathway.

- [ ] **Step 4: Add legal-order output-equivalence test**

Evaluate CPU score fixtures and precomputed GPU calibration scores through representative legal orders `N,P,Q,A,C,B` and `Q,A,N,P,C,B`; assert identical retained `record_id` sets. This test validates predicate commutativity without loading three models repeatedly.

- [ ] **Step 5: Run feature tests and commit**

Run the Step 2 command. Expected: all tests pass.

```bash
git add evaluation/pipelines/multimodal_running_example/cedar_dataset.py tests/test_multimodal_running_example_feature.py
git commit -m "feat: compose multimodal Cedar workload"
```

### Task 4: Calibration and bounded pilot-grid selection

**Files:**
- Create: `evaluation/motivation_multimodal/__init__.py`
- Create: `evaluation/motivation_multimodal/calibrate.py`
- Create: `evaluation/motivation_multimodal/pilot.py`
- Create: `evaluation/motivation_multimodal/artifacts.py`
- Create: `tests/test_motivation_multimodal_pilot.py`

**Interfaces:**
- Consumes: calibration/pilot JSONL and six score functions.
- Produces: `calibration_scores.json`, 27 threshold JSON files, `pilot/results.json`, and `select_configuration(results) -> SelectedConfiguration`.

- [ ] **Step 1: Write failing grid and deterministic-selection tests**

```python
def test_declared_grid_has_27_unique_configs():
    configs = list(iter_threshold_configs())
    assert len(configs) == len({c.key for c in configs}) == 27

def test_selection_uses_speedup_then_cardinality_then_key():
    chosen = select_configuration(fake_qualifying_results())
    assert chosen.key == "p20-q80-a60-c80-b80"
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_motivation_multimodal_pilot.py'
```

- [ ] **Step 3: Implement score caching, quantiles, and immutable artifact schemas**

Store one raw score per `(record_id, operator)`. Derive thresholds with NumPy quantiles for `P={.20,.35,.50}`, `Q={.70,.80,.90}` retained sharpest, `A={.40,.60,.80}` retained highest, and fixed `C=B=.80`. Include command, git revision, model revisions, fixture checksum, environment, and UTC timestamps in every summary.

- [ ] **Step 4: Implement pilot orchestration and native cost replay**

For each configuration, generate both plans with `use_my_optimizer=4` (`dp_two_stage_optimizer`) and `use_my_optimizer=2` (`dp_optimizer`), score both materialized plans through `Optimizer.calculate_cost` and `DpOptimizer.calculate_dp_objective_cost`, execute two alternating pilot repetitions, verify output ID equality, and write the complete result even when a plan fails.

- [ ] **Step 5: Run unit tests and a two-record mocked pilot**

Run the Step 2 command plus:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && python -m evaluation.motivation_multimodal.pilot --fixture outputs/motivation_multimodal/fixture --output outputs/motivation_multimodal/test-pilot --mock-scores --max-configs 1'
```

Expected: tests pass; the mock run writes plans, scores, two repetitions, equality status, and a selection decision without CUDA.

- [ ] **Step 6: Commit pilot machinery**

```bash
git add evaluation/motivation_multimodal tests/test_motivation_multimodal_pilot.py
git commit -m "feat: add bounded multimodal pilot selection"
```

### Task 5: Operator-scaling benchmark and raw-data validation

**Files:**
- Create: `evaluation/motivation_multimodal/scaling.py`
- Create: `tests/test_motivation_multimodal_scaling.py`

**Interfaces:**
- Consumes: first 128 scaling records and warmed operators.
- Produces: `operator_scaling/raw.json`, `operator_scaling/summary.csv`, and `summarize_trials(rows) -> list[ScalingSummary]`.

- [ ] **Step 1: Write failing grid, interleaving, and summary tests**

```python
def test_scaling_grid_matches_spec():
    assert text_grid() == (16, 32, 64, 128, 256, 512)
    assert image_side_grid() == (128, 256, 512, 1024)
    assert len(multimodal_grid()) == 16

def test_each_point_has_seven_trials(rows):
    counts = Counter((r.operator, r.text_tokens, r.image_side) for r in rows)
    assert set(counts.values()) == {7}
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_motivation_multimodal_scaling.py'
```

- [ ] **Step 3: Implement controlled input generation and timed trials**

Generate token targets by repeating complete source-caption clauses and trimming with the operator tokenizer. Resize each source image with Lanczos while preserving content. Warm every operator, synchronize CUDA before and after GPU timing, interleave scale points with a fixed seeded order per repetition, and store nanosecond raw durations before computing median and IQR.

- [ ] **Step 4: Run tests and a one-record smoke benchmark**

Run the Step 2 command and a smoke invocation with `--records 1 --repetitions 1 --text-grid 16 --image-grid 128`; verify that smoke artifacts are written outside the formal output path.

- [ ] **Step 5: Commit scaling benchmark**

```bash
git add evaluation/motivation_multimodal/scaling.py tests/test_motivation_multimodal_scaling.py
git commit -m "feat: measure multimodal operator scaling"
```

### Task 6: Data-driven Figure 2 and Figure 5 generators

**Files:**
- Create: `evaluation/motivation_multimodal/plot_figure2.py`
- Create: `evaluation/motivation_multimodal/plot_figure5.py`
- Create: `tests/test_motivation_multimodal_plots.py`
- Modify after valid formal results: `my_paper/69e75a0100d7b4afeb1cfc20/figures/pipeline_cooptimization_equivalent.pdf`
- Modify after valid scaling results: `my_paper/69e75a0100d7b4afeb1cfc20/figures/operator_input_size_scaling.pdf`

**Interfaces:**
- Consumes: selected configuration, materialized YAML plans, formal timing JSON, Cedar/PICO scores, and scaling CSV.
- Produces: reproducible PDF/PNG Figure 2 and Figure 5 plus a `figure_manifest.json` mapping every plotted value to source artifacts.

- [ ] **Step 1: Write failing provenance and missing-data tests**

```python
def test_plot_rejects_missing_repetition(tmp_path):
    with pytest.raises(ValueError, match="three formal repetitions"):
        load_figure2_data(incomplete_formal_artifact(tmp_path))

def test_manifest_hashes_every_input(rendered_figure):
    manifest = json.loads(rendered_figure.manifest.read_text())
    assert all(len(item["sha256"]) == 64 for item in manifest["inputs"])
```

- [ ] **Step 2: Run plot tests and verify failure**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_motivation_multimodal_plots.py'
```

- [ ] **Step 3: Implement Figure 2 renderer**

Parse actual fused blocks, variants, widths, and logical order from plan YAML. Render `(a)` logical modality/dependency lanes, `(b)` staged and joint stage boxes, `(c)` measured median/IQR and Cedar scores as separate axes. Use accessible backend colors consistently with the paper and label every numeric bar.

- [ ] **Step 4: Implement Figure 5 renderer**

Render six aligned panels: `N/P` line charts versus tokens, `Q/A` line charts versus megapixels, and `C/B` heatmaps over tokens and pixels. Read all medians/IQR values from `summary.csv`; never calculate or embed benchmark values in the plotting module.

- [ ] **Step 5: Run plot tests with checked synthetic fixtures and commit**

Run the Step 2 command. Expected: tests pass and fixture PDFs contain six Figure 5 panels and separate runtime/cost axes in Figure 2.

```bash
git add evaluation/motivation_multimodal/plot_figure2.py evaluation/motivation_multimodal/plot_figure5.py tests/test_motivation_multimodal_plots.py
git commit -m "feat: plot multimodal motivation figures"
```

### Task 7: Smoke validation and offline launchers

**Files:**
- Create: `evaluation/motivation_multimodal/run_pilot.sh`
- Create: `evaluation/motivation_multimodal/run_formal.sh`
- Create: `evaluation/motivation_multimodal/run_scaling.sh`
- Create: `tests/test_motivation_multimodal_launchers.py`

**Interfaces:**
- Consumes: fixture, workload, pilot/scaling runners, and fixed resource environment.
- Produces: idempotent launch commands, PID files, top-level logs, metadata, and nonzero failure status artifacts.

- [ ] **Step 1: Write failing shell-interface tests**

Assert `bash -n` succeeds, each launcher contains `source env/bin/activate`, refuses an existing live PID, records `W=8` and `CPU_BUDGET=64`, and never contains fallback sample counts or threshold rewrites.

- [ ] **Step 2: Run launcher tests and verify failure**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_motivation_multimodal_launchers.py'
```

- [ ] **Step 3: Implement exact launchers**

Each script accepts `--output-root`, writes `metadata.txt`, redirects stdout/stderr to `run.log`, stores the child PID, uses `set -euo pipefail`, and invokes one Python module. The scripts themselves run inside the already selected container; the host launch command wraps them with `docker exec` and `nohup`.

- [ ] **Step 4: Run unit suite and end-to-end smoke test**

Run all new tests, then a two-record CPU/CUDA-Ray smoke path with both optimizers. Validate identical record IDs, model revisions, plan YAML parsing, and GPU cleanup. Smoke outputs go to `outputs/motivation_multimodal/smoke/` and are never accepted by plot loaders.

- [ ] **Step 5: Commit launchers**

```bash
git add evaluation/motivation_multimodal/run_*.sh tests/test_motivation_multimodal_launchers.py
git commit -m "feat: add reproducible multimodal experiment launchers"
```

### Task 8: Run pilot selection, formal comparison, and scaling benchmark

**Files:**
- Create at runtime: `outputs/motivation_multimodal/pilot/`
- Create at runtime: `outputs/motivation_multimodal/formal/`
- Create at runtime: `outputs/motivation_multimodal/operator_scaling/`

**Interfaces:**
- Consumes: all validated code from Tasks 1--7.
- Produces: the immutable experimental evidence consumed by Task 9.

- [ ] **Step 1: Launch the 27-point pilot offline**

Run from the host:

```bash
nohup docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && evaluation/motivation_multimodal/run_pilot.sh --output-root outputs/motivation_multimodal/pilot' > outputs/motivation_multimodal/pilot.nohup.log 2>&1 &
```

Record the host PID in `outputs/motivation_multimodal/pilot.nohup.pid` and report the log path to the user.

- [ ] **Step 2: Validate and freeze the selected configuration**

Run the artifact validator. It must see 27 terminal configurations, output equality, at least two backends in the joint plan, lower joint pilot median, and a Cedar ranking reversal. Copy the selected threshold artifact by checksum into `selected_configuration.json`; never edit it manually.

- [ ] **Step 3: Launch formal and scaling experiments offline**

After Step 2 succeeds, start `run_formal.sh` and `run_scaling.sh` with separate nohup logs and PID files. Formal execution uses 3,000 records and three round-robin repetitions; scaling uses 128 records and seven interleaved repetitions at every declared point.

- [ ] **Step 4: Validate completed evidence**

Require terminal success markers, exact counts, stable model/fixture checksums, three formal repetitions per optimizer, seven scaling trials per point, identical staged/joint output IDs, no runtime fallback, and no overlap among splits.

### Task 9: Generate figures, update paper references, and verify the PDF

**Files:**
- Modify: `my_paper/69e75a0100d7b4afeb1cfc20/section/02_background_and_motivation.tex`
- Modify: `my_paper/69e75a0100d7b4afeb1cfc20/section/03_cost_model.tex`
- Modify: `my_paper/69e75a0100d7b4afeb1cfc20/section/04_plan_selection.tex`
- Modify: the two paper PDF assets listed in Task 6
- Create: `outputs/motivation_multimodal/figures/figure_manifest.json`

**Interfaces:**
- Consumes: validated Task 8 artifacts only.
- Produces: final Figure 2/Figure 5, consistent running-example prose in Sections 2--4, and a compiling paper.

- [ ] **Step 1: Render and inspect both figures**

Run both plotting modules against formal artifacts, render PDF pages to PNG, and inspect them at publication size. Verify no clipped labels, illegible stage annotations, misleading shared axes, or rasterized text.

- [ ] **Step 2: Replace the running example prose with measured facts**

Use the generated operator names, actual plan layouts, exact formal medians/IQR, and Cedar scores. Identify Figure 2 as a controlled pilot-selected counterexample. Replace the audio example in Chapters 3 and 4 while retaining their current logical organization.

- [ ] **Step 3: Compile the paper fully inside the container**

Run:

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar/my_paper/69e75a0100d7b4afeb1cfc20 && source /workspace/OptimalCedar/env/bin/activate && pdflatex -interaction=nonstopmode -halt-on-error main.tex && bibtex main && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex'
```

Expected: exit status zero with no undefined citations or references.

- [ ] **Step 4: Run the complete new test suite and artifact validators**

```bash
docker exec optimalcedar-torch201-dev bash -lc 'cd /workspace/OptimalCedar && source env/bin/activate && pytest -q tests/test_multimodal_running_example_*.py tests/test_motivation_multimodal_*.py'
```

Expected: all tests pass. Re-run figure provenance validation after compilation.

- [ ] **Step 5: Commit paper and reproducible figure artifacts**

Stage only the new workload, experiment code, source figure artifacts, paper TeX, and the two final PDFs. Exclude model caches, datasets, smoke outputs, PIDs, and logs.

```bash
git commit -m "paper: add measured multimodal running example"
```
