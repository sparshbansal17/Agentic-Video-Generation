# Agentic AV Lullaby Benchmark v1

This benchmark makes the project’s actual research claim testable: a single request is expanded
into song, multi-scene video, assembly, review, and targeted repair. AutoMV is the primary baseline
because it is the closest published multi-agent music-video system. MAVIN is the strongest joint
multi-shot audio-video model comparison, and MovieAgent is the closest hierarchical multi-agent
movie-planning comparison.

The comparison has two tracks because the systems do not accept equivalent inputs:

1. `conditioned_song_to_video`: all systems receive the same mastered song, lyrics, and case
   description. This is the direct AutoMV comparison. StoryMem Agentic must bypass song creation;
   MAVIN and MovieAgent use the documented adapters in `storymem_agentic.benchmark.adapters`.
2. `end_to_end_prompt_to_song_video`: one request must produce the song and video. This is the
   project’s target setting. Report which baselines need an external audio generator and include
   that generator's time and cost; do not label those cascades as native baseline results.

There is also a renderer-controlled diagnostic called the **shared-generator agent-design track**.
It runs the published AutoMV or MovieAgent planning contract and then renders the normalized plan
with the same StoryMem/Wan/ACE-Step backend. This isolates differences in hierarchical agent
design, but it is not a native end-to-end score for either publication. Always label it separately
from native-backend results.

## Locked protocol

- Run every case with three fixed seeds. No manual prompt, plan, media, or timeline edits.
- Preserve each system’s native reviewer/planner. Use the same generated output for automatic and
  blinded human evaluation.
- Normalize to 24 fps and the benchmark audio sample rate only after generation. Record native
  resolution, output duration, wall time, accelerator hours, retries, and estimated API cost.
- Store one `submission.json` per system/case/seed. The required fields are defined in
  `submission.schema.json`; paths are relative to that JSON file.
- Report failures and missing outputs in coverage. Never replace them with zero-quality fabricated
  videos or published table values.

The binary audio pack is intentionally versioned separately from this source repository. Its IDs
are locked in `manifest.json`. A benchmark release must publish checksums for those masters before
claiming conditioned-track results.

For `conditioned_twinkle`, the manifest now locks the exact 24-second PCM master checksum and its
deterministic construction recipe. Store it at
`results_baselines/assets/conditioned_twinkle/locked_audio.wav`; `results_baselines/` is ignored so
external repositories, model artifacts, and generated media cannot enter source commits.

## Metric families

The head-to-head table should include:

| Family | Metrics | Why |
|---|---|---|
| Delivery | completion rate, valid A/V/subtitle streams, duration error | Measures whether the agentic pipeline returns a usable artifact |
| Audio/lyrics | FAD, CLAP text-audio score, Whisper WER, lyric-line completeness, line-boundary MAE | Covers MAVIN audio quality and this project’s timing-first song requirements |
| Visual/semantic | FVD, ViCLIP text-video score, prompt/lyric shot retrieval | Covers MAVIN and AutoMV content alignment |
| Continuity | VBench subject/background consistency, cross-shot ViCLIP, blinded identity score | Covers all three comparison papers |
| Audio-video | ImageBind A/V score, SyncNet when speech/lip motion exists, beat/onset alignment | Covers joint or cascaded synchronization without forcing lip sync on non-vocal shots |
| Narrative | shot-transition accuracy, order accuracy, script faithfulness, coherence | Covers MAVIN/MovieAgent and the multi-scene claim |
| Safety | unsafe-content rate and false-positive rate on matched safe controls | Essential for lullaby outputs |
| Agentic review | injected-fault detection F1, scene localization F1, repair success, collateral-edit rate | Directly tests the novel review-and-repair design |
| Efficiency | wall time, accelerator hours, API cost, regenerated seconds | Tests whether targeted repair actually saves work |

FVD/FAD require enough samples and must be reported only for the full locked case/seed set. For a
small diagnostic run, report per-output embedding similarities and human confidence intervals, not
sample-starved FVD/FAD as if they were stable.

## Review-and-repair benchmark

`fault_suite.json` defines eight controlled failures spanning assembly, audio, semantics,
continuity, safety, and narrative order. Starting from an accepted output, inject exactly one fault,
run the native reviewer once, and then permit one repair iteration. Score:

- detection micro-precision/recall/F1 over fault labels;
- scene localization precision/recall/F1;
- post-repair pass rate and blinded quality delta;
- collateral-edit rate: unaffected scenes changed divided by unaffected scenes;
- regenerated seconds, wall time, and cost relative to full regeneration.

Run an ablation with review disabled and an ablation that always regenerates the full video. The
agentic design is supported only if targeted review improves final quality or completion while
reducing collateral regeneration/time/cost.

## Commands

```bash
python -m storymem_agentic.benchmark validate \
  --manifest benchmarks/agentic_av_v1/manifest.json

python -m storymem_agentic.benchmark coverage \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --submissions-root benchmark_submissions

python -m storymem_agentic.benchmark history \
  --results-root results \
  --output-dir benchmark_results/agentic_av_v1

python -m storymem_agentic.benchmark score-plan \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --case-id conditioned_twinkle \
  --plan results_baselines/plans/automv/conditioned_twinkle/seed_000/storymem_story.json \
  --provenance results_baselines/plans/automv/conditioned_twinkle/seed_000/provenance.json \
  --output results_baselines/plans/automv/conditioned_twinkle/seed_000/planning_metrics.json

python -m storymem_agentic.benchmark compare \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --submissions-root results_baselines/submissions \
  --plans-root results_baselines/plans \
  --output-dir results_baselines/reports/conditioned_twinkle_seed_000

python -m storymem_agentic.benchmark score-media \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --case-id conditioned_twinkle \
  --video path/to/raw-pre-subtitle-video.mp4 \
  --output path/to/submission/media_metrics.json \
  --clip-cache /home/bansa125/.cache/clip
```

Cluster smoke and common-renderer runs use:

```bash
sbatch slurm/agentic_benchmark_planning_ai.slurm
sbatch --export=ALL,SYSTEM=automv slurm/published_baseline_planning_smoke.slurm
sbatch --export=ALL,SYSTEM=movieagent slurm/published_baseline_planning_smoke.slurm
sbatch --export=ALL,SYSTEM=automv slurm/published_baseline_storymem_generate.slurm
sbatch --export=ALL,SYSTEM=movieagent slurm/published_baseline_storymem_generate.slurm
```

The planning adapter records the SHA-256 digest of the external repository source file whose agent
contract it executes. AutoMV uses its screenwriter/director storyboard contract. MovieAgent loads
the upstream `screenwriterCoT-sys`, `ScenePlanningCoT-sys`, and `ShotPlotCreateCoT-sys` prompts
directly. No external repository files are copied into this repository or committed.
Exact upstream URLs/commits, install results, and native API blockers are recorded in
`baseline_lock.json`.

The history command is intentionally conservative: it excludes evaluation JSON files whose
iteration directory has no non-empty MP4. Its prompt-adherence and reviewer values are internal
diagnostics, not direct scores against the publications in `published_reference.json`.
