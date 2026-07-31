# Agentic AV Benchmark: Retrospective Results

> These measurements describe local, media-backed StoryMem Agentic runs. They are not head-to-head baseline results because AutoMV, MAVIN, and MovieAgent outputs are not installed in this workspace.

## Coverage

- Reports discovered: 388
- Reports with non-empty MP4 evidence: 90
- Media-backed runs: 56

## Pipeline metrics

- First media-backed iteration acceptance: 0.0357
- Latest media-backed iteration acceptance: 0.0893
- Mean iterations/run: 1.6071
- Repair success after a media-backed failure: 0.1429
- Mean scene pass rate: 1.0
- Mean heuristic prompt-adherence proxy: 0.8
- Command-review terminal runs: 40
- Command-review terminal acceptance: 0.1
- Final video-stream pass rate: 1.0
- Final audio-stream pass rate: 0.9286
- Subtitle pass rate: 0.8393

## Audio diagnostics

- Terminal runs with WER: 52
- Mean WER: 0.3161
- Mean lyric-line completeness: 0.6613
- Mean absolute line-boundary drift (seconds): 1.8116

## Interpretation

Pipeline-internal scores are diagnostic only. They are not direct AutoMV, MAVIN, or MovieAgent comparisons until all systems run the locked benchmark inputs.
 Legacy non-command reports assign 0.8 when a clip exists; this is an artifact proxy, not a semantic model or human quality score.
