# StoryMem Agentic

StoryMem Agentic is a self-contained StoryMem fork plus an agentic nursery-rhyme production layer. It keeps the original StoryMem/Wan video pipeline available, then adds planning and evaluation tools for scene design, timing-first audio generation, music, subtitles, and feedback loops.

The audio system deliberately separates responsibilities:

- exact lyrics/dialogue: F5-TTS or CosyVoice adapters
- continuous background music: MusicGen, Stable Audio, or ACE-Step instrumental/full-song adapters
- full sung songs: ACE-Step 1.5 or YuE as a single whole-track generation
- timing verification: WhisperX word-level alignment
- final assembly: ffmpeg mix/mux manifests with ducking and loudness targets

This avoids the old failure mode where ACE-Step was asked to satisfy exact lyrics, segment continuity, timing, and musical coherence all at once.

## Repository Layout

```text
src/storymem_agentic/        Agentic orchestration, audio planning, evaluation, CLI
audio_backends/              Backend adapter definitions live under the package
configs/audio/               Runtime backend command templates and mix defaults
slurm/                       HPC entrypoints for dry-run and GPU workflows
story/, wan/, pipeline.py    Upstream StoryMem/Wan code retained for compatibility
docs/storymem-upstream-readme.md  Original upstream README snapshot
THIRD_PARTY_NOTICES.md       Licenses and model/source notes
```

## Local Setup With uv

```bash
cd /scratch/gautschi/bansa125/storymem-agentic
uv venv
source .venv/bin/activate
uv pip install -e '.[dev,agents]'
```

GPU/video dependencies are intentionally optional:

```bash
uv pip install -e '.[video]'
uv pip install -e '.[audio-tts,audio-music,align]'
```

Some engines, such as ACE-Step 1.5, YuE, AudioCraft, and Stable Audio Tools, are best installed in separate engine-specific environments and invoked through command templates in `configs/audio/default.yaml`.

## Dry Run

```bash
storymem-agentic plan-audio \
  --rhyme-file story/twinkle_first4_lyrics.txt \
  --output-dir results/agentic_twinkle_dryrun \
  --target-duration 24 \
  --story-json story/storybook_star_toddler_playroom_singalong.json
```

The dry run writes:

- `audio_plan.json`
- `mix_manifest.json`
- `audio_evaluation_report.json`

To create a top-level agentic run manifest without invoking GPU/media backends:

```bash
storymem-agentic run \
  --rhyme-file story/twinkle_first4_lyrics.txt \
  --output-dir results/agentic_twinkle_run \
  --target-duration 24 \
  --story-json story/storybook_star_toddler_playroom_singalong.json
```

This writes `run_manifest.json` plus nested audio planning artifacts under `audio/`. StoryMem video generation is recorded as a pending stage in dry-run mode.

## GitHub Setup

Create the remote as private first:

```bash
gh repo create storymem-agentic --private --source . --remote origin --push
```

Do not commit models, checkpoints, generated media, results, logs, caches, or virtual environments.

## License

The retained StoryMem code is governed by the upstream S-Lab License 1.0 in `LICENSE.md`. Additional third-party code/model notes are tracked in `THIRD_PARTY_NOTICES.md`.
