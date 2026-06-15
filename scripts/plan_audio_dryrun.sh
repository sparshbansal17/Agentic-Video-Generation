#!/bin/bash
set -euo pipefail

module load python
source .venv/bin/activate

storymem-agentic plan-audio \
  --rhyme-file "${RHYME_FILE:-story/twinkle_first4_lyrics.txt}" \
  --story-json "${STORY_JSON:-story/storybook_star_toddler_playroom_singalong.json}" \
  --output-dir "${OUTPUT_DIR:-results/agentic_audio_dryrun}" \
  --target-duration "${TARGET_DURATION:-24}"
