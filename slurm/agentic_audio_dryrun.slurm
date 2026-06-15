#!/bin/bash
#SBATCH --job-name=storymem-agentic-audio-plan
#SBATCH --output=logs/storymem_agentic_audio_plan_%j.out
#SBATCH --error=logs/storymem_agentic_audio_plan_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --time=00:20:00

set -euo pipefail

SCRIPT_DIR="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
cd "${SCRIPT_DIR}"

mkdir -p logs results
source .venv/bin/activate

storymem-agentic plan-audio \
  --rhyme-file "${RHYME_FILE:-story/twinkle_first4_lyrics.txt}" \
  --story-json "${STORY_JSON:-story/storybook_star_toddler_playroom_singalong.json}" \
  --output-dir "${OUTPUT_DIR:-results/agentic_audio_plan_${SLURM_JOB_ID}}" \
  --target-duration "${TARGET_DURATION:-24}"
