#!/usr/bin/env bash
#SBATCH --job-name=audio-redesign-canary
#SBATCH --output=logs/audio_redesign_canary_%j.out
#SBATCH --error=logs/audio_redesign_canary_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
umask 002

REPO_ROOT="${REPO_ROOT:-/scratch/gautschi/bansa125/storymem-agentic}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/results/final_varied_rockabye_agentic_4gpu_13541363}"
SOURCE_ITERATION="${SOURCE_ITERATION:-${SOURCE_ROOT}/iterations/001}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/results/audio_redesign_canary/${SLURM_JOB_ID:-manual}}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
VENV="${VENV:-${REPO_ROOT}/.venv}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${REPO_ROOT}/.torch-cache}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
mkdir -p "${REPO_ROOT}/logs" "${RUN_ROOT}/iterations/001/generated" "${TMPDIR}"

module load python
cd "${REPO_ROOT}"

cp "${SOURCE_ITERATION}/generated/"[0-9][0-9]_[0-9][0-9].mp4 "${RUN_ROOT}/iterations/001/generated/"
cp "${SOURCE_ITERATION}/generated/generated_subtitled.mp4" "${RUN_ROOT}/iterations/001/generated/"
cp "${SOURCE_ITERATION}/generated/subtitles.ass" "${RUN_ROOT}/iterations/001/generated/"

ACE_STEP_CMD="${REPO_ROOT}/audio_tools/bin/storymem-acestep --lyrics-file \${lyrics_file} --prompt-file \${prompt_file} --duration \${duration} --seed \${seed} --output \${output_file}"
WHISPERX_CMD="${REPO_ROOT}/.venv-whisperx/bin/whisperx \${audio_file} --model small --language en --device cuda --compute_type float16 --vad_method silero --output_format json --output_dir \${output_dir}"
VLM_CMD="${VENV}/bin/python ${REPO_ROOT}/scripts/local_lullaby_reviewer.py --model Qwen/Qwen2-VL-7B-Instruct --max-new-tokens 1800 --debug-output ${RUN_ROOT}/vlm_review_raw_output.txt"
OMNI_CMD="${VENV}/bin/python ${REPO_ROOT}/scripts/local_omni_audio_reviewer.py --model Qwen/Qwen2.5-Omni-7B"

set +e
"${VENV}/bin/python" -m storymem_agentic.cli generate \
  --topic "Rock-a-bye Baby" \
  --lyrics-file "${SOURCE_ITERATION}/lyrics.txt" \
  --production-plan "${SOURCE_ITERATION}/production_plan.json" \
  --output-dir "${RUN_ROOT}" \
  --target-duration 20 \
  --clip-count 4 \
  --max-iterations 1 \
  --seed 1701 \
  --no-execute-video \
  --planner-backend mock \
  --review-backend command \
  --vlm-command "${VLM_CMD}" \
  --audio-review-command "${OMNI_CMD}" \
  --audio-aligner whisperx \
  --whisperx-command "${WHISPERX_CMD}" \
  --generate-audio \
  --media-audio-mode full_song \
  --voice-backend ace_step_full_song \
  --music-backend ace_step_full_song \
  --ace-step-cmd "${ACE_STEP_CMD}" \
  --full-song-candidates 8 \
  --ffmpeg-bin "${FFMPEG_BIN}"
rc=$?
set -e

"${VENV}/bin/python" - "${RUN_ROOT}" "${rc}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
iteration = root / "iterations" / "001"
summary = {"returncode": int(sys.argv[2]), "run_root": str(root)}
for name in [
    "audio_candidate_manifest.json",
    "audio_postprocess_result.json",
    "whisperx_alignment.json",
    "evaluation_report.json",
    "audio_analysis.json",
]:
    path = iteration / name
    summary[name] = {"exists": path.exists(), "path": str(path)}
    if path.exists() and name == "evaluation_report.json":
        report = json.loads(path.read_text())
        summary["passed"] = report.get("passed")
        summary["failure_reasons"] = report.get("failure_reasons")
        summary["reviewers"] = {
            item.get("reviewer"): {
                "passed": item.get("passed"),
                "failure_reasons": item.get("failure_reasons"),
            }
            for item in report.get("reviewer_reports", [])
        }
(root / "canary_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

exit "${rc}"
