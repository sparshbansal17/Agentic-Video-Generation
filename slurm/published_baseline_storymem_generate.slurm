#!/bin/bash
#SBATCH --job-name=published-baseline-generate
#SBATCH --output=logs/published_baseline_generate_%j.out
#SBATCH --error=logs/published_baseline_generate_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=56
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail
umask 002

ROOT="${STORYMEM_AGENTIC_DIR:-/scratch/gautschi/bansa125/storymem-agentic}"
STORYMEM_DIR="${STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
SYSTEM="${SYSTEM:?Set SYSTEM to automv or movieagent}"
CASE_ID="${CASE_ID:-conditioned_twinkle}"
SEED="${SEED:-0}"
NGPUS="${NGPUS:-4}"
PLAN_SEED_DIR="${PLAN_SEED_DIR:-seed_000}"
printf -v SEED_DIR 'seed_%03d' "${SEED}"
PLAN_DIR="${ROOT}/results_baselines/plans/${SYSTEM}/${CASE_ID}/${PLAN_SEED_DIR}"
STORY_SCRIPT="${PLAN_DIR}/storymem_story.json"
OUTPUT_PATH="${ROOT}/results_baselines/raw_renders/${SYSTEM}/${CASE_ID}/${SEED_DIR}"
SUBMISSION_DIR="${ROOT}/results_baselines/submissions/${SYSTEM}/${CASE_ID}/${SEED_DIR}"
LOCKED_AUDIO="${ROOT}/results_baselines/assets/${CASE_ID}/locked_audio.wav"
MODEL_ROOT="${MODEL_ROOT:-${STORYMEM_DIR}/models}"
T2V_MODEL_PATH="${T2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-T2V-A14B}"
I2V_MODEL_PATH="${I2V_MODEL_PATH:-${MODEL_ROOT}/Wan2.2-I2V-A14B}"
LORA_WEIGHT_PATH="${LORA_WEIGHT_PATH:-${MODEL_ROOT}/StoryMem/Wan2.2-MI2V-A14B}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
CLIP_CACHE="${CLIP_CACHE:-/home/bansa125/.cache/clip}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
START_SECONDS="$(date +%s)"

module load python
source "${ROOT}/.venv/bin/activate"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export HF_HOME="${ROOT}/.hf-cache"
export TRANSFORMERS_CACHE="${ROOT}/.hf-cache/transformers"
export TORCH_HOME="${ROOT}/.torch-cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export STORYMEM_PIPELINE_PATH="${STORYMEM_DIR}/pipeline.py"
export STORYMEM_DIST_TIMEOUT_SECONDS="${STORYMEM_DIST_TIMEOUT_SECONDS:-21600}"
export TMPDIR="${ROOT}/.tmp"
export MPLCONFIGDIR="${ROOT}/.tmp/matplotlib"
mkdir -p "${ROOT}/logs" "${OUTPUT_PATH}" "${SUBMISSION_DIR}" "${TMPDIR}" "${MPLCONFIGDIR}"

for required in "${STORYMEM_DIR}/pipeline.py" "${STORY_SCRIPT}" "${LOCKED_AUDIO}" \
  "${T2V_MODEL_PATH}" "${I2V_MODEL_PATH}" "${LORA_WEIGHT_PATH}"; do
  if [ ! -e "${required}" ]; then
    echo "Required benchmark input is missing: ${required}" >&2
    exit 2
  fi
done

python - "${STORY_SCRIPT}" "${LOCKED_AUDIO}" "${ROOT}/benchmarks/agentic_av_v1/manifest.json" "${CASE_ID}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
story = json.loads(path.read_text(encoding="utf-8"))
scenes = story.get("scenes")
assert isinstance(scenes, list) and len(scenes) == 4
for expected, scene in enumerate(scenes, 1):
    assert scene["scene_num"] == expected
    assert len(scene["video_prompts"]) == len(scene["cut"]) == 1
    assert isinstance(scene["cut"][0], bool)
    assert scene["video_prompts"][0].strip()
    assert scene["subtitle_text"].strip()
audio_path = Path(sys.argv[2])
manifest = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
case = next(item for item in manifest["cases"] if item["case_id"] == sys.argv[4])
audio_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()
assert audio_hash == case["locked_audio_sha256"], (audio_hash, case["locked_audio_sha256"])
print(f"validated {path}: {len(scenes)} scenes")
print(f"validated locked audio: {audio_hash}")
PY

cd "${STORYMEM_DIR}"
export PYTHONPATH="${STORYMEM_DIR}"
DISTRIBUTED_ARGS=(--nproc_per_node="${NGPUS}" --master_port="${MASTER_PORT}")
COMMON_ARGS=(
  --story_script_path "${STORY_SCRIPT}"
  --t2v_model_path "${T2V_MODEL_PATH}"
  --i2v_model_path "${I2V_MODEL_PATH}"
  --lora_weight_path "${LORA_WEIGHT_PATH}"
  --size "832*480"
  --max_memory_size 10
  --output_dir "${OUTPUT_PATH}"
  --ffmpeg_bin "${FFMPEG_BIN}"
  --seed "${SEED}"
  --dit_fsdp
  --t5_fsdp
  --ulysses_size "${NGPUS}"
  --offload_model
  --lora_rank 128
  --t5_cpu
)

torchrun "${DISTRIBUTED_ARGS[@]}" "${ROOT}/scripts/storymem_pipeline_with_timeout.py" \
  "${COMMON_ARGS[@]}" --t2v_first_shot
torchrun "${DISTRIBUTED_ARGS[@]}" "${ROOT}/scripts/storymem_pipeline_with_timeout.py" \
  "${COMMON_ARGS[@]}" --mi2v

RAW_VIDEO="${OUTPUT_PATH}/$(basename "${OUTPUT_PATH}").mp4"
END_SECONDS="$(date +%s)"
GENERATION_SECONDS="$((END_SECONDS - START_SECONDS))"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}"
python scripts/package_benchmark_submission.py \
  --system "${SYSTEM}" \
  --case-id "${CASE_ID}" \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --visual-video "${RAW_VIDEO}" \
  --locked-audio "${LOCKED_AUDIO}" \
  --plan "${STORY_SCRIPT}" \
  --trace "${PLAN_DIR}/planning_trace.json" \
  --output-dir "${SUBMISSION_DIR}" \
  --generation-seconds "${GENERATION_SECONDS}"

python -m storymem_agentic.benchmark score-media \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --case-id "${CASE_ID}" \
  --video "${RAW_VIDEO}" \
  --output "${SUBMISSION_DIR}/media_metrics.json" \
  --device cuda \
  --clip-cache "${CLIP_CACHE}"

python -m storymem_agentic.benchmark validate \
  --manifest benchmarks/agentic_av_v1/manifest.json \
  --submission "${SUBMISSION_DIR}/submission.json" \
  --require-media
