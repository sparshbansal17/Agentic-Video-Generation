#!/bin/bash
#SBATCH --job-name=storymem-audio-shift
#SBATCH --output=/scratch/gautschi/bansa125/StoryMem/logs/audio_shift_%j.out
#SBATCH --error=/scratch/gautschi/bansa125/StoryMem/logs/audio_shift_%j.err
#SBATCH --partition=ai
#SBATCH --qos=preemptible
#SBATCH --account=aarc
#SBATCH --cpus-per-task=2
#SBATCH --time=00:02:00

set -euo pipefail
umask 002

SCRIPT_DIR="${STORYMEM_DIR:-/scratch/gautschi/bansa125/StoryMem}"
cd "${SCRIPT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR}"
FINAL_VIDEO="${FINAL_VIDEO:?Set FINAL_VIDEO}"
SHIFT_SECONDS="${SHIFT_SECONDS:-12.0}"
TEMPO="${TEMPO:-0.55}"
FADE_DURATION="${FADE_DURATION:-2.0}"
AUDIO_SOURCE="${AUDIO_SOURCE:-${OUTPUT_DIR}/audio/mixed_song.wav}"
FFMPEG_BIN="${FFMPEG_BIN:-/home/bansa125/bin/ffmpeg}"
FFPROBE_BIN="${FFPROBE_BIN:-/home/bansa125/bin/ffprobe}"

mkdir -p "${OUTPUT_DIR}/audio"

VIDEO_DURATION="$("${FFPROBE_BIN}" -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "${FINAL_VIDEO}")"
FADE_OUT_START="$(awk -v d="${VIDEO_DURATION}" -v f="${FADE_DURATION}" 'BEGIN { s=d-f; if (s < 0) s=0; printf "%.3f", s }')"
RETIME_AUDIO="${OUTPUT_DIR}/audio/mixed_song_aligned.wav"
OUTPUT_VIDEO="${FINAL_VIDEO%.mp4}_with_music_aligned_v2.mp4"

"${FFMPEG_BIN}" -y \
  -i "${AUDIO_SOURCE}" \
  -af "atrim=start=${SHIFT_SECONDS},asetpts=N/SR/TB,atempo=${TEMPO},apad=pad_dur=${VIDEO_DURATION},atrim=0:${VIDEO_DURATION},afade=t=out:st=${FADE_OUT_START}:d=${FADE_DURATION}" \
  -ar 48000 -ac 2 \
  "${RETIME_AUDIO}"

"${FFMPEG_BIN}" -y \
  -i "${FINAL_VIDEO}" \
  -i "${RETIME_AUDIO}" \
  -map 0:v:0 \
  -map 1:a:0 \
  -c:v copy \
  -c:a aac \
  -b:a 192k \
  -movflags +faststart \
  -shortest \
  "${OUTPUT_VIDEO}"

echo "${OUTPUT_VIDEO}"
