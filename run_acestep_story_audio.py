import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
from acestep.pipeline_ace_step import ACEStepPipeline


def _read_text(path):
    return Path(path).read_text(encoding="utf-8").strip()


def _read_prompt(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_wav_with_soundfile(self, target_wav, idx, save_path=None, sample_rate=48000, format="wav"):
    if save_path is None:
        output_path = Path("outputs") / f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
    else:
        output_path = Path(save_path)
        if output_path.is_dir():
            output_path = output_path / f"output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = target_wav.float().detach().cpu().numpy()
    if audio.ndim == 2:
        audio = audio.T
    sf.write(str(output_path), audio, sample_rate)
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate StoryMem song audio with ACE-Step.")
    parser.add_argument("--lyrics-file", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-path", default=os.getenv("ACE_STEP_CHECKPOINT_PATH", ""))
    parser.add_argument("--device-id", type=int, default=int(os.getenv("ACE_STEP_DEVICE_ID", "0")))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlapped-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--infer-step", type=int, default=int(os.getenv("ACE_STEP_INFER_STEP", "27")))
    parser.add_argument("--guidance-scale", type=float, default=float(os.getenv("ACE_STEP_GUIDANCE_SCALE", "15.0")))
    parser.add_argument("--scheduler-type", default=os.getenv("ACE_STEP_SCHEDULER_TYPE", "euler"))
    parser.add_argument("--cfg-type", default=os.getenv("ACE_STEP_CFG_TYPE", "apg"))
    parser.add_argument("--omega-scale", type=float, default=float(os.getenv("ACE_STEP_OMEGA_SCALE", "10.0")))
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    prompt_data = _read_prompt(args.prompt_file)
    lyrics = _read_text(args.lyrics_file)
    prompt = prompt_data.get("style_prompt") or (
        "gentle adult lullaby, music box, soft bells, warm bedtime nursery rhyme"
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32",
        torch_compile=args.torch_compile,
        cpu_offload=args.cpu_offload,
        overlapped_decode=args.overlapped_decode,
    )
    pipeline.save_wav_file = _save_wav_with_soundfile.__get__(pipeline, ACEStepPipeline)
    pipeline(
        audio_duration=args.duration,
        prompt=prompt,
        lyrics=lyrics,
        infer_step=args.infer_step,
        guidance_scale=args.guidance_scale,
        scheduler_type=args.scheduler_type,
        cfg_type=args.cfg_type,
        omega_scale=args.omega_scale,
        manual_seeds=str(args.seed),
        guidance_interval=0.5,
        guidance_interval_decay=0.0,
        min_guidance_scale=3.0,
        use_erg_tag=True,
        use_erg_lyric=True,
        use_erg_diffusion=True,
        oss_steps="",
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
        save_path=str(output),
    )


if __name__ == "__main__":
    main()
