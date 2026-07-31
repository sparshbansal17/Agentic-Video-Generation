from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _case(manifest: dict[str, Any], case_id: str) -> dict[str, Any]:
    for case in manifest["cases"]:
        if case["case_id"] == case_id:
            return dict(case)
    raise ValueError(f"unknown benchmark case: {case_id}")


def sample_scene_midpoints(video_path: Path, scene_count: int) -> list[Any]:
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frames <= 0:
        capture.release()
        raise ValueError(f"cannot decode video: {video_path}")
    images = []
    for index in range(scene_count):
        frame_index = min(frames - 1, round((index + 0.5) * frames / scene_count))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise ValueError(f"cannot decode frame {frame_index}: {video_path}")
        images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    return images


def score_clip_media(
    *,
    video_path: Path,
    case: dict[str, Any],
    device: str = "cpu",
    clip_cache: str | Path | None = None,
) -> dict[str, Any]:
    import clip
    import torch

    scene_count = int(case["expected_scenes"])
    images = sample_scene_midpoints(video_path, scene_count)
    model, preprocess = clip.load(
        "ViT-B/32", device=device, download_root=str(clip_cache) if clip_cache else None
    )
    image_input = torch.stack([preprocess(image) for image in images]).to(device)
    lyrics = [str(line) for line in case["lyrics"]]
    assigned_texts = [f"{line}. {case['prompt']}" for line in lyrics]
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        lyric_features = model.encode_text(clip.tokenize(lyrics).to(device)).float()
        assigned_features = model.encode_text(clip.tokenize(assigned_texts).to(device)).float()
        global_feature = model.encode_text(clip.tokenize([case["prompt"]]).to(device)).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    lyric_features /= lyric_features.norm(dim=-1, keepdim=True)
    assigned_features /= assigned_features.norm(dim=-1, keepdim=True)
    global_feature /= global_feature.norm(dim=-1, keepdim=True)
    lyric_matrix = image_features @ lyric_features.T
    assigned = (image_features * assigned_features).sum(dim=-1)
    global_scores = (image_features @ global_feature.T).squeeze(-1)
    adjacent = (image_features[:-1] * image_features[1:]).sum(dim=-1)
    predictions = lyric_matrix.argmax(dim=-1)
    expected = torch.arange(scene_count, device=predictions.device)
    return {
        "case_id": case["case_id"],
        "metric_scope": "raw_visual_clip_vit_b32",
        "sampled_frames": scene_count,
        "clip_assigned_scene_similarity_mean": float(assigned.mean().item()),
        "clip_global_prompt_similarity_mean": float(global_scores.mean().item()),
        "clip_lyric_retrieval_order_accuracy": float((predictions == expected).float().mean().item()),
        "clip_adjacent_scene_similarity_mean": float(adjacent.mean().item()) if len(adjacent) else None,
        "per_scene_assigned_similarity": [float(value) for value in assigned.tolist()],
        "per_scene_predicted_lyric_index": [int(value) + 1 for value in predictions.tolist()],
        "note": (
            "CLIP scores are diagnostic single-output similarities on raw pre-subtitle frames; "
            "they are not FVD, ViCLIP, VBench, or a blinded human score."
        ),
    }


def write_clip_media_score(
    *,
    manifest_path: str | Path,
    case_id: str,
    video_path: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    clip_cache: str | Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result = score_clip_media(
        video_path=Path(video_path),
        case=_case(manifest, case_id),
        device=device,
        clip_cache=clip_cache,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
