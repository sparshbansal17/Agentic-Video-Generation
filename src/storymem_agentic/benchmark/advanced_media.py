from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .media import _case


def sample_scene_frames(
    video_path: Path, scene_count: int, frames_per_scene: int = 8
) -> list[list[Any]]:
    """Sample multiple interior frames from each equal-duration benchmark scene."""
    import cv2
    from PIL import Image

    if scene_count < 1 or frames_per_scene < 2:
        raise ValueError("scene_count must be positive and frames_per_scene must be at least two")
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"cannot decode video: {video_path}")
    scenes: list[list[Any]] = []
    for scene_index in range(scene_count):
        images = []
        for frame_offset in range(frames_per_scene):
            fraction = (scene_index + (frame_offset + 0.5) / frames_per_scene) / scene_count
            index = min(frame_count - 1, round(fraction * (frame_count - 1)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                raise ValueError(f"cannot decode frame {index}: {video_path}")
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        scenes.append(images)
    capture.release()
    return scenes


def _normalize_rows(values: Any) -> Any:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _bootstrap_ci(values: Sequence[float], *, seed: int = 0) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(4000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, [0.025, 0.975])]


def contrastive_semantic_metrics(
    scene_features: Any, frame_features: Any, text_features: Any
) -> dict[str, Any]:
    """Compute continuous alignment and retrieval metrics from normalized embeddings."""
    import torch

    scene_scores = scene_features @ text_features.T
    frame_scores = torch.einsum("skd,td->skt", frame_features, text_features)
    count = int(scene_scores.shape[0])
    if tuple(scene_scores.shape) != (count, count):
        raise ValueError("scene and reference counts must match")
    negative_mask = ~torch.eye(count, dtype=torch.bool, device=scene_scores.device)
    diagonal = scene_scores.diagonal()
    hardest_negative = scene_scores.masked_fill(~negative_mask, -torch.inf).max(dim=1).values
    margins = diagonal - hardest_negative
    ranks = [
        1 + int((row > row[index]).sum().item())
        for index, row in enumerate(scene_scores)
    ]
    frame_diagonal = torch.stack(
        [frame_scores[index, :, index] for index in range(count)]
    )
    frame_negative = frame_scores.masked_fill(
        ~negative_mask[:, None, :], -torch.inf
    ).max(dim=-1).values
    frame_margins = frame_diagonal - frame_negative
    scene_margin_values = [float(value) for value in margins.tolist()]
    frame_margin_values = [float(value) for value in frame_margins.flatten().tolist()]
    return {
        "semantic_alignment_mean": float(diagonal.mean().item()),
        "semantic_alignment_min": float(diagonal.min().item()),
        "semantic_contrastive_margin_mean": float(margins.mean().item()),
        "semantic_contrastive_margin_min": float(margins.min().item()),
        "semantic_contrastive_margin_ci95": _bootstrap_ci(scene_margin_values),
        "semantic_retrieval_recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "semantic_retrieval_mrr": sum(1.0 / rank for rank in ranks) / count,
        "dense_frame_margin_mean": float(frame_margins.mean().item()),
        "dense_frame_margin_p10": float(torch.quantile(frame_margins.flatten(), 0.10).item()),
        "dense_frame_margin_ci95": _bootstrap_ci(frame_margin_values),
        "per_scene_alignment": [float(value) for value in diagonal.tolist()],
        "per_scene_contrastive_margin": scene_margin_values,
        "per_scene_reference_rank": ranks,
        "scene_reference_similarity_matrix": [
            [float(value) for value in row] for row in scene_scores.tolist()
        ],
    }


def temporal_embedding_metrics(frame_features: Any) -> dict[str, Any]:
    """Separate within-shot stability from desired between-shot semantic change."""
    import torch

    consecutive = (frame_features[:, :-1] * frame_features[:, 1:]).sum(dim=-1)
    scene_features = _normalize_rows(frame_features.mean(dim=1))
    cross_scene = (scene_features[:-1] * scene_features[1:]).sum(dim=-1)
    within = float(consecutive.mean().item())
    between = float(cross_scene.mean().item()) if len(cross_scene) else within
    return {
        "within_scene_embedding_consistency_mean": within,
        "within_scene_embedding_consistency_p10": float(
            torch.quantile(consecutive.flatten(), 0.10).item()
        ),
        "adjacent_scene_embedding_similarity_mean": between,
        "scene_change_separation": within - between,
        "per_scene_embedding_consistency": [
            float(value) for value in consecutive.mean(dim=1).tolist()
        ],
    }


def flow_warp_metrics(frames: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Measure motion-compensated temporal residual and optical-flow magnitude."""
    import cv2

    residuals: list[float] = []
    motions: list[float] = []
    for scene in frames:
        gray = [
            cv2.resize(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY), (320, 184))
            for image in scene
        ]
        for previous, current in zip(gray, gray[1:]):
            flow = cv2.calcOpticalFlowFarneback(
                previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            height, width = previous.shape
            grid_x, grid_y = np.meshgrid(
                np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
            )
            warped = cv2.remap(
                current,
                grid_x + flow[..., 0],
                grid_y + flow[..., 1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            residuals.append(float(np.mean(np.abs(previous.astype(float) - warped)) / 255))
            motions.append(float(np.median(np.linalg.norm(flow, axis=-1))))
    return {
        "flow_warp_error_mean": float(np.mean(residuals)),
        "flow_warp_error_p90": float(np.quantile(residuals, 0.90)),
        "optical_flow_median_pixels": float(np.mean(motions)),
        "optical_flow_median_pixels_std": float(np.std(motions)),
    }


def _scene_references(case: dict[str, Any]) -> list[str]:
    references = case.get("evaluation_scene_descriptions")
    if not isinstance(references, list) or len(references) != int(case["expected_scenes"]):
        raise ValueError(
            "advanced media scoring requires one locked evaluation_scene_description per scene"
        )
    return [str(value) for value in references]


def score_advanced_media(
    *,
    video_path: Path,
    case: dict[str, Any],
    device: str = "cpu",
    clip_cache: str | Path | None = None,
    dino_model: str = "facebook/dinov2-base",
    dino_cache: str | Path | None = None,
    frames_per_scene: int = 8,
) -> dict[str, Any]:
    import clip
    import torch
    from transformers import AutoImageProcessor, Dinov2Model

    scene_count = int(case["expected_scenes"])
    frames = sample_scene_frames(video_path, scene_count, frames_per_scene)
    flat_frames = [image for scene in frames for image in scene]
    references = _scene_references(case)
    clip_model, preprocess = clip.load(
        "ViT-B/32", device=device, download_root=str(clip_cache) if clip_cache else None
    )
    clip_images = torch.stack([preprocess(image) for image in flat_frames]).to(device)
    with torch.no_grad():
        clip_frames = _normalize_rows(clip_model.encode_image(clip_images).float())
        clip_text = _normalize_rows(
            clip_model.encode_text(clip.tokenize(references).to(device)).float()
        )
    clip_frames = clip_frames.reshape(scene_count, frames_per_scene, -1)
    clip_scenes = _normalize_rows(clip_frames.mean(dim=1))

    processor = AutoImageProcessor.from_pretrained(dino_model, cache_dir=dino_cache)
    dino = Dinov2Model.from_pretrained(dino_model, cache_dir=dino_cache).to(device).eval()
    dino_batches = []
    with torch.no_grad():
        for offset in range(0, len(flat_frames), 8):
            inputs = processor(images=flat_frames[offset : offset + 8], return_tensors="pt").to(
                device
            )
            output = dino(**inputs).last_hidden_state[:, 0]
            dino_batches.append(_normalize_rows(output.float()))
    dino_frames = torch.cat(dino_batches).reshape(scene_count, frames_per_scene, -1)

    return {
        "case_id": case["case_id"],
        "metric_scope": "advanced_dense_clip_dinov2_flow_v1",
        "frames_per_scene": frames_per_scene,
        "sampled_frames": len(flat_frames),
        "scene_references": references,
        **contrastive_semantic_metrics(clip_scenes, clip_frames, clip_text),
        "clip_temporal": temporal_embedding_metrics(clip_frames),
        "dinov2_temporal": temporal_embedding_metrics(dino_frames),
        "flow_temporal": flow_warp_metrics(frames),
        "metric_directions": {
            "semantic_alignment_mean": "higher",
            "semantic_contrastive_margin_mean": "higher",
            "semantic_retrieval_mrr": "higher",
            "dense_frame_margin_mean": "higher",
            "clip_temporal.scene_change_separation": "higher_with_identity_caveat",
            "dinov2_temporal.within_scene_embedding_consistency_mean": "higher",
            "dinov2_temporal.scene_change_separation": "higher_with_identity_caveat",
            "flow_temporal.flow_warp_error_mean": "lower",
            "flow_temporal.optical_flow_median_pixels": "diagnostic_no_fixed_direction",
        },
        "note": (
            "Single-output diagnostic with dense interior-frame sampling. Contrastive margins "
            "use all other locked scene descriptions as hard negatives. DINOv2 and flow report "
            "stability separately from scene change; no FVD claim is made from one sample."
        ),
    }


def write_advanced_media_score(
    *,
    manifest_path: str | Path,
    case_id: str,
    video_path: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    clip_cache: str | Path | None = None,
    dino_model: str = "facebook/dinov2-base",
    dino_cache: str | Path | None = None,
    frames_per_scene: int = 8,
) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    result = score_advanced_media(
        video_path=Path(video_path),
        case=_case(manifest, case_id),
        device=device,
        clip_cache=clip_cache,
        dino_model=dino_model,
        dino_cache=dino_cache,
        frames_per_scene=frames_per_scene,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
