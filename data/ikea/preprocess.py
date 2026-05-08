"""
Preprocess IKEA dataset:
1. FFmpeg downsample + resize         → datasets/ikea/fps2_384_videos
2. SigLIP encode (CLS + 3x3 avg pool) → datasets/ikea/fps2_384_siglip_tokens  shape (L, 10, D)
3. Downsample labels                  → datasets/ikea/fps2_labels
4. Metadata JSON                      → datasets/ikea/fps2_384_siglip_tokens/metadata.json
"""

import json
import os
import subprocess
import sys

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/data/ssd2/thw/data/dataset/ikea"
VIDEO_DIR = os.path.join(DATA_DIR, "ANU_ikea_dataset_video")
ANNO_PATH = os.path.join(DATA_DIR, "annotations/gt_action.npy")

WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_VIDEO_DIR = os.path.join(DATA_DIR, "fps2_384_videos")
OUTPUT_TOKEN_DIR = os.path.join(DATA_DIR, "fps2_384_siglip_tokens")
OUTPUT_LABEL_DIR = os.path.join(DATA_DIR, "fps2_labels")
SEGMENT_PATH = os.path.join(DATA_DIR, "annotations/formatted_segments.json")
# ── Hyper-parameters ───────────────────────────────────────────────────────────
TARGET_FPS = 2
RESOLUTION = 384
VIT_MODEL = "google/siglip-large-patch16-384"
PATCH_SIZE = 16
SPATIAL_GRID = RESOLUTION // PATCH_SIZE  # 24  →  24×24 = 576 spatial tokens
POOL_SIZE = SPATIAL_GRID // 3  # 8   →  pool 8×8 → 1 cell, giving 3×3 = 9 cells
BATCH_SIZE = 32
DEVICE = "cuda:0"

# Prefer system ffmpeg, fall back to local binary
FFMPEG_BIN = "ffmpeg"
_local_ffmpeg = os.path.join(WORKSPACE, "ffmpeg", "ffmpeg")
if os.path.isfile(_local_ffmpeg):
    FFMPEG_BIN = _local_ffmpeg


# ── Helpers ────────────────────────────────────────────────────────────────────


def ffmpeg_once(
    src_path: str,
    dst_path: str,
    *,
    fps: int,
    resolution: int,
    pad: str = "#000000",
    mode: str = "bicubic",
) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # vf = (
    #     f"scale='if(gt(iw\\,ih)\\,{resolution}\\,-2)':'if(gt(iw\\,ih)\\,-2\\,{resolution})',"
    #     f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color='{pad}'"
    # ) # 填充  +  resize
    vf = f"scale={resolution}:{resolution}"  #  强制resize, 不填充
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-sws_flags",
        mode,
        "-i",
        src_path,
        "-an",
        "-r",
        str(fps),
        "-vf",
        vf,
        "-threads",
        "10",
        dst_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src_path}:\n{result.stderr.decode()[-2000:]}"
        )


def encode_frames(
    model, processor, frames_np: np.ndarray, device: str, batch_size: int = 32
) -> torch.Tensor:
    """
    frames_np : (L, H, W, C) uint8 numpy
    Returns   : (L, 10, D) bfloat16 CPU tensor
                10 = 1 CLS + 9 (3×3 avg-pooled spatial tokens)
    """
    all_features = []
    for i in range(0, len(frames_np), batch_size):
        batch_np = frames_np[i : i + batch_size]
        pil_imgs = [Image.fromarray(f) for f in batch_np]
        inputs = processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.vision_model(**inputs)
        cls = outputs.pooler_output.unsqueeze(1)  # (B, 1, D)
        spatial = outputs.last_hidden_state  # (B, 576, D)

        B, _, D = spatial.shape

        # Reshape 576 → 24×24, then avg-pool 8×8 blocks → 3×3
        spatial = spatial.reshape(B, SPATIAL_GRID, SPATIAL_GRID, D)
        spatial = spatial.reshape(B, 3, POOL_SIZE, 3, POOL_SIZE, D)
        spatial = spatial.mean(dim=(2, 4))  # (B, 3, 3, D)
        spatial = spatial.reshape(B, 9, D)  # (B,   9, D)

        features = torch.cat([cls, spatial], dim=1)  # (B,  10, D)
        all_features.append(features.cpu().to(torch.bfloat16))

    return torch.cat(all_features, dim=0)  # (L, 10, D)


def downsample_labels(
    label: np.ndarray, orig_fps: float, target_fps: int, num_new_frames: int
) -> np.ndarray:
    """Pick the label closest in time for each new-fps frame index."""
    ratio = orig_fps / target_fps
    indices = np.round(np.arange(num_new_frames) * ratio).astype(int)
    indices = np.clip(indices, 0, len(label) - 1)
    return label[indices]


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TOKEN_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    # Load annotations
    data_dict = np.load(ANNO_PATH, allow_pickle=True).item()
    scan_names = data_dict["scan_name"]
    gt_labels = data_dict["gt_labels"]
    print(f"Total videos: {len(scan_names)}")

    # Load SigLIP model + processor
    print(f"Loading {VIT_MODEL} …")
    model = AutoModel.from_pretrained(VIT_MODEL, device_map=DEVICE)
    processor = AutoProcessor.from_pretrained(VIT_MODEL)
    model.eval()

    # Try to load existing metadata so we can resume interrupted runs
    meta_path = os.path.join(OUTPUT_TOKEN_DIR, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {}

    with open(SEGMENT_PATH, "r") as f:
        segments = json.load(f)
        database = segments["database"]
    for name, label in tqdm(
        zip(scan_names, gt_labels), total=len(scan_names), desc="videos"
    ):
        assert name in database, f"name {name} not in database"
        row = database[name]
        annotation = row["annotation"]
        src_video = os.path.join(VIDEO_DIR, name, "dev3/images/scan_video.avi")
        if not os.path.exists(src_video):
            tqdm.write(f"[skip] not found: {src_video}")
            continue

        # Use '__' as separator so the flat filename is unambiguous
        safe_name = name.replace(os.sep, "__").replace("/", "__")
        dst_video = os.path.join(OUTPUT_VIDEO_DIR, safe_name + ".mp4")
        token_path = os.path.join(OUTPUT_TOKEN_DIR, safe_name + ".pt")
        label_path = os.path.join(OUTPUT_LABEL_DIR, safe_name + ".npy")

        # ── Step 1: FFmpeg ──────────────────────────────────────────────────
        if not os.path.isfile(dst_video):
            try:
                ffmpeg_once(src_video, dst_video, fps=TARGET_FPS, resolution=RESOLUTION)
            except RuntimeError as e:
                tqdm.write(f"[error] ffmpeg failed for {name}: {e}")
                continue

        # ── Step 2: Read processed frames ───────────────────────────────────
        try:
            vr = VideoReader(dst_video, ctx=cpu(0))
            frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # (L, H, W, C)
        except Exception as e:
            tqdm.write(f"[error] reading {dst_video}: {e}")
            continue

        L = len(frames)
        if L == 0:
            tqdm.write(f"[skip] empty video: {dst_video}")
            continue

        # ── Step 3: SigLIP encoding ─────────────────────────────────────────
        if not os.path.isfile(token_path):
            tokens = encode_frames(model, processor, frames, DEVICE, BATCH_SIZE)
            torch.save(tokens, token_path)
        else:
            tokens = torch.load(token_path, map_location="cpu")

        D = tokens.shape[-1]

        # ── Step 4: Downsample labels ───────────────────────────────────────
        transed_anno = []
        if True:
            src_vr = VideoReader(src_video, ctx=cpu(0))
            src_fps: float = src_vr.get_avg_fps()
            src_frames = len(src_vr)
            del src_vr
            if not os.path.isfile(label_path):
                new_labels = downsample_labels(
                    np.asarray(label), src_fps, TARGET_FPS, L
                )
                np.save(label_path, new_labels)
            duration: float = src_frames / src_fps
            for segment in annotation:
                start = segment["segment"][0]
                end = segment["segment"][1]
                label = segment["label"]
                transed_anno.append(
                    {
                        "start": start / src_fps,
                        "end": end / src_fps,
                        "label": label,
                        "segment": [start / src_fps, end / src_fps],
                    }
                )

        # ── Step 5: Collect metadata ────────────────────────────────────────
        metadata[name] = {
            "subset": row["subset"],
            "scan_name": name,
            "video_path": os.path.abspath(dst_video),
            "token_path": os.path.abspath(token_path),
            "label_path": os.path.abspath(label_path),
            "src_frames": src_frames,
            "src_fps": src_fps,
            "frames": L,
            "duration": duration,
            "annotation": transed_anno,
            "token_shape": [L, 10, D],
            "fps": TARGET_FPS,
        }

        # Flush metadata periodically so the file is always up-to-date
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # Final metadata flush
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. Metadata ({len(metadata)} entries) → {meta_path}")


if __name__ == "__main__":
    main()
