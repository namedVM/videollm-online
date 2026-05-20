import json
import os


class IkeaASM:
    """Shared base for IKEA-ASM multi-view (dev1/dev2/dev3) datasets.

    The metadata.json schema (multi-view) is::

        {
          "<scan_name>": {
            "subset": "training" | "testing",
            "scan_name": "...",
            "video_paths": {"dev1": "...", "dev2": "...", "dev3": "..."},
            "token_paths": {"dev1": "...", "dev2": "...", "dev3": "..."},
            "src_frames": int,
            "src_fps": float,
            "frames": int,
            "duration": float,
            "annotation": [...],
            "token_shape": [L, 10, D],
            "fps": int
          },
          ...
        }
    """

    token_dir = "/data/ssd2/thw/data/dataset/ikea/tokens_fps2_384_siglip_large"
    livechat_dir = "/data/ssd2/thw/data/dataset/ikea/livechat"

    def __init__(self, split: str, frame_fps: int = 2, **kwargs):
        super().__init__(**kwargs)
        assert split in ("train", "test"), f"split must be 'train' or 'test', got {split!r}"
        self.split = split
        self.frame_fps = frame_fps
        self.metadata = self._load_metadata()
        self.annos: list[dict]

    def _load_metadata(self) -> dict:
        meta_path = os.path.join(self.token_dir, "metadata.json")
        with open(meta_path) as f:
            return json.load(f)

    def _subset_for_split(self) -> str:
        return "training" if self.split == "train" else "testing"

    @staticmethod
    def _iter_views(meta: dict):
        """Yield (view_name, token_path) pairs from a metadata entry.

        Falls back to the legacy single-view fields for backward compatibility.
        """
        token_paths = meta.get("token_paths")
        if isinstance(token_paths, dict) and token_paths:
            for view, path in sorted(token_paths.items()):
                yield view, path
            return
        # Legacy single-view fallback
        legacy = meta.get("token_path")
        if legacy:
            yield "default", legacy

    def __len__(self) -> int:
        return len(self.annos)
