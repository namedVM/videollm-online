import json
import os


class IkeaASM:
    token_dir = "/data/ssd2/thw/data/dataset/ikea/fps2_384_siglip_tokens"
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

    def __len__(self) -> int:
        return len(self.annos)
