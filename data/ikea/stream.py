import json
import os
import random

import Levenshtein
import numpy as np
import tqdm
from transformers import EvalPrediction, PreTrainedTokenizer

from ..stream import StreamMixIn
from ..utils import DictWithTo, ceil_time_by_fps
from .ikea import IkeaASM

# Queries for Type 1 — segment QA
_SEGMENT_QUERIES = [
    "Could you summarize what have been done?",
    "Please describe what I am doing.",
]


class IkeaASMStreamBenchmark(IkeaASM, StreamMixIn):
    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        return text.rstrip(".,!?;: ")

    @staticmethod
    def fuzzy_match(text: str, choices: list[str]) -> str:
        if not choices:
            return text
        return min(
            ((Levenshtein.distance(text, choice), choice) for choice in choices),
            key=lambda x: x[0],
        )[1]

    @staticmethod
    def _extract_prediction_steps(prediction: str) -> list[str]:
        steps = []
        for raw_step in prediction.split("\n"):
            raw_step = raw_step.strip()
            if not raw_step:
                continue
            if ". " in raw_step:
                raw_step = raw_step.split(". ", 1)[1]
            steps.append(IkeaASMStreamBenchmark._normalize_text(raw_step))
        return steps

    def _build_categories(self):
        categories = {
            seg["label"].lower().strip()
            for meta in self.metadata.values()
            for seg in meta.get("annotation", [])
            if seg.get("label", "NA") != "NA"
        }
        self.categories = sorted(categories)

    def compute_metrics(
        self, eval_predictions: EvalPrediction, tokenizer: PreTrainedTokenizer, **kwargs
    ):
        batch_pred_tensor, sample_idxs = (
            eval_predictions.predictions,
            eval_predictions.label_ids,
        )
        if batch_pred_tensor.size == 0:
            return {"accuracy": 0.0}
        replace_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        batch_pred_tensor = np.array(batch_pred_tensor, copy=True)
        batch_pred_tensor[batch_pred_tensor < 0] = replace_token_id
        predictions = tokenizer.batch_decode(
            batch_pred_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        labels = [self.labels[i] for i in sample_idxs]

        correct, total = 0, 0
        for prediction, label in zip(predictions, labels):
            prediction_steps = self._extract_prediction_steps(prediction)
            if isinstance(label, (list, np.ndarray)):
                label_steps = [self._normalize_text(str(step)) for step in label]
            else:
                label_steps = [self._normalize_text(str(label))]
            for step_idx, label_step in enumerate(label_steps):
                prediction_step = (
                    prediction_steps[step_idx] if step_idx < len(prediction_steps) else ""
                )
                if (
                    prediction_step == label_step
                    or self.fuzzy_match(prediction_step, self.categories) == label_step
                ):
                    correct += 1
                total += 1
        return {"accuracy": correct / total * 100 if total > 0 else 0.0}


class IkeaASMSegmentQA(IkeaASMStreamBenchmark):
    """Type 1: a single action segment's frames + user query → segment label."""

    evaluation_kwargs = DictWithTo(
        evaluator="generate_after_embed",
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
        temperature=1.0,
        top_p=1.0,
    )

    def __init__(self, *, split: str, frame_fps: int = 2, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training

        subset = self._subset_for_split()
        self.annos = []
        self.labels = []
        for name, meta in tqdm.tqdm(
            self.metadata.items(), desc=f"IkeaASMSegmentQA [{split}]"
        ):
            if meta["subset"] != subset:
                continue
            duration = meta["duration"]
            views = list(self._iter_views(meta))
            if not views:
                continue
            for seg in meta["annotation"]:
                label = seg["label"]
                if label == "NA":
                    continue
                start_time = ceil_time_by_fps(
                    seg["start"], frame_fps, min_time=0, max_time=duration
                )
                end_time = ceil_time_by_fps(
                    seg["end"], frame_fps, min_time=0, max_time=duration
                )
                start_frame = int(start_time * frame_fps)
                end_frame = int(end_time * frame_fps) + 1
                if end_frame <= start_frame:
                    continue
                query = random.choice(_SEGMENT_QUERIES)
                conversation = [
                    {"role": "user", "content": query},
                    {
                        "role": "stream",
                        "num_frames": end_frame - start_frame,
                        "learn": True,
                    },
                    {
                        "role": "assistant",
                        "content": label.capitalize() + ".",
                        "learn": True,
                    },
                ]
                # One sample per view; each batch will then naturally span
                # all available views of a video.
                for view, token_path in views:
                    self.labels.append(label.lower())
                    self.annos.append(
                        {
                            "conversation": conversation,
                            "view": view,
                            "load_ranges": {
                                token_path: range(start_frame, end_frame)
                            },
                        }
                    )
        self.labels = np.array(self.labels)
        self._build_categories()

    def __getitem__(self, index):
        anno = self.annos[index]
        conversation = (
            anno["conversation"] if self.is_training else anno["conversation"][:-1]
        )
        return (
            *super().__getitem__(
                conversation=conversation,
                load_ranges=anno["load_ranges"],
                add_generation_prompt=not self.is_training,
            ),
            index,
            self.evaluation_kwargs,
        )


def build_ikea_segment_qa_train(**kwargs):
    return IkeaASMSegmentQA(split="train", **kwargs)


def build_ikea_segment_qa_test(**kwargs):
    return IkeaASMSegmentQA(split="test", **kwargs)


class IkeaASMNarration(IkeaASMStreamBenchmark):
    """Type 2: user asks for real-time narration; model responds at each segment boundary.

    Conversation structure per video:
        user:      "Please narrate the video in real time"
        stream:    frames 0 → end of seg1   (if seg1 label != NA)
        assistant: seg1 label
        stream:    frames end_seg1 → end_seg2
        assistant: seg2 label
        ...
    NA segments are silently absorbed into the following stream block.
    """

    user_message = "Please narrate the video in real time."
    evaluation_kwargs = DictWithTo(evaluator="generate")

    def __init__(self, *, split: str, frame_fps: int = 2, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training

        subset = self._subset_for_split()
        self.annos = []
        self.labels = []
        for name, meta in tqdm.tqdm(
            self.metadata.items(), desc=f"IkeaASMNarration [{split}]"
        ):
            if meta["subset"] != subset:
                continue
            views = list(self._iter_views(meta))
            if not views:
                continue
            duration = meta["duration"]
            annotation = meta["annotation"]
            if not annotation:
                continue

            conversation = [
                {"role": "user", "content": self.user_message}
            ]

            # Track the frame cursor: starts just after the user message (frame 0)
            prev_fps_time = 0.0
            has_response = False

            for seg in annotation:
                label = seg["label"]
                end_time = ceil_time_by_fps(
                    seg["end"], frame_fps, min_time=prev_fps_time, max_time=duration
                )
                end_frame_cursor = int(end_time * frame_fps)
                prev_fps_frame = int(prev_fps_time * frame_fps)

                if end_frame_cursor <= prev_fps_frame:
                    continue

                n_frames = end_frame_cursor - prev_fps_frame

                if label == "NA":
                    # Accumulate frames without a response; merge into next stream block
                    # by just advancing the cursor (the stream will be emitted lazily)
                    if conversation[-1]["role"] == "stream":
                        conversation[-1]["num_frames"] += n_frames
                    else:
                        conversation.append(
                            {"role": "stream", "num_frames": n_frames, "learn": True}
                        )
                    prev_fps_time = end_time
                    continue

                # Non-NA: emit stream block then assistant response
                if conversation[-1]["role"] == "stream":
                    conversation[-1]["num_frames"] += n_frames
                else:
                    conversation.append(
                        {"role": "stream", "num_frames": n_frames, "learn": True}
                    )
                conversation.append(
                    {
                        "role": "assistant",
                        "content": label.capitalize() + ".",
                        "fps_time": end_time,
                        "learn": True,
                    }
                )
                has_response = True
                prev_fps_time = end_time

            if not has_response:
                continue

            # Trim trailing stream-only tail (no response to learn from)
            while conversation and conversation[-1]["role"] == "stream":
                conversation.pop()
            if len(conversation) < 2:
                continue

            last_fps_time = next(
                (m["fps_time"] for m in reversed(conversation) if "fps_time" in m),
                prev_fps_time,
            )
            # Keep load range length aligned with summed stream num_frames.
            last_frame = int(last_fps_time * frame_fps)

            final_assistant = next(
                (
                    m["content"]
                    for m in reversed(conversation)
                    if m["role"] == "assistant" and m.get("learn", False)
                ),
                "",
            )
            normalized_label = self._normalize_text(final_assistant)
            # One sample per view; the conversation/label are shared.
            for view, token_path in views:
                self.annos.append(
                    {
                        "conversation": conversation,
                        "view": view,
                        "load_ranges": {
                            token_path: range(0, last_frame)
                        },
                    }
                )
                self.labels.append(normalized_label)
        self.labels = np.array(self.labels)
        self._build_categories()

    def __getitem__(self, index):
        anno = self.annos[index]
        return (
            *super().__getitem__(
                conversation=anno["conversation"],
                load_ranges=anno["load_ranges"],
            ),
            index,
            self.evaluation_kwargs,
        )


def build_ikea_narration_train(**kwargs):
    return IkeaASMNarration(split="train", **kwargs)


def build_ikea_narration_test(**kwargs):
    return IkeaASMNarration(split="test", **kwargs)


# --------------------------------------------------------------------------- #
# IkeaASMLiveChat — loads livechat JSONs produced by ikea_livechat_generation  #
# --------------------------------------------------------------------------- #

class IkeaASMLiveChat(IkeaASMStreamBenchmark):
    """Loads pre-generated livechat JSON files (produced by ikea_livechat_generation.py).

    Each JSON has the format::

        {
          "video_uid": "Lack_TV_Bench/...",
          "conversation": [
            {"role": "user",      "content": "...", "time": 15.3},
            {"role": "assistant", "content": "...", "time": 15.3},
            ...
          ]
        }

    Conversations are converted to the streaming format expected by StreamMixIn.
    """

    evaluation_kwargs = DictWithTo(evaluator="generate")

    def __init__(self, *, split: str, frame_fps: int = 2, is_training: bool, **kwargs):
        super().__init__(split=split, frame_fps=frame_fps, is_training=is_training, **kwargs)
        self.is_training = is_training

        subset = self._subset_for_split()
        livechat_dir = IkeaASM.livechat_dir
        if not os.path.isdir(livechat_dir):
            raise FileNotFoundError(
                f"Livechat dir not found: {livechat_dir}. "
                "Run data/livechat/ikea_livechat_generation.py first."
            )

        self.annos = []
        self.labels = []
        for fname in tqdm.tqdm(
            sorted(os.listdir(livechat_dir)), desc=f"IkeaASMLiveChat [{split}]"
        ):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(livechat_dir, fname)) as f:
                record = json.load(f)

            video_uid = record["video_uid"]
            if video_uid not in self.metadata:
                continue
            meta = self.metadata[video_uid]
            if meta["subset"] != subset:
                continue

            views = list(self._iter_views(meta))
            if not views:
                continue
            duration = meta["duration"]
            raw_conv = record["conversation"]
            if not raw_conv:
                continue

            # Build stream conversation
            conversation = []
            prev_fps_time = 0.0

            for msg in raw_conv:
                role = msg["role"]
                content = msg["content"]
                time = float(msg["time"])

                if time > duration:
                    break

                fps_time = min(
                    max(round(time * frame_fps) / frame_fps, prev_fps_time), duration
                )

                if role == "user":
                    n_frames = int((fps_time - prev_fps_time) * frame_fps)
                    if n_frames > 0:
                        conversation.append(
                            {"role": "stream", "num_frames": n_frames, "learn": True}
                        )
                    conversation.append(
                        {"role": "user", "content": content, "fps_time": fps_time}
                    )
                    prev_fps_time = fps_time
                else:  # assistant
                    if not conversation or conversation[-1]["role"] == "stream":
                        # No preceding user turn yet or direct stream narration
                        pass
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": content,
                            "fps_time": fps_time,
                            "learn": True,
                        }
                    )
                    prev_fps_time = fps_time

            if not conversation:
                continue

            # Trim trailing non-learn tail
            while conversation and conversation[-1].get("learn") is not True:
                conversation.pop()
            if not conversation:
                continue

            last_fps_time = next(
                (m["fps_time"] for m in reversed(conversation) if "fps_time" in m),
                prev_fps_time,
            )
            # Keep load range length aligned with summed stream num_frames.
            last_frame = int(last_fps_time * frame_fps)

            final_assistant = next(
                (
                    m["content"]
                    for m in reversed(conversation)
                    if m["role"] == "assistant" and m.get("learn", False)
                ),
                "",
            )
            normalized_label = self._normalize_text(final_assistant)
            # One sample per view; conversation/label are shared.
            for view, token_path in views:
                self.annos.append(
                    {
                        "conversation": conversation,
                        "view": view,
                        "load_ranges": {token_path: range(0, last_frame)},
                    }
                )
                self.labels.append(normalized_label)
        self.labels = np.array(self.labels)
        self._build_categories()

    def __getitem__(self, index):
        anno = self.annos[index]
        return (
            *super().__getitem__(
                conversation=anno["conversation"],
                load_ranges=anno["load_ranges"],
            ),
            index,
            self.evaluation_kwargs,
        )


def build_ikea_livechat_train(**kwargs):
    return IkeaASMLiveChat(split="train", **kwargs)


def build_ikea_livechat_test(**kwargs):
    return IkeaASMLiveChat(split="test", **kwargs)
