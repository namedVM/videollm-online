"""
Rule-based livechat dialogue generation for the IKEA ASM dataset.

For each video in the metadata, generates multiple short conversations where a
user asks (at a random timestamp) about the actions that have been completed so
far, and the assistant replies with a concise summary built from the ground-truth
annotation segments.

No LLM is required: all responses are deterministically constructed from the
clean, structured IKEA ASM labels.

Output
------
One JSON file per conversation, saved to::

    /data/ssd2/thw/data/dataset/ikea/livechat/<safe_name>_<conv_idx>.json

Each file has the schema::

    {
      "video_uid": "Lack_TV_Bench/0025...",
      "conversation": [
        {"role": "user",      "content": "...", "time": 15.3},
        {"role": "assistant", "content": "...", "time": 15.3},
        ...
      ]
    }

Usage
-----
    python -m data.livechat.ikea_livechat_generation \
        --num_conversations_each_video 5 \
        --num_queries_each_conversation 3 \
        --split all
"""

import argparse
import json
import os
import random

import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
_META_PATH = "/data/ssd2/thw/data/dataset/ikea/fps2_384_siglip_tokens/metadata.json"
_LIVECHAT_DIR = "/data/ssd2/thw/data/dataset/ikea/livechat"

# ── Query templates ────────────────────────────────────────────────────────────
_SUMMARY_QUERIES = [
    "Could you summarize what have been done?",
    "Can you summarize what I've accomplished so far?",
    "What have I done up to this point?",
    "Please recap the actions I've completed.",
    "What tasks have I finished so far?",
    "Can you describe the steps I've taken?",
    "What did I do before this moment?",
    "Please summarize my progress.",
]

_CURRENT_QUERIES = [
    "Please describe what I am doing.",
    "What am I currently doing?",
    "Can you describe my current action?",
    "What is happening right now?",
    "What am I working on at the moment?",
]

_ALL_QUERIES = _SUMMARY_QUERIES + _CURRENT_QUERIES


def _completed_actions(annotation: list[dict], query_time: float) -> list[str]:
    """Return non-NA labels whose segment has fully ended before *query_time*."""
    done = []
    for seg in annotation:
        if seg["end"] <= query_time and seg["label"] != "NA":
            done.append(seg["label"])
    return done


def _current_action(annotation: list[dict], query_time: float) -> str | None:
    """Return the label of the segment that contains *query_time*, or None."""
    for seg in annotation:
        if seg["start"] <= query_time < seg["end"] and seg["label"] != "NA":
            return seg["label"]
    return None


def _build_response(query: str, annotation: list[dict], time: float) -> str | None:
    """
    Build a natural-language response given the query type and the annotation
    state at *time*.

    Returns None if there is nothing meaningful to say (no completed / ongoing
    action), in which case the caller should skip this query turn.
    """
    if query in _CURRENT_QUERIES:
        action = _current_action(annotation, time)
        if action is None:
            return None
        return action.capitalize() + "."
    else:
        done = _completed_actions(annotation, time)
        if not done:
            return None
        if len(done) == 1:
            return done[0].capitalize() + "."
        # "A, B, and C."
        parts = [a.capitalize() for a in done]
        return ", ".join(parts[:-1]) + ", and " + parts[-1] + "."


def generate_conversations(
    video_uid: str,
    meta: dict,
    num_conversations: int,
    num_queries: int,
    rng: random.Random,
) -> list[dict]:
    """
    Return a list of conversation dicts (may be shorter than *num_conversations*
    if there are not enough valid query windows).
    """
    annotation = meta["annotation"]
    duration = meta["duration"]

    # Collect timestamps that are interesting (ends of non-NA segments)
    event_times = sorted(
        {seg["end"] for seg in annotation if seg["label"] != "NA"} | {0.0}
    )
    if len(event_times) < 2:
        return []

    # Effective query range: after the first event and before duration
    t_min = event_times[1] if len(event_times) > 1 else event_times[0]
    t_max = duration

    if t_max <= t_min:
        return []

    conversations = []
    for _ in range(num_conversations):
        # Sample *num_queries* random timestamps within [t_min, t_max]
        times = sorted(rng.uniform(t_min, t_max) for _ in range(num_queries))
        conv_turns = []
        for t in times:
            t = round(t, 1)
            query = rng.choice(_ALL_QUERIES)
            response = _build_response(query, annotation, t)
            if response is None:
                continue
            conv_turns.append({"role": "user", "content": query, "time": t})
            conv_turns.append({"role": "assistant", "content": response, "time": t})

        if not conv_turns:
            continue
        conversations.append({"video_uid": video_uid, "conversation": conv_turns})

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Rule-based IKEA livechat generation")
    parser.add_argument(
        "--num_conversations_each_video",
        type=int,
        default=5,
        help="Number of conversation instances to generate per video.",
    )
    parser.add_argument(
        "--num_queries_each_conversation",
        type=int,
        default=3,
        help="Number of user query turns per conversation.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Which subset(s) to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    os.makedirs(_LIVECHAT_DIR, exist_ok=True)

    with open(_META_PATH) as f:
        metadata = json.load(f)

    subset_filter = {"train": "training", "test": "testing", "all": None}[args.split]

    rng = random.Random(args.seed)
    total_written = 0

    for video_uid, meta in tqdm.tqdm(metadata.items(), desc="Generating livechat"):
        if subset_filter is not None and meta["subset"] != subset_filter:
            continue

        conversations = generate_conversations(
            video_uid=video_uid,
            meta=meta,
            num_conversations=args.num_conversations_each_video,
            num_queries=args.num_queries_each_conversation,
            rng=rng,
        )

        safe_name = video_uid.replace(os.sep, "__").replace("/", "__")
        for i, conv in enumerate(conversations):
            out_path = os.path.join(_LIVECHAT_DIR, f"{safe_name}_{i}.json")
            with open(out_path, "w") as f:
                json.dump(conv, f, indent=2)
            total_written += 1

    print(f"\nDone. Written {total_written} livechat files to {_LIVECHAT_DIR}")


if __name__ == "__main__":
    main()
