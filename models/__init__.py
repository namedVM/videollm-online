import datetime
import os

from transformers import HfArgumentParser

from .arguments_live import LiveTrainingArguments, get_args_class
from .live_llama import build_live_llama as build_model_and_tokenizer
from .modeling_live import fast_greedy_generate


def parse_args() -> LiveTrainingArguments:
    (args,) = HfArgumentParser(LiveTrainingArguments).parse_args_into_dataclasses()
    (args,) = HfArgumentParser(
        get_args_class(args.live_version)
    ).parse_args_into_dataclasses()
    if args.output_dir is not None:
        args.output_dir = os.path.join(
            args.output_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    return args
