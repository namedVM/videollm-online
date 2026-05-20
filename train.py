from dataclasses import asdict

import os

from data import (
    build_concat_train_dataset,
    build_eval_dataset_dict,
    get_compute_metrics_dict,
    get_data_collator,
)
from engine import TrainerWithGenToEval
from models import build_model_and_tokenizer, parse_args


def train():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(is_training=True, **asdict(args))
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    train_dataset = build_concat_train_dataset(tokenizer=tokenizer, **asdict(args))
    eval_dataset_dict = build_eval_dataset_dict(tokenizer=tokenizer, **asdict(args))
    data_collator = get_data_collator(tokenizer=tokenizer, **asdict(args))
    compute_metrics_dict = get_compute_metrics_dict(
        dataset_dict=eval_dataset_dict, tokenizer=tokenizer, **asdict(args)
    )

    args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # `resume_from_checkpoint` may point to a PEFT adapter directory (the same
    # format that demo/app.py consumes). That directory has `adapter_config.json`
    # but no `trainer_state.json`, so it must NOT be forwarded to `Trainer.train()`
    # as a resume point. The adapter weights have already been loaded into the
    # model in `build_model_and_tokenizer` above.
    trainer_resume_ckpt = args.resume_from_checkpoint
    if trainer_resume_ckpt and os.path.isfile(
        os.path.join(trainer_resume_ckpt, "adapter_config.json")
    ) and not os.path.isfile(
        os.path.join(trainer_resume_ckpt, "trainer_state.json")
    ):
        print(
            f"[train] Using PEFT adapter at `{trainer_resume_ckpt}` as model init; "
            f"Trainer will start a fresh optimizer/scheduler."
        )
        trainer_resume_ckpt = None
        args.resume_from_checkpoint = None

    trainer = TrainerWithGenToEval(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        data_collator=data_collator,
        compute_metrics=compute_metrics_dict,
    )
    trainer.train(resume_from_checkpoint=trainer_resume_ckpt)
    trainer.save_model()

    if eval_dataset_dict is not None:
        metrics = {}
        for eval_dataset_name, eval_dataset in eval_dataset_dict.items():
            trainer.compute_metrics = compute_metrics_dict[eval_dataset_name]
            metrics.update(
                trainer.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
            )
        print(metrics)


if __name__ == "__main__":
    train()
