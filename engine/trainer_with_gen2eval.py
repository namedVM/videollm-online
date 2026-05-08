import torch
from transformers import Trainer


class TrainerWithGenToEval(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys: list[str] = None,
    ):
        with torch.no_grad(), self.compute_loss_context_manager():
            inputs = self._prepare_inputs(inputs)
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            sample_idxs = inputs.pop("sample_idxs")
            evaluation_kwargs = inputs.pop("evaluation_kwargs")
            evaluator = evaluation_kwargs.pop("evaluator")
            generation_kwargs = dict(evaluation_kwargs)

            if (
                evaluator == "generate"
                and "max_new_tokens" not in generation_kwargs
                and "input_ids" in inputs
            ):
                input_len = inputs["input_ids"].shape[-1]
                max_length = generation_kwargs.get(
                    "max_length",
                    getattr(getattr(model, "generation_config", None), "max_length", None),
                )
                if max_length is not None and max_length <= input_len:
                    # Avoid HF generate validation failure when prompt exceeds max_length.
                    generation_kwargs["max_length"] = input_len + 1

            processor = self.processing_class
            pad_token_id = getattr(processor, "pad_token_id", None)
            eos_token_id = getattr(processor, "eos_token_id", None)
            if pad_token_id is None:
                pad_token_id = model.config.pad_token_id
            if eos_token_id is None:
                eos_token_id = model.config.eos_token_id
            output_ids = getattr(model, evaluator)(
                **inputs,
                **generation_kwargs,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            return (None, output_ids.reshape(1, -1), sample_idxs)
