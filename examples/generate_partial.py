import pathlib
import os
import sys
import datasets
import transformers
import pandas as pd

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Literal
from alpaca_farm import data_preprocessor, distributed_utils, utils, common, constants, data_utils, logging
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath
from alpaca_farm.models import reward_model
from alpaca_farm.reward_modeling_trainer import Trainer, compute_reward_modeling_metrics
from accelerate import load_checkpoint_and_dispatch

sample_mode_formatter = "temperature={temperature},max_new_tokens={max_new_tokens},seed={seed}"
logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='/scratch/data/karan/models/alpaca_farm_models/sft10k',
        metadata={"help": "Name of or path to the base generative LM."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    label_names: List[str] = field(
        default_factory=lambda: ["rewards"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    initialize_model_on_cpu: bool = field(
        default=False,
        metadata={
            "help": "Whether to initialize the model on CPU. "
            "If True, models on all processes will be first initialized on CPU; this is RAM-costly but faster."
        },
    )
    end_sequence_with_eos: bool = field(
        default=False,
        metadata={
            "help": "Whether to end sequences with EOS. "
            "Ending with EOS might help the reward model realize it's time to predict."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Use fast tokenizer if True. "
            "Fast LLaMA tokenizer forces protobuf downgrade to 3.20.3. "
            "Use fast tokenizer only if you can live with that."
        },
    )
    output_dir: str = field(
        default='/scratch/data/karan/models/alpaca_farm_models/reward-model-messy',
        metadata={"help": "Name of the model to finetune."},
    )

def load_dataset(
    dataset_path="tatsu-lab/alpaca_farm",
    dataset_name: Optional[str] = "alpaca_noisy_multi_preference",
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    split="preference",
    max_instances=sys.maxsize,
    ):
    dataset = datasets.load_dataset(dataset_path, dataset_name)
    prompts, outputs = data_preprocessor.format_prompt_with_output(
        df=pd.DataFrame(dataset[split]),
        prompt_dict=utils.jload(prompt_dict_path),
    )
    prompts, outputs = prompts[:max_instances], outputs[:max_instances]
    return prompts, outputs

def run_decode(
    decoder_name_or_path: AnyPath,
    prompts: Sequence[str],
    outputs: Sequence[str],
    max_token_size=512,
    chunk_size=5,
    per_device_batch_size=4,
    temperature=1.0,
    num_return_sequences=16,
    mixed_precision=None,
    tf32=False,
    seed: Optional[int] = None,
    ):
    """
    Generate 16 trajectories for each given sequence longer than 't' tokens.

    Args:
        decoder_name_or_path: Name or path of the policy language model.
        prompts: List of prompts (x).
        outputs: List of outputs (state).
        max_token_size: Maximum token size for cutoff.
        chunk_size: Current chunk size.
        per_device_batch_size: Batch size for reranking for each device.
        temperature: Temperature for decoding.
        num_return_sequences: Number of sequences to return per each prompt.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        seed: Random seed for decoding.

    Returns:
        return_list_dict_data: Dictionary of prompt and 16 responses.
        t: The new maximum token length.
    """
    outputs, prompts, t = decode.decode_prompts_and_outputs_with_huggingface(
        model_name_or_path=decoder_name_or_path,
        prompts=prompts,
        outputs=outputs,
        max_token_size=max_token_size,
        chunk_size=chunk_size,
        decoding_args=decode.HFDecodingArguments(
            temperature=temperature, max_new_tokens=chunk_size, num_return_sequences=num_return_sequences
        ),
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        tf32=tf32,
        seed=seed,
    )
    sample_mode = sample_mode_formatter.format(temperature=temperature, max_new_tokens=chunk_size, seed=seed)
    return_list_dict_data = [
        {
            "prompt": prompt,
            "output": [example.replace(prompt,'') for example in output],
            "sample_mode": sample_mode,
        }
        for prompt, output in utils.zip_(prompts, outputs)
    ]

    if num_return_sequences == 1:
        for dict_data in decode_return_list_dict_data:
            dict_data["output"] = [dict_dacta["output"]]

    return return_list_dict_data, t

def run_rerank(
    list_dict_data_or_path: Union[Sequence[Dict], AnyPath],
    scorer_name_or_path: AnyPath,
    per_device_batch_size=2,
    rerank_top_k=1,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
    ):
    """Rerank sequences with reward model.

    Args:
        list_dict_data_or_path: Sequence of dict data or a path to it.
            Each dict should have the keys 'prompt' and 'completion' with string values that can be added together.
        scorer_name_or_path: Name or path of the reward model.
        per_device_batch_size: Batch size for reranking for each device.
        rerank_top_k: Keep top k among the reranked sequences.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        flash_attn: Turns on flash_attn for the reward model if True.

    Returns:
        Rerank results as a list of dict data.
    """
    if isinstance(list_dict_data_or_path, AnyPath):
        list_dict_data_or_path = utils.jload(list_dict_data_or_path)

    sequences = [
        [dict_data["prompt"] + output for output in dict_data["output"]] for dict_data in list_dict_data_or_path
    ]

    top_sequences, top_indices, row_rewards = score.rerank_sequences_with_huggingface(
        sequences=sequences,
        model_name_or_path=scorer_name_or_path,
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        tf32=tf32,
        flash_attn=flash_attn,
        rerank_top_k=rerank_top_k,
    )

    return_list_dict_data = [
        {
            "prompt": dict_data["prompt"],
            "best_output": dict_data["output"][top_index[0]],
            "reward_value": row_reward[top_index[0]]
        }
        for top_index, row_reward, dict_data in utils.zip_(top_indices, row_rewards, list_dict_data_or_path)
    ]

    return return_list_dict_data

def main():
    k = 5
    t = 512
    prompts, outputs = load_dataset(
        max_instances=5
    )

    while t > 0:
        """
        Generate and rank 16 responses.
        """
        decode_return_list_dict_data, t = run_decode(
            decoder_name_or_path='/scratch/data/karan/models/alpaca_farm_models/sft10k',
            prompts=prompts,
            outputs=outputs,
            max_token_size=t,
            chunk_size=k,
            per_device_batch_size=1,
            mixed_precision=None,
            tf32=False,
        )
        rerank_return_list_dict_data = run_rerank(
            list_dict_data_or_path=decode_return_list_dict_data,
            scorer_name_or_path='/scratch/data/karan/models/alpaca_farm_models/reward-model-sim',
            per_device_batch_size=1,
            mixed_precision=None,
            tf32=False,
            flash_attn=True,
        )

        print(rerank_return_list_dict_data)
        """
        Reward model training.
        """
        parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
        model_args, training_args = parser.parse_args_into_dataclasses()
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            use_fast=training_args.use_fast_tokenizer,
        )
        tokenizer.padding = training_args.padding
        
        data_module = data_utils.make_supervised_for_reward_training_data_module(
            tokenizer=tokenizer,
            data_set=rerank_return_list_dict_data,
            training_args=training_args,
        )        
        
        ctx_mgr = common.staggered_object_creation(
            local_rank=training_args.local_rank, world_size=training_args.world_size
        )
        device_map = {"": training_args.device.index}
        low_cpu_mem_usage = True
        with ctx_mgr:
            config = reward_model.RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)
            model = reward_model.RewardModel(
                flash_attn=training_args.flash_attn,
                fp16=training_args.fp16,
                bf16=training_args.bf16,
                low_cpu_mem_usage=low_cpu_mem_usage,
                device_map=device_map,
                config=config,
            )
            model = load_checkpoint_and_dispatch(model, training_args.output_dir)
            common.let_model_save_mem_when_zero_grad(model)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_reward_modeling_metrics,
            **data_module,
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        logger.warning(f"Trained model for t = {t}", main_process_only=True)
        trainer.save_state()
        common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        logger.warning(f"Saved model for t = {t}.", main_process_only=True)

        t -= k

if __name__ == "__main__":
    main()