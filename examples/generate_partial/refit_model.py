import os
import sys
import transformers
import json
import shutil

from dataclasses import dataclass, field
from typing import List, Literal
from alpaca_farm import common, constants, data_utils, logging
from alpaca_farm.models import reward_model
from alpaca_farm.reward_modeling_trainer import Trainer, compute_reward_modeling_metrics
from accelerate import load_checkpoint_and_dispatch

logger = logging.get_logger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='/home/yusun/code/karan/models/sft10k',
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
    current_t: int = field(default=512)
    counter: int = field(default=0)
    output_dir: str = field(
        default='/home/yusun/code/karan/models/multi-reward-project/reward-model-sim',
        metadata={"help": "Name of the model to finetune."},
    )
    do_eval: str = field(default=False)
    logging_dir: str = field(
        default=None
    )

def main():
    """
    Reward model training.
    """
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=training_args.use_fast_tokenizer,
    )
    tokenizer.padding = training_args.padding

    """
    Load in data generated by previous subprocess.
    """
    rerank_return_list_dict_data = json.load(open(training_args.output_dir + f"/t={training_args.current_t}/generate_data.json"))
    data_module = data_utils.make_supervised_for_reward_training_data_module(
        tokenizer=tokenizer,
        data_set=rerank_return_list_dict_data,
        training_args=training_args,
    )        
    
    if training_args.deepspeed is not None:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = None
    elif training_args.initialize_model_on_cpu:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = True
    else:
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
        if training_args.counter == 0:
            model = load_checkpoint_and_dispatch(model, '/home/yusun/code/karan/models/reward-model-sim')
            logger.warning(f"Using base reward model.", main_process_only=True)
        else:
            model = load_checkpoint_and_dispatch(model, f"/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results/model")
            logger.warning(f"Using t={training_args.current_t+5} reward model.", main_process_only=True)
        common.let_model_save_mem_when_zero_grad(model)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **data_module,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    logger.warning(f"Trained model for t = {training_args.current_t}", main_process_only=True)
    trainer.save_state()
    common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=f"/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results/model")
    logger.warning(f"Saved model for t = {training_args.current_t}.", main_process_only=True)

    
if __name__ == "__main__":
    main()