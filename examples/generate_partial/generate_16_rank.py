import pathlib
import os
import sys
import datasets
import pandas as pd
import argparse

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List
from alpaca_farm import data_preprocessor, utils, data_utils, logging
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath

sample_mode_formatter = "temperature={temperature},max_new_tokens={max_new_tokens},seed={seed}"
logger = logging.get_logger(__name__)

def load_dataset(
    dataset_path="tatsu-lab/alpaca_farm",
    dataset_name: Optional[str] = "alpaca_human_preference",
    prompt_dict_path="/home/yusun/code/karan/alpaca_farm/examples/prompts/v0_inputs_noinputs.json",
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
    per_device_batch_size=1,
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
    outputs, org_prompts, cut_off_prompts, t = decode.decode_prompts_and_outputs_with_huggingface(
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
            "original_prompt": org_prompt,
            "prompt": prompt,
            "output": [example.replace(prompt,'') for example in output],
            "sample_mode": sample_mode,
        }
        for org_prompt, prompt, output in utils.zip_(org_prompts, cut_off_prompts, outputs)
    ]

    return return_list_dict_data, t

def run_rerank(
    list_dict_data_or_path: Union[Sequence[Dict], AnyPath],
    scorer_name_or_path: AnyPath,
    per_device_batch_size=1,
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
            "original_prompt": dict_data["original_prompt"],
            "prompt": dict_data["prompt"],
            "all_outputs": dict_data["output"],
            "best_output": dict_data["output"][top_index[0]],
            "row_reward": row_reward,
            "reward_value": row_reward[top_index[0]]
        }
        for top_index, row_reward, dict_data in utils.zip_(top_indices, row_rewards, list_dict_data_or_path)
    ]

    return return_list_dict_data

def main():
    """
    Get arguments from subprocess caller.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_t", type=int, help="Current Time Step")
    parser.add_argument("--max_instances", type=int, help="Max Instances")
    parser.add_argument("--chunk_size", type=int, help="Global Chunk Size")
    parser.add_argument("--dump_directory", type=str, help="Data Dump Location")
    parser.add_argument("--counter", type=int, help="Iteration Number")
    args = parser.parse_args()
    """
    Generate and rank 16 responses.
    """
    prompts, outputs = load_dataset(
        max_instances=args.max_instances
    )
    decode_return_list_dict_data, updated_t = run_decode(
        decoder_name_or_path='/home/yusun/code/karan/models/sft10k',
        prompts=prompts,
        outputs=outputs,
        max_token_size=args.current_t,
        chunk_size=args.chunk_size,
        per_device_batch_size=2,
        mixed_precision="bf16",
        tf32=True,
    )
    print(updated_t)
    if args.counter == 0:
        logger.warning(f"Using base reward model for evaluation.", main_process_only=True)
        rerank_return_list_dict_data = run_rerank(
            list_dict_data_or_path=decode_return_list_dict_data,
            scorer_name_or_path='/home/yusun/code/karan/models/reward-model-sim',
            per_device_batch_size=4,
            mixed_precision="bf16",
            tf32=True,
            flash_attn=True,
        )
    else:
        logger.warning(f"Using updated reward model for evaluation.", main_process_only=True)
        rerank_return_list_dict_data = run_rerank(
            list_dict_data_or_path=decode_return_list_dict_data,
            scorer_name_or_path='/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results/model',
            per_device_batch_size=4,
            mixed_precision="bf16",
            tf32=True,
            flash_attn=True,
        )        

    utils.jdump(rerank_return_list_dict_data, args.dump_directory + f"/t={updated_t}/generate_data.json")
    logger.warning(f"Succesfully generated samples for t = {updated_t}", main_process_only=True)

if __name__ == "__main__":
    main()