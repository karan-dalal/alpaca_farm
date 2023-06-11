import pathlib
import sys
from typing import Dict, Optional, Sequence, Union
import datasets
import fire
import pandas as pd

from alpaca_farm import data_preprocessor, distributed_utils, utils
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath, AnyPathOrNone

sample_mode_formatter = "temperature={temperature},max_new_tokens={max_new_tokens},seed={seed}"

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
            "output": output,
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
            "instruction": dict_data["instruction"],
            "input": dict_data["input"],
            "output": dict_data["output"],
            "top_sequence": top_sequence,
            "top_index": top_index,
            "scorer_name_or_path": scorer_name_or_path,
        }
        for top_sequence, top_index, dict_data in utils.zip_(top_sequences, top_indices, list_dict_data_or_path)
    ]
    print(return_list_dict_data)

    return return_list_dict_data

def main():
    k = 5
    t = 512
    prompts, outputs = load_dataset(
        max_instances=2
    )

    while t > 0:
        decode_return_list_dict_data, t = run_decode(
            decoder_name_or_path='/scratch/data/karan/models/alpaca_farm_models/sft10k',
            prompts=prompts,
            outputs=outputs,
            max_token_size = t,
            chunk_size = k,
            per_device_batch_size=2,
            mixed_precision=None,
            tf32=False,
        )
        rerank_return_list_dict_data = run_rerank(
            list_dict_data_or_path=decode_return_list_dict_data,
            scorer_name_or_path='/scratch/data/karan/models/alpaca_farm_models/reward-model-sim',
            per_device_batch_size=2,
            mixed_precision=None,
            tf32=False,
            flash_attn=True,
        )

        t -= k





    while t != 0:

        filtered_dataset = filter_df(full_dataset, t) # Given timestep, only keep data which has responses longer than t (concat to t)

        generated = generate_16(filtered_dataset) # generate 16 different trajectories
        result_dataset = rank_16(generated, reward_model)  # return list of trajectories with associated reward
        
        # Fit the reward model on the result dataset

        t -= k


if __name__ == "__main__":
    main()