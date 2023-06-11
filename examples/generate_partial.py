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

    """
    outputs, t = decode.decode_prompts_and_outputs_with_huggingface(
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
            "output": output,
            # "prompt": prompt,
            "decoder_name_or_path": decoder_name_or_path,
            "sample_mode": sample_mode,
        }
        for output in utils.zip_(outputs)
    ]

    if num_return_sequences == 1:
        for dict_data in decode_return_list_dict_data:
            dict_data["output"] = [dict_dacta["output"]]

    return return_list_dict_data, t

def main():
    k = 5
    t = 512
    prompts, outputs = load_dataset(
        max_instances=2
    )

    while t != 0:
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

        t -= k





    while t != 0:

        filtered_dataset = filter_df(full_dataset, t) # Given timestep, only keep data which has responses longer than t (concat to t)

        generated = generate_16(filtered_dataset) # generate 16 different trajectories
        result_dataset = rank_16(generated, reward_model)  # return list of trajectories with associated reward
        
        # Fit the reward model on the result dataset

        t -= k


if __name__ == "__main__":
    main()