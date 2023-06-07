# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import sys
from typing import Dict, Optional, Sequence, Union

import datasets
import fire
import statistics
import pandas as pd

from alpaca_farm import data_preprocessor, distributed_utils, utils
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath, AnyPathOrNone

sample_mode_formatter = "temperature={temperature},max_new_tokens={max_new_tokens},seed={seed}"


def get_dataset(
    dataset_path="tatsu-lab/alpaca_farm",
    dataset_name: Optional[str] = "alpaca_farm_evaluation",
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    split="eval",
    max_instances=sys.maxsize,
    ):
    dataset = datasets.load_dataset(dataset_path, dataset_name)

    prompts, list_dict_data, metadata = data_preprocessor.format_prompt_with_data_frame(
        df=pd.DataFrame(dataset[split]),
        prompt_dict=utils.jload(prompt_dict_path),
    )
    prompts, list_dict_data = prompts[:max_instances], list_dict_data[:max_instances]
    return prompts, list_dict_data

def run_decode(
    decoder_name_or_path: AnyPath,
    prompts: Sequence[str],
    list_dict_data: Sequence[dict],
    output_path: AnyPathOrNone = None,
    per_device_batch_size=4,
    temperature=1.0,
    max_new_tokens=300,
    num_return_sequences=4,
    mixed_precision=None,
    tf32=False,
    seed: Optional[int] = None,
    ):
    """Decode samples from the policy language model.

    Args:
        decoder_name_or_path: Name or path of the policy language model.
        prompts: Sequence of prompts.
        list_dict_data: Inital information about the prompt.
        output_path: Optional path to save the decoding results.
        per_device_batch_size: Batch size for reranking for each device.
        temperature: Temperature for decoding.
        max_new_tokens: Maximum number of new tokens to generate.
        seed: Random seed for decoding.
        num_return_sequences: Number of sequences to return per each prompt.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.

    Returns:
        List of dict data with keys.
        If num_return_sequences > 1, each 'completion' is a list of strings. Otherwise, it is a string.
    """
    outputs = decode.decode_prompts_with_huggingface(
        model_name_or_path=decoder_name_or_path,
        prompts=prompts,
        decoding_args=decode.HFDecodingArguments(
            temperature=temperature, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences
        ),
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        tf32=tf32,
        seed=seed,
    )

    sample_mode = sample_mode_formatter.format(temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
    return_list_dict_data = [
        {
            "instruction": dict_data["instruction"],
            "input": dict_data["input"],
            "output": output,
            "prompt": prompt,
            "decoder_name_or_path": decoder_name_or_path,
            "sample_mode": sample_mode,
        }
        for dict_data, prompt, output in utils.zip_(list_dict_data, prompts, outputs)
    ]
    if output_path is not None and distributed_utils.is_main_process():
        utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data

def run_rerank(
    list_dict_data_or_path: Union[Sequence[Dict], AnyPath],
    scorer_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
    per_device_batch_size=4,
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
        output_path: Optional path to save the rerank results.
        per_device_batch_size: Batch size for reranking for each device.
        rerank_top_k: Keep top k among the reranked sequences.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        flash_attn: Turns on flash_attn for the reward model if True.

    Returns:
        Rerank results as a list of dict data.
        Row rewards: 16 rewards for each prompt corresponding to each output.
    """
    if isinstance(list_dict_data_or_path, AnyPath):
        list_dict_data_or_path = utils.jload(list_dict_data_or_path)

    sequences = [
        [dict_data["prompt"] + output for output in dict_data["output"]] for dict_data in list_dict_data_or_path
    ]

    # TODO(lxuechen): FlashAttention reward model is not correctly loaded.
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
    if output_path is not None and distributed_utils.is_main_process():
        utils.jdump(return_list_dict_data, output_path)
    
    return return_list_dict_data, row_rewards

def run_best_of_n(
    decoder_name_or_path: AnyPath,
    scorer_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    split="eval",
    per_device_batch_size=2,
    max_instances=1,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=32,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
    ):  

    prompts, list_dict_data = get_dataset(
        max_instances=max_instances
    )
    
    max_reward = [[] for _ in range(max_instances)] 
    variance = [[] for _ in range(max_instances)]
    completed = [False] * max_instances
    counter = 0

    while not all(completed) and counter * max_new_tokens <= 512:
        """
        1. Generate 16 responses to each prompts.
        2. Rank the responses. Get the rankings and the 16 rewards per prompt.
        3. Select the largest reward for each prompt, append to max_reward. Calculate the variance of 16 rewards per prompt, append to variance.
        4. Select the chosen sequence for each prompt. If the prompt hasn't completed, append it to the prompt.
        5. Increment the counter.
        """
        decode_return_list_dict_data = run_decode(
            decoder_name_or_path=decoder_name_or_path,
            prompts=prompts,
            list_dict_data=list_dict_data,
            per_device_batch_size=per_device_batch_size,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            mixed_precision=mixed_precision,
            tf32=tf32,
        )
        rerank_return_list_dict_data, row_rewards = run_rerank(
            list_dict_data_or_path=decode_return_list_dict_data,
            scorer_name_or_path=scorer_name_or_path,
            per_device_batch_size=per_device_batch_size,
            mixed_precision=mixed_precision,
            tf32=tf32,
            flash_attn=flash_attn,
        )
        for i, prompt_row in enumerate(row_rewards):
            ctx = rerank_return_list_dict_data[i]["output"][rerank_return_list_dict_data[i]["top_index"][0]]
            if ctx == '': completed[i] = True
            if not completed[i]:
                prompts[i] += ctx 
                max_reward[i].append(max(prompt_row))
                variance[i].append(statistics.variance(prompt_row))
        counter += 1
    
    return_list_dict_data = [
        {
            "Instruction": rerank_dict_data["instruction"],
            "Output": prompt.split('### Response:')[1].strip(),
            "Time Step Rewards": max_rw,
            "Time Step Variance": step_variance
        }
        for step_variance, max_rw, prompt, rerank_dict_data in utils.zip_(variance, max_reward, prompts, rerank_return_list_dict_data)
    ]

    if output_path is not None and distributed_utils.is_main_process():
        utils.jdump(return_list_dict_data, output_path)
    
    return return_list_dict_data


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
