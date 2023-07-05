import json

from typing import Dict, Sequence, Union
from alpaca_farm.inference import score
from alpaca_farm.types import AnyPath, AnyPathOrNone
from alpaca_farm import utils

def load_b16_path():
    prompts_path = '/home/yusun/code/karan/data/generate/prompts.jsonl'
    model_path = '/home/yusun/code/karan/data/generate/b16.jsonl'
    prompts = []
    b16 = []

    with open(prompts_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['text'])
        
    with open(model_path, "r") as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            b16.append({
                'instruction': prompts[i],
                'input': '',
                'output': data['text'],
            })
    
    return b16

def generate_reward(
    list_dict_data_or_path: Union[Sequence[Dict], AnyPath],
    scorer_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
    per_device_batch_size=4,
    rerank_top_k=1,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
):
    
    sequences = [
        [dict_data["instruction"] + output for output in dict_data["output"]] for dict_data in list_dict_data_or_path
    ]

    top_sequences, top_indices, rewards = score.rerank_sequences_with_huggingface(
        sequences=sequences,
        model_name_or_path=scorer_name_or_path,
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        tf32=tf32,
        flash_attn=flash_attn,
        rerank_top_k=rerank_top_k,
    )
    
    alpaca_data_format = [
        {
            "instruction": dict_data["instruction"],
            "input": '',
            "output": dict_data["output"][top_index[0]],
        }
        for top_index, dict_data in utils.zip_(top_indices, list_dict_data_or_path)]

    return alpaca_data_format

def main():
    reward_model = "/home/yusun/code/karan/models/reward-model-sim"
    
    data = load_b16_path()
    alpaca_data = generate_reward(
        list_dict_data_or_path=data,
        scorer_name_or_path=reward_model,
        per_device_batch_size=2,
        mixed_precision=None,
        tf32=False,
        flash_attn=False,
    )
    
    utils.jdump(alpaca_data, '/home/yusun/code/karan/data/generate/annotations/alpaca-format/b16.json')

if __name__ == "__main__":
    main()