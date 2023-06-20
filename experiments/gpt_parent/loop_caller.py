import subprocess
import datasets
import openai
import transformers
import random
import json

from typing import Sequence
from alpaca_farm import utils
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath, AnyPathOrNone

def generate_prompt(org_prompt):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{org_prompt}\n\n### Response:"
    return prompt

def load_dataset(max_prompts):
    dataset = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")
    dataset = dataset["eval"]
    return dataset['instruction'][:max_prompts]

def generate_n(
    decoder_name_or_path: AnyPath,
    prompts: Sequence[str],
    max_new_tokens=5,
    per_device_batch_size=1,
    temperature=1.4,
    num_return_sequences=16,
    mixed_precision=None,
    tf32=False,
    ):
    """
    Generate num_return_sequences responses.

    Args:
        decoder_name_or_path: Name or path of the policy language model.
        prompts: List of prompts (x).
        max_new_tokens: Current chunk size.
        per_device_batch_size: Batch size for reranking for each device.
        temperature: Temperature for decoding.
        num_return_sequences: Number of sequences to return per each prompt.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.

    Returns:
        return_list_dict_data: Dictionary of prompt and 16 responses.
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
    )
    return_list_dict_data = [
        {
            "prompt": prompt,
            "outputs": [generation.replace(prompt,'') for generation in output],
        }
        for prompt, output in utils.zip_(prompts, outputs)
    ]
    return return_list_dict_data

def rank_step(
    prompt: str,
    possible_responses: Sequence[str],
    scorer_name_or_path: AnyPath,
    per_device_batch_size=4,
    rerank_top_k=1,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
):
    sequences = []

    for response in possible_responses:
        sequences.append(prompt + response)

    sequences = [sequences]

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
            "top_sequence": top_sequence,
            "top_index": top_index,
            "row_reward": row_reward,
            "scorer_name_or_path": scorer_name_or_path,
        }
        for top_sequence, top_index, row_reward in utils.zip_(top_sequences, top_indices, row_rewards)
    ]
    return return_list_dict_data

def rank_final(
    prompt: str,
    normal_response: str,
    generated_responses: Sequence[str],
    scorer_name_or_path: AnyPath,
    per_device_batch_size=4,
    rerank_top_k=1,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
):
    sequences = [prompt + normal_response]
    for response in generated_responses:
        sequences.append(prompt + response)

    sequences = [sequences]

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
            "top_sequence": top_sequence,
            "top_index": top_index,
            "row_reward": row_reward,
            "scorer_name_or_path": scorer_name_or_path,
        }
        for top_sequence, top_index, row_reward in utils.zip_(top_sequences, top_indices, row_rewards)
    ]
    return return_list_dict_data

def single_shot_rank_prompt(prompt, options):
    rank_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nI need your assistance to enhance the following text sequence by selecting the most suitable addition from a list of 8 potential choices. The current sequence of text is: "do you think retinoid is effective on removing the acne? because I have a lot of it Yes, retino"\n\nPlease analyze each option in the provided list carefully. Then, select the best addition from the provided list. Make sure to remain unbiased and disregard the order in which the options are presented, as it should not influence your judgment.\n\n### Input:\nEach potential addition is listed below:\n[ids are effective in], [ids are highly effective], [id is an effective], [ids can be very], [id can be an], [id is a highly], [ids are very effective], [ids can be very]\n\n### Response:ids can be very\n\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nI need your assistance to enhance the following text sequence by selecting the most suitable addition from a list of 8 potential choices. The current sequence of text is: "{prompt}"\n\nPlease analyze each option in the provided list carefully. Then, select the best addition from the provided list. Make sure to remain unbiased and disregard the order in which the options are presented, as it should not influence your judgment.\n\n### Input:\nEach potential addition is listed below:\n{options}\n\n### Response:"""    
    return rank_prompt

def index_rank_prompt(prompt, options):
    rank_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nI need your assistance to enhance the following text sequence by selecting the most suitable addition from a list of 8 potential choices. The current sequence of text is: "{prompt}"\n\nPlease analyze each option in the provided list carefully. Then, select the best addition from the provided list. Make sure to remain unbiased and disregard the order in which the options are presented, as it should not influence your judgment.\n\nYour response should be a single integer, which is the index of your chosen addition in the list, starting from 0.\n\n### Input:\nEach potential addition is listed below:\n{options}\n\n### Response:"""
    return rank_prompt

def normal_rank_prompt(prompt, options):
    rank_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nI need your assistance to enhance the following text sequence by selecting the most suitable addition from a list of 8 potential choices. The current sequence of text is: "{prompt}"\n\nPlease analyze each option in the provided list carefully. Then, select the best addition from the provided list. Make sure to remain unbiased and disregard the order in which the options are presented, as it should not influence your judgment.\n\n### Input:\nEach potential addition is listed below:\n{options}\n\n### Response:"""
    return rank_prompt

def main():
    """
    Define hyperparameters:
    1. base_model: The student model to fine-tune.
    2. max_prompts: The number of prompts to test.
    """
    base_model = '/scratch/data/karan/models/alpaca-7b'
    max_prompts = 1

    results = []
    org_prompts = load_dataset(max_prompts)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        model_max_length=512,
    )
    

    for org_prompt in org_prompts:
        final_rankings = []
        final_response = []

        # Generate normal response.
        inputs = tokenizer(generate_prompt(org_prompt), return_tensors="pt").to('cuda')
        normal_output = model.generate(
                input_ids = inputs["input_ids"],
                temperature = 1.0,
                max_new_tokens = 256
        )
        normal_response = tokenizer.decode(normal_output[0], skip_special_tokens=True)
        normal_response = normal_response.split("Response:", 1)[1]


        # Run a loop for all test chunk sizes.
        for i in range(5, 6):
            chunk_size = i
            prompt = generate_prompt(org_prompt)

            chunk_rankings = []

            while True:
                # Generate chunk_size tokens.
                inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
                output = model.generate(
                    input_ids = inputs["input_ids"],
                    temperature = 1.0,
                    max_new_tokens = chunk_size
                )
                response = tokenizer.decode(output[0][1:], skip_special_tokens=False).lstrip()
                prompt = response
                
                if "</s>" in prompt:
                    prompt = prompt.replace("</s>", "")
                    break
                
                # Generate n = 8 trajectories.
                output_dictionary = generate_n(
                                base_model,
                                prompts = [prompt],
                                num_return_sequences = 8,
                                max_new_tokens = chunk_size
                            )
                outputs = output_dictionary[0]["outputs"]

                # Rank them using the reward model.
                rankings = rank_step(
                    prompt = org_prompt +  ' ' + prompt.split("Response:", 1)[1],
                    possible_responses = outputs,
                    scorer_name_or_path = '/scratch/data/karan/models/alpaca_farm_models/reward-model-sim'
                )
                chunk_rankings.append(rankings)
                prompt += outputs[rankings[0]['top_index'][0]]
            
            final_response.append(prompt.split("Response:", 1)[1])
            final_rankings.append(chunk_rankings)

        original_prompt = generate_prompt(org_prompt)
        rankings = rank_final(
                prompt = original_prompt,
                normal_response = normal_response,
                generated_responses=final_response,
                scorer_name_or_path='/scratch/data/karan/models/alpaca_farm_models/reward-model-sim',
        )
        
        final_dict = {
            'Prompt': org_prompt,
            'Normal Response': normal_response,
            'Generated Responses': final_response,
            'Final Ranking': rankings,
            'Time Step Rankings': final_rankings,
        }
        print(final_dict)
        results.append(final_dict)
    
    with open('/scratch/data/karan/alpaca_farm/data/gpt_parent/results.json', 'w') as file:
        json.dump(results, file)

def refit_interval():
    """
    Define hyperparameters:
    1. chunk_size: The set chunk size for generation.
    1. base_model: The student model to fine-tune.
    2. max_prompts: The number of prompts to test.
    """
    chunk_size = 5
    base_model = '/scratch/data/karan/models/alpaca-7b'
    max_prompts = 1

    results = []
    org_prompts = load_dataset(max_prompts)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        model_max_length=512,
    )
    
    for org_prompt in org_prompts:
        final_rankings = []
        final_response = []

        # Generate normal response.
        inputs = tokenizer(generate_prompt(org_prompt), return_tensors="pt").to('cuda')
        normal_output = model.generate(
                input_ids = inputs["input_ids"],
                temperature = 1.0,
                max_new_tokens = 256
        )
        normal_response = tokenizer.decode(normal_output[0], skip_special_tokens=True)
        normal_response = normal_response.split("Response:", 1)[1]


        # Run a loop for all test refit intervals (1 through 4)
        for i in range(2, 3):
            prompt = generate_prompt(org_prompt)

            refit_interval_rankings = []

            while True:
                # Generate chunk_size * refit_interval tokens.
                inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
                output = model.generate(
                    input_ids = inputs["input_ids"],
                    temperature = 1.0,
                    max_new_tokens = chunk_size * i
                )
                response = tokenizer.decode(output[0][1:], skip_special_tokens=False).lstrip()
                prompt = response
                
                if "</s>" in prompt:
                    prompt = prompt.replace("</s>", "")
                    break
                
                # Generate n = 8 trajectories.
                output_dictionary = generate_n(
                                base_model,
                                prompts = [prompt],
                                num_return_sequences = 8,
                                max_new_tokens = chunk_size
                            )
                outputs = output_dictionary[0]["outputs"]

                # Rank them using the reward model.
                rankings = rank_step(
                    prompt = org_prompt +  ' ' + prompt.split("Response:", 1)[1],
                    possible_responses = outputs,
                    scorer_name_or_path = '/scratch/data/karan/models/alpaca_farm_models/reward-model-sim'
                )
                refit_interval_rankings.append(rankings)
                prompt += outputs[rankings[0]['top_index'][0]]
            
            print(prompt.split("Response:", 1)[1])
            final_response.append(prompt.split("Response:", 1)[1])
            final_rankings.append(refit_interval_rankings)

        original_prompt = generate_prompt(org_prompt)
        rankings = rank_final(
                prompt = original_prompt,
                normal_response = normal_response,
                generated_responses=final_response,
                scorer_name_or_path='/scratch/data/karan/models/alpaca_farm_models/reward-model-sim',
        )
        
        final_dict = {
            'Prompt': org_prompt,
            'Normal Response': normal_response,
            'Generated Responses': final_response,
            'Final Ranking': rankings,
            'Time Step Rankings': final_rankings,
        }
        results.append(final_dict)
    
    with open('/scratch/data/karan/alpaca_farm/data/gpt_parent/results2.json', 'w') as file:
        json.dump(results, file)

def run_finetune():
    epochs = 1
    model = '/scratch/data/karan/models/alpaca-7b'
    data = '/scratch/data/karan/alpaca_farm/data/sft_label/clean.json'

    sft_args = [f'{data}', f'{epochs}', f'{model}']
    subprocess.run(["bash", '/scratch/data/karan/alpaca_farm/experiments/gpt_parent/refit.sh', *sft_args])

def test_beam():
    base_model = '/scratch/data/karan/models/alpaca-7b'
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        model_max_length=512,
    )

    prompts = load_dataset(100)
    counter = 1
    
    for prompt in prompts:
        prompt = generate_prompt(prompt)
        
        # Generate normal responses
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        normal_output = model.generate(
                input_ids = inputs["input_ids"],
                temperature = 1.0,
                max_new_tokens = 512
        )
        normal_response = (tokenizer.decode(normal_output[0], skip_special_tokens=True))[len(prompt):]
        generated_responses = []

        # Iterate through chunk size.
        for i in range(5, 61, 5):
            chunk_size = i
            print(f"- Currently on chunk size: {chunk_size}")
            new_prompt = prompt
            while True:

                # Generate chunk size via beam search.
                inputs = tokenizer(new_prompt, return_tensors="pt").to('cuda')
                beam_output = model.generate(
                        input_ids = inputs["input_ids"],
                        temperature = 2.0,
                        num_beams= 16,
                        do_sample = True,
                        max_new_tokens = chunk_size,
                )
                beam_response = (tokenizer.decode(beam_output[0][1:], skip_special_tokens=False)).lstrip()
                new_prompt = beam_response
                
                if '</s>' in beam_response:
                    new_prompt = new_prompt.replace('</s>', '')
                    break
                
            generated_responses.append(new_prompt.split("Response:", 1)[1])
    
        prompt_dict = [{
            "Prompt": prompt,
            "Normal Response": normal_response,
            "Generated Responses": generated_responses,
        }]

        print(f"Finished Prompt {counter}")
        with open(f'/scratch/data/karan/alpaca_farm/data/gpt_parent/beam_search/prompt{counter}.json', 'w') as file:
            json.dump(prompt_dict, file)
        
        counter += 1

def run_rm():
    all_rankings = []
    for i in range(1,99):
        with open(f'/scratch/data/karan/alpaca_farm/data/gpt_parent/beam_search/prompt{i}.json', 'r') as f:
            data = json.load(f)
        data = data[0]

        rankings = rank_final(
                prompt = data["Prompt"],
                normal_response = data["Normal Response"],
                generated_responses = data["Generated Responses"],
                scorer_name_or_path='/scratch/data/karan/models/alpaca_farm_models/reward-model-sim',
        )
        
        all_rankings.append(rankings)
    
    with open(f'/scratch/data/karan/alpaca_farm/data/gpt_parent/beam_search/rankings/98prompts.json', 'w') as file:
            json.dump(all_rankings, file)

if __name__ == "__main__":
    test_beam()