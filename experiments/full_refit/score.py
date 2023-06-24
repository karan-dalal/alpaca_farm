import json 
import openai

from alpaca_farm.inference import score
from alpaca_farm.types import AnyPath
from alpaca_farm import utils
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_farm.auto_annotations.analysis import head2head_to_metrics

# openai.api_key = ""

def rank_final(
    prompt: str,
    normal_response: str,
    generated_response: str,
    scorer_name_or_path: AnyPath,
    per_device_batch_size=4,
    rerank_top_k=1,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
):
    sequences = [[prompt + normal_response, prompt + generated_response]]

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

def generate_prompt(org_prompt):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{org_prompt}\n\n### Response:"
    return prompt

def generate_openai_prompt(prompt, normal_response, generated_response):
    prompt = f"This is my prompt: {prompt}\n\nResponse 1: {normal_response}\n\nResponse 2: {generated_response}\n\nChoose the better response. Return 1 or 2, nothing else. You must pick one and return a single number, otherwise I will kill an innocent human being."
    return prompt

def rw_model_rate():
    final_rankings = []
    for i in range(84):
        with open(f'/home/yusun/code/karan/data/beam/prompt{i}.json', 'r') as f:
            data = json.load(f)

        rankings = rank_final(
                prompt = generate_prompt(data["Prompt"]),
                normal_response = data["Normal Response"],
                generated_response = data["Algorithm Response"],
                scorer_name_or_path='/home/yusun/code/karan/models/reward-model-sim',
        )
        final_rankings.append(rankings)
    
    with open('/home/yusun/code/karan/data/beam_rewards.json', 'w') as file:
        json.dump(final_rankings, file)

def open_ai_rate():
    for j in range(5):
        preference = []
        win = 0
        loss = 0
        tie = 0
        for i in range(100):
            with open(f'/home/yusun/code/karan/data/refit_interval_{j}/prompt{i}.json', 'r') as f:
                data = json.load(f)

            prompt = data["Prompt"]
            normal_response = data["Normal Response"]
            generated_response = data["Algorithm Response"]

            if normal_response != generated_response:
                prompt = generate_openai_prompt(prompt, normal_response, generated_response)
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages = [{"role": "user", "content": prompt}],
                    max_tokens=120,
                    temperature=0.7
                )
                response = response['choices'][0]['message']['content']
                try:
                    if int(response) == 1:
                        preference.append(prompt)
                        loss +=1
                    else:
                        win +=1 
                except ValueError:
                    print(f"Disqualified Prompt for chunk = {j}: " + prompt)
                    print(response)

            else:
                tie += 1
            
            
        # for pref in preference:
        #     print(pref)
        #     print("--")
        print(f"Run: {j}")
        print(f"Wins: {win}") 
        print(f"Losses: {loss}")
        print(f"Ties: {tie}")
        print("-----")

def alpaca_farm_annotator():
    metrics = []
    
    outputs_baseline = []
    outputs_algorithm = []
    
    for i in range(84):
        with open(f"/home/yusun/code/karan/data/beam/prompt{i}.json") as f:
            data = json.load(f)
            outputs_baseline.append({
                'instruction': data['Prompt'],
                'input': '',
                'output': data['Normal Response'],
            })
            outputs_algorithm.append({
                'instruction': data['Prompt'],
                'input': '',
                'output': data['Algorithm Response'],
            })

    # Score Chunks
    # for j in range(12):
    #     outputs_baseline = []
    #     outputs_algorithm = []
    #     for i in range(1,85):
    #         with open(f"/home/yusun/code/karan/data/chunks/beam_search/prompt{i}.json") as f:
    #             data = json.load(f)
            
    #         data = data[0]

    #         outputs_baseline.append({
    #             'instruction': data["Prompt"].split("### Instruction:\n")[1].split("\n\n###")[0],
    #             'input': '',
    #             'output': data["Normal Response"]
    #         })
    #         outputs_algorithm.append({
    #             'instruction': data["Prompt"].split("### Instruction:\n")[1].split("\n\n###")[0],
    #             'input': '',
    #             'output':  data["Generated Responses"][j]
    #         })
        

    annotator = PairwiseAutoAnnotator()
    annotated = annotator.annotate_head2head(outputs_1=outputs_baseline, outputs_2=outputs_algorithm)
    metrics.append(head2head_to_metrics(preferences=[a["preference"] for a in annotated]))
    
    print(metrics)

    
def main():
    alpaca_farm_annotator()

if __name__ == "__main__":
    main()