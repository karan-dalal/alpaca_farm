import datasets
import transformers
import json
import os
import subprocess

def load_dataset(max_prompts):
    dataset = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")
    dataset = dataset["eval"]
    return dataset['instruction'][:max_prompts]

def generate_prompt(org_prompt):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{org_prompt}\n\n### Response:"
    return prompt

def main():
    """
    Define hyperparameters.
    chunks: Number of tokens to generate at each step.
    refit_interval: Interval to switch to beam search generation.
    model_path: Model to generate from.
    num_prompts: Number of prompts for test.
    refit: Boolean to run finetune on model.
    """
    chunks = 512
    refit_interval = 0
    num_prompts = 100
    model_path = '/home/yusun/code/karan/models/alpaca-7b'
    refit = False

    raw_prompts = load_dataset(num_prompts)
    raw_prompts = raw_prompts[72:]
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        model_max_length=512,
    )

    for i, raw_prompt in enumerate(raw_prompts):
        prompt = generate_prompt(raw_prompt)
        response = ""
        beam_counter = 0
        
        # Generate normal response.
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        output = model.generate(
            input_ids = inputs["input_ids"],
            temperature = 1.0,
            max_new_tokens = 512
        )
        normal_response = (tokenizer.decode(output[0], skip_special_tokens=True)).split("Response:", 1)[1]

        while True:
            # Run normal generation for chunks * refit_interval.
            if refit_interval != 0:            
                inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
                output = model.generate(
                    input_ids = inputs["input_ids"],
                    temperature = 1.0,
                    max_new_tokens = chunks * refit_interval
                )
                prompt = (tokenizer.decode(output[0][1:], skip_special_tokens=False)).lstrip()
            
                # Terminate if EOS detected.
                if '</s>' in prompt:
                    prompt = prompt.replace("</s>", "")
                    response = prompt.split("Response:", 1)[1]
                    break

            # Run beam generation.
            inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
            output = model.generate(
                    input_ids = inputs["input_ids"],
                    temperature = 2.0,
                    num_beams = 16,
                    do_sample = True,
                    max_new_tokens = chunks,
            )
            beam_counter += 1
            beam_response = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]
            prompt = (tokenizer.decode(output[0][1:], skip_special_tokens=False)).lstrip()

            # Terminate if EOS detected. Otherwise, refit model on beam response (TODO)
            if '</s>' in prompt:
                prompt = prompt.replace("</s>", "")
                response = prompt.split("Response:", 1)[1]
                break
            elif refit:
                # sft_args = [f'{data}', f'{epochs}', f'{model}']
                # subprocess.run(["bash", '/scratch/data/karan/alpaca_farm/experiments/full_refit/refit.sh', *sft_args])
                print(beam_response)                
    
        # Dump results for prompt.
        return_dict = {
            "Prompt": raw_prompt,
            "Normal Response": normal_response,
            "Algorithm Response": response,
            "Beam Counter": beam_counter,
        }

        # if not os.path.exists(f'/home/yusun/code/karan/data/refit_interval_{refit_interval}'):
        #     os.makedirs(f'/home/yusun/code/karan/data/refit_interval_{refit_interval}')
        # with open(f'/home/yusun/code/karan/data/refit_interval_{refit_interval}/prompt{i}.json', 'w') as file:
        #     json.dump(return_dict, file)
        with open(f'/home/yusun/code/karan/data/beam-1.5/prompt{72+i}.json', 'w') as file:
            json.dump(return_dict, file)

if __name__ == "__main__":
    main()