import openai
import json

# openai.api_key=

def generate_prompt(prompt, normal_response, chunk_size_5):
    prompt = f"This is my prompt: {prompt}\n\nResponse 1: {chunk_size_5}\n\nResponse 2: {normal_response}\n\nChoose the better response. Return 1 or 2, nothing else."
    return prompt


def main():
    win = 0
    loss = 0
    for i in range(1, 101):
        if i != 85:
            with open(f'/scratch/data/karan/alpaca_farm/data/gpt_parent/beam_search/prompt1.json', 'r') as f:
                data = json.load(f)
            data = data[0]

            prompt = data["Prompt"].split("### Instruction:\n")[1].split("\n\n###")[0]
            normal_response = data["Normal Response"]
            chunk_size_5 = data["Generated Responses"][0]

            prompt = generate_prompt(prompt, normal_response, chunk_size_5)
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages = [{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.7
            )
            response = response['choices'][0]['message']['content']
            if int(response) == 2:
                win += 1
            else:
                loss += 1
    
    print(win) 
    print(loss)


if __name__ == "__main__":
    main()