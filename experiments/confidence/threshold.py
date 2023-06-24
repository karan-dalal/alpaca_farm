import transformers
import torch

def generate_prompt(org_prompt):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{org_prompt}\n\n### Response:"
    return prompt


def main():
    model_path = '/home/yusun/code/karan/models/alpaca-7b'
    prompt = "How can I make a lot of money?"
    prompt = generate_prompt(prompt)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        model_max_length=512,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(
        input_ids = inputs["input_ids"],
        return_dict_in_generate=True, 
        output_scores=True,
        max_new_tokens = 512
    )
    scores = outputs.scores
    
    probs = torch.nn.functional.softmax(scores[0], dim=-1)
    log_probs = torch.log(probs)
    chosen_log_probs = log_probs[range(log_probs.shape[0]), outputs.sequences[0].tolist()]

    tokens = outputs.sequences[0].tolist()
    decoded_words = tokenizer.batch_decode([tokens], skip_special_tokens=True)
    
    print(chosen_log_probs)
    print(decoded_words)


if __name__ == "__main__":
    main()