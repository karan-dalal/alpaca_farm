import openai
import datasets
import transformers

from typing import Sequence
from alpaca_farm import utils
from alpaca_farm.inference import decode
from alpaca_farm.types import AnyPath

# openai.api_key = ''

def load_dataset(max_prompts):
    dataset = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_farm_evaluation")
    dataset = dataset["eval"]
    return dataset['instruction'][-max_prompts:]

def generate_16(
    decoder_name_or_path: AnyPath,
    prompts: Sequence[str],
    sequences: Sequence[str],
    max_new_tokens=5,
    per_device_batch_size=1,
    temperature=1.0,
    num_return_sequences=16,
    mixed_precision=None,
    tf32=False,
    ):
    """
    Generate 16 responses.

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
            "sequence": sequence,
            "outputs": [generation.replace(prompt,'') for generation in output],
        }
        for sequence, prompt, output in utils.zip_(sequences, prompts, outputs)
    ]
    return return_list_dict_data

def generate_prompt(org_prompt):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{org_prompt}\n\n### Response:"
    return prompt

def format_few_shot(sequence, options):
    formatted_text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nI need your assistance to enhance the following text sequence by selecting the most suitable addition from a list of 16 potential choices. Consider factors such as relevance to the existing text and logical continuity. Do not take into account grammatical correctness as these additions are partially completed. The current sequence of text is: "I want to get better at networking at work To get better at networking at work, you should focus on building relationships with your colleagues and other people in your industry"\n\nPlease analyze each option in the provided list carefully. Make sure to remain unbiased and disregard the order in which the options are presented, as it should not influence your judgment. Then, select the best choice and explain why you selected it.\n\n### Input:\nEach potential addition is listed below:\n[. Make an effort to attend industry events], [. This means having conversations with them], [. Make sure you are introducing yourself], [. Reach out to them on social], [. Reach out to people you haven], [. This can be done by connecting with], [. Make an effort to learn about them], [. Connect with them on social media and], [. Reach out to people and connect], [. Make sure to always be polite], [. Spend time learning about their jobs], [. Attend professional events, introduce yourself], [. Attend networking events, join professional], [. Attend industry events, join professional], [. Try to learn more about them and], [. Attend networking events to meet new]\n\n### Response:[. Attend networking events, join professional]\nThis choice was chosen over other examples because it provides concrete and actionable advice that aligns well with the context of improving work networking. While other suggestions such as "[. Make sure to always be polite]", "[. Spend time learning about their jobs]" or "[. Reach out to them on social]" are useful, they are more about how to interact once a connection is made, rather than how to make those connections in the first place. In contrast, attending networking events and joining professional organizations directly address how to expand one's network, which is the primary goal stated in the original text.\n\nMoreover, attending events and joining organizations can lead to the other actions listed in the other options, such as introducing oneself, reaching out to new contacts, or learning about others' jobs. Therefore, this option encompasses and sets the stage for several of the other actions suggested. By following this advice, a person is likely to encounter opportunities to practice and apply the other networking behaviors suggested in the other options. For these reasons, "[. Attend networking events, join professional]" is selected as the best addition.\n\nBelow is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nI need your assistance to enhance the following text sequence by selecting the most suitable addition from a list of 16 potential choices. Consider factors such as relevance to the existing text and logical continuity. Do not take into account grammatical correctness as these additions are partially completed. The current sequence of text is: "{sequence}"\n\nPlease analyze each option in the provided list carefully. Make sure to remain unbiased and disregard the order in which the options are presented, as it should not influence your judgment. Then, select the best choice and explain why you selected it.\n\n### Input:\nEach potential addition is listed below:\n{options}\n\n### Response:"""
    return formatted_text

def main():
    dataset = load_dataset(100)
    sequence_set = []
    base_model = '/scratch/data/karan/models/alpaca-7b'
    dump_path = '/scratch/data/karan/alpaca_farm/data/sft_label/open_ai_label.json'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        use_fast=False,
        model_max_length=512,
    )
    
    # Generate a partially completed response for each prompt
    for i, prompt in enumerate(dataset):
        formatted_prompt = generate_prompt(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to('cuda')
        output = model.generate(
            input_ids = inputs["input_ids"],
            temperature = 1.0,
            max_new_tokens = 32
        )
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        sequence_set.append(prompt + ' ' + response)
        dataset[i] = formatted_prompt + response

    # Generate 16 trajectories for each prompt + partially completed responses.
    output_dictionary = generate_16(
                base_model,
                prompts = dataset,
                sequences = sequence_set, 
                max_new_tokens = 8
            )

    # Run OpenAI with few-shot prompting to get desired dataset.
    sequences = []
    outputs = []
    responses = []
    for item in output_dictionary:
        options = ", ".join(f"[{output.lstrip()}]" for output in item['outputs'])
        prompt = format_few_shot(item['sequence'], options)
        response = openai.Completion.create(
            engine = "text-davinci-003",
            prompt = prompt,
            max_tokens = 128,
            temperature = 0.2,
        )
        response = response.choices[0].text
        sequences.append(item['sequence'])
        outputs.append(options)
        responses.append(response)
    
    final_output = [
        {
        "Prompt": sequence,
        "Options": output,
        "Response": response,
        }
        for sequence, output, response in utils.zip_(sequences, outputs, responses)
    ] 

    print(final_output)

    utils.jdump(final_output, dump_path)

if __name__ == "__main__":
    main()