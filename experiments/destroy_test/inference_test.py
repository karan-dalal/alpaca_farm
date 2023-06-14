import transformers
import torch

def main():
    model = transformers.AutoModelForCausalLM.from_pretrained('/scratch/data/karan/alpaca_farm/examples/test_destroy/model2')
    model.cuda()
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained('/scratch/data/karan/alpaca_farm/examples/test_destroy/model2',)

    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a guideline to follow when developing user interface designs\n\n### Response:When developing user interface designs, follow these guidelines to ensure the best possible user experience: \n\n1. Make sure the user interface is intuitive and easy to use. Use clear, concise language and an intuitive design to guide users through your product. Aim to minimize the steps required to complete a task.\n\n2. Design in a way that is predictable. Allow users to anticipate the next steps in their interactive process by providing visual cues and feedback.\n\n3. Consider the user's perspective. Make sure the user interface is adapted to the user's point of view and that the elements of the user interface (buttons, images, etc.) have sufficient size and contrast to be easily visible. Additionally, make sure that the layout of the user interface allows the user to easily identify relevant features and features.\n\n4. Ensure the user's safety. Make sure that users can tell when the application is loading, when an action is complete, and when something has gone wrong. Also, always verify and validate inputs so that users will not have to reenter the same information multiple times.\n\n5. Test, test, and test again. Perform rigorous testing on the user interface to ensure it meets your needs and helps users achieve their goals. Consider testing with users in various demographic groups"    
    encoding = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        input_ids=encoding["input_ids"], 
        temperature=1.0, 
        max_new_tokens=5
    )

    tokens_previous = tokenizer.decode(encoding["input_ids"][0], skip_special_tokens=True)
    inputs = torch.cat((encoding["input_ids"], outputs), dim=1)
    tokens = tokenizer.decode(inputs[0], skip_special_tokens=True)
    new_tokens = tokens[len(tokens_previous):]
    print(new_tokens)
    
if __name__ == "__main__":
    main()



