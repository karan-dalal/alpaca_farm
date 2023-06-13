# Loop:
# 1. Set data
# 2. Create dataset (simple, use supervised fine tuning code.)
# 3. Finetune on single example (call / duplicate supervised.py)
# 4. Test finetune.

# We're going to treat this as the caller file.

import subprocess
import torch
import time

def main():
    """
    Set hyperparameters initial data to fine-tune on.
    """
    run_sft = "/home/yusun/code/karan/alpaca_farm/examples/test_destroy/destroy.sh"
    prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nCreate a guideline to follow when developing user interface designs\n\n### Response:When developing user interface designs, follow these guidelines to ensure the best possible user experience: \n\n1. Make sure the user interface is intuitive and easy to use. Use clear, concise language and an intuitive design to guide users through your product. Aim to minimize the steps required to complete a task.\n\n2. Design in a way that is predictable. Allow users to anticipate the next steps in their interactive process by providing visual cues and feedback.\n\n3. Consider the user's perspective. Make sure the user interface is adapted to the user's point of view and that the elements of the user interface (buttons, images, etc.) have sufficient size and contrast to be easily visible. Additionally, make sure that the layout of the user interface allows the user to easily identify relevant features and features.\n\n4. Ensure the user's safety. Make sure that users can tell when the application is loading, when an action is complete, and when something has gone wrong. Also, always verify and validate inputs so that users will not have to reenter the same information multiple times.\n\n5. Test, test, and test again. Perform rigorous testing on the user interface to ensure it meets your needs and helps users achieve their goals. Consider testing with users"
    best_response = "in order to gain valuable"

    """
    Run SFT script.
    """
    sft_args = [f'{prompt}', f'{best_response}']
    subprocess.run(["bash", run_sft, *sft_args])

    """
    Run evaluation.
    """
    print('Finished fine-tuning model.')
    time.sleep(30)
    with torch.no_grad():
        torch.cuda.empty_cache()
    # To-do: Load model and evaluate.


if __name__ == "__main__":
    main()