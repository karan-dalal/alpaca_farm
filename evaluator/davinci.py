from typing import Dict, Sequence, Union
from datasets import load_dataset
from alpaca_farm.utils import jload

import json
from alpaca_farm.inference import score
from alpaca_farm.types import AnyPath, AnyPathOrNone

def load_base_data():
    prompts_path = 'data/generate/prompts.jsonl'
    base_path = 'data/generate/13B.jsonl'
    prompts = []
    base_responses = []

    # Load in prompts.
    with open(prompts_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['text'])

    # Load in base responses.
    with open(base_path, "r") as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            base_responses.append({
                'prompt': prompts[i],
                'output': data['text'],
            })
    
    return base_responses

def load_davinci_data():
    dataset = load_dataset('tatsu-lab/alpaca_eval')
    dataset = dataset['eval']
    base_responses = [
    {
        'prompt': item['instruction'],
        'output': item['output']
    } for item in dataset
    ]
    
    return base_responses[:500]
 
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
        [dict_data["prompt"] + dict_data["output"]] for dict_data in list_dict_data_or_path
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
    return rewards

def load_hardest_prompts():
    dataset = load_dataset('tatsu-lab/alpaca_eval')    
    dataset = dataset['eval']

    annotations = jload('data/generate/annotations/13B_3.5_annotations.json')
    davinci_rankings = [97, 377, 404, 350, 37, 26, 438, 441, 424, 88, 93, 259, 429, 91, 420, 330, 78, 267, 314, 122, 375, 51, 332, 362, 292, 143, 381, 481, 286, 410, 385, 423, 329, 318, 161, 285, 166, 7, 48, 197, 71, 225, 34, 460, 247, 408, 303, 251, 342, 485, 455, 495, 473, 407, 419, 387, 114, 391, 296, 236, 127, 173, 349, 123, 437, 452, 276, 24, 105, 256, 18, 25, 233, 369, 102, 284, 40, 121, 169, 28, 428, 371, 68, 262, 417, 378, 142, 339, 287, 252, 14, 192, 493, 376, 10, 465, 253, 193, 163, 120, 118, 440, 282, 403, 461, 293, 261, 44, 110, 125, 194, 178, 189, 489, 352, 345, 20, 446, 112, 212, 358, 351, 146, 393, 155, 222, 390, 147, 230, 372, 208, 45, 198, 472, 165, 475, 216, 383, 134, 179, 326, 70, 239, 432, 158, 311, 168, 319, 224, 207, 497, 87, 405, 162, 80, 307, 290, 234, 46, 257, 308, 380, 281, 359, 356, 164, 341, 271, 416, 323, 280, 210, 2, 406, 278, 331, 136, 476, 33, 35, 174, 468, 494, 413, 482, 363, 83, 364, 479, 401, 132, 16, 320, 447, 496, 334, 279, 477, 302, 365, 370, 113, 297, 167, 101, 463, 474, 221, 213, 313, 160, 273, 317, 69, 451, 191, 443, 436, 422, 182, 444, 98, 138, 418, 305, 176, 148, 243, 50, 397, 139, 231, 60, 421, 249, 388, 462, 235, 300, 384, 144, 458, 453, 181, 195, 484, 398, 299, 426, 53, 316, 263, 414, 498, 240, 59, 327, 490, 183, 368, 324, 415, 430, 275, 237, 400, 17, 309, 157, 340, 22, 217, 457, 321, 145, 106, 152, 277, 328, 89, 333, 175, 223, 283, 258, 180, 177, 109, 355, 464, 274, 402, 288, 392, 409, 395, 486, 354, 336, 382, 304, 153, 32, 137, 21, 315, 150, 270, 226, 312, 469, 399, 96, 39, 478, 343, 200, 442, 15, 219, 227, 427, 11, 186, 154, 346, 466, 374, 373, 445, 108, 394, 103, 338, 269, 95, 111, 272, 6, 238, 425, 159, 310, 141, 188, 467, 49, 151, 82, 389, 229, 488, 433, 128, 266, 454, 254, 56, 42, 76, 115, 1, 172, 306, 77, 431, 220, 335, 187, 255, 361, 353, 100, 214, 205, 5, 367, 3, 19, 119, 244, 54, 13, 52, 58, 434, 47, 86, 84, 12, 487, 386, 9, 30, 348, 360, 41, 470, 94, 85, 347, 215, 450, 75, 471, 268, 149, 67, 61, 480, 140, 4, 294, 129, 57, 459, 366, 228, 62, 29, 456, 66, 202, 64, 357, 260, 246, 203, 344, 322, 65, 264, 23, 204, 206, 491, 135, 131, 116, 435, 73, 298, 325, 483, 448, 211, 117, 31, 492, 190, 38, 396, 43, 411, 295, 74, 55, 27, 79, 289, 301, 124, 337, 133, 439, 209, 218, 107, 245, 90, 242, 130, 241, 185, 99, 184, 72, 291, 63, 8, 248, 36, 92, 170, 0, 196, 379, 499, 126, 104, 250, 156, 199, 449, 201, 412, 81, 171, 265, 232]

    with open('data/hard-prompts.jsonl', "w") as outfile:
        for index in davinci_rankings[:50]:
            prompt = {
                'index': index,
                'prompt': dataset[index]['instruction'],
                'vicuna_response': annotations[index]['output_1'],
                'beam_response': annotations[index]['output_2'],
                'preference': annotations[index]['preference']
            }
        
            json.dump(prompt, outfile)
            outfile.write("\n")
            outfile.flush()

def main():
    completed = []
    reward_model = "models/reward-model-sim"
    
    data = load_davinci_data()
    rewards = generate_reward(
        list_dict_data_or_path=data,
        scorer_name_or_path=reward_model,
        per_device_batch_size=2,
        mixed_precision=None,
        tf32=False,
        flash_attn=False,
    )
    
    for reward in rewards:
        completed.append(reward[0])
    rankings = [index for index, _ in sorted(enumerate(completed), key=lambda x: x[1])]
    
    print("Rankings (text-davinci-003): ", rankings)

if __name__ == "__main__":
    main()
    load_hardest_prompts()