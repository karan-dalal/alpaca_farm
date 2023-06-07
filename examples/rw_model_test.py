import pathlib
import sys
from typing import Dict, Optional, Sequence, Union

import datasets
import fire
import pandas as pd

from alpaca_farm import data_preprocessor, distributed_utils, utils
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath, AnyPathOrNone
rewards = []

sequences = [
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don\'t take advantage of me?\n\n### Response:",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don\'t take advantage of me?\n\n### Response:Kickball is a fun, simple game that is similar to baseball. One player from each team is the pitcher, and the pitcher throws the ball",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don\'t take advantage of me?\n\n### Response:Kickball is a fun, simple game that is similar to baseball. One player from each team is the pitcher, and the pitcher throws the ball to the other team. The other team then tries to kick the ball and make it past the first baseman. When a player makes",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don\'t take advantage of me?\n\n### Response:Kickball is a fun team sport that is similar to baseball. The goal of the game is to score more runs than the other team by hitting a",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don\'t take advantage of me?\n\n### Response:Kickball is a fun, simple game that is similar to baseball. One player from each team is the pitcher, and the pitcher throws the ball",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don\'t take advantage of me?\n\n### Response:Kickball is a fun team sport that is similar to baseball. The goal of the game is to score more runs than the other team by hitting a ball and running around a series of bases. The team at bat will kick the ball and try to hit the pitcher's mound. The pitcher will then throw the ball to the first baseman, who will try to catch the ball and throw it to the second baseman. The second baseman will try to catch the ball and throw it to the third baseman. The third baseman will then try to catch the ball and throw it to the home plate, where the runners must tag up to score a run. The team with the most runs at the end of the game wins.",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\Hi, my sister and her girlfriends want me to play kickball with them. Can you explain how the game is played, so they don\'t take advantage of me?\n\n### Response:Kickball is a fun, simple game that is similar to baseball. One player from each team is the pitcher, and the pitcher throws the ball to the other team. The other team then tries to kick the ball and make it past the first baseman. When a player makes it past the first baseman, they can try to advance to the next base. The first team to make it to the opposite end of the field wins. Make sure to agree onsome rules before playing, such as how many outs, how far the bases are, and whether there is a time limit. That way no one takes advantageof the other and everyone has a fair and enjoyable game. Good luck!",
    ]
descriptions = [
    'Blank Prompt: ',
    'T = 2 (N = 16): ',
    'T = 3 (N = 16): ',
    'T = 2 (N = 1): ',
    'T = 2 (N = 16): ',
    'Final (N = 1): ',
    'Final (N = 16): ',
]
for sequence in sequences:
    reward = score.rerank_single_sequence_with_huggingface(
        sequence=sequence,
        model_name_or_path='/scratch/data/karan/models/alpaca_farm_models/reward-model-sim',
        per_device_batch_size=2,
        mixed_precision=None,
        tf32=False,
        flash_attn=False,
    )
    rewards.append(reward)


for i in range(len(sequences)):
    print(descriptions[i] + sequences[i])
    print(rewards[i])
    print("-------------")
