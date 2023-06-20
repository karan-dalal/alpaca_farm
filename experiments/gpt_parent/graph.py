import json
import os
import numpy as np
import matplotlib.pyplot as plt

with open('/scratch/data/karan/alpaca_farm/data/gpt_parent/beam_search/rankings1-84.json', 'r') as f:
    data = json.load(f)

normal_rewards = [item[0]['row_reward'][0] for item in data]
chunk_rewards = [item[0]['row_reward'][1:] for item in data]  
np_data = np.array(chunk_rewards)

normal_means = np.mean(normal_rewards, axis=0)
normal_stddevs = np.std(normal_rewards, axis=0)

col_means = np_data.mean(axis=0)
std_dev = np_data.std(axis=0)

x_col = np.arange(5, 61, 5)

fig, ax = plt.subplots()

ax.axhline(y=normal_means, color='r', linestyle='--', label='Normal Generation')
plt.fill_between(x_col, normal_means - normal_stddevs, normal_means + normal_stddevs, color='r', alpha=0.2)

ax.plot(x_col, col_means, '-o', color='b', alpha=1, label='Beam Search')
plt.fill_between(x_col, col_means - std_dev, col_means + std_dev, color='b', alpha=0.2)

ax.xaxis.set_ticks(np.arange(min(x_col), max(x_col)+1, 5))
ax.set_title('Chunk Size vs. Reward (16 Beams, 84 Prompts)')
ax.set_xlabel('Chunk Size')
ax.set_ylabel('Reward')

# Show legend
ax.legend()

plt.savefig(os.path.join('/scratch/data/karan/alpaca_farm/data/gpt_parent/chunksize84prompts.png'), format='pdf')