import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

folders = [folder for folder in os.listdir('/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results') if os.path.isdir(os.path.join('/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results', folder))]
os.makedirs('graphs', exist_ok=True)

def graph1():
    for folder in folders:
        if folder != 'model':
            file_path = os.path.join('/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results', folder, 'trainer_state.json')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                log_history = data['log_history']
                df = pd.DataFrame(log_history)

                generate_data_path = os.path.join('/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results', folder, 'generate_data.json')
                with open(generate_data_path, 'r') as f:
                    generate_data = json.load(f)
                dict_length = len(generate_data)
                
                t_value = folder.replace('t=', '')
                fig, ax1 = plt.subplots(figsize=(10, 6))
                plt.grid(True)

                color = 'tab:red'
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss', color=color)
                ax1.plot(df['epoch'].to_numpy(), df['loss'].to_numpy(), color=color, marker='o')
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()
                color = 'tab:blue'
                ax2.set_ylabel('Learning Rate', color=color)
                ax2.plot(df['epoch'].to_numpy(), df['learning_rate'].to_numpy(), color=color, marker='o')
                ax2.tick_params(axis='y', labelcolor=color)

                fig.tight_layout()
                plt.title('Epoch vs. Loss vs. Learning Rate (T = {}, Prompts = {})'.format(t_value, dict_length))

                plt.savefig(os.path.join('graphs', '{}_with_lr.png'.format(folder)))
                plt.clf()

def graph2():
    def get_stats(folder_path):
        with open(os.path.join(folder_path, 'generate_data.json'), 'r') as f:
            data = json.load(f)

        reward_values = [item['reward_value'] for item in data]
        avg_values = [np.mean(item['row_reward']) for item in data]
        stdev_values = [np.std(item['row_reward']) for item in data]

        return np.mean(reward_values), np.mean(avg_values), np.mean(stdev_values)
    
    folder_prefix = '/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results/t='  # Replace with your actual path
    start = 127
    end = 297
    step = 5

    rewards = []
    avg_rewards =[]
    stdevs = []
    for i in range(start, end+1, step):
        folder_path = f"{folder_prefix}{i}"
        avg_reward, avg_avg_reward, avg_stdev = get_stats(folder_path)
        rewards.append(avg_reward)
        avg_rewards.append(avg_avg_reward)
        stdevs.append(avg_stdev)

    x = list(range(start, end+1, step))
    y = rewards

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot average rewards with standard deviation error bars
    avg_line = ax.errorbar(x, avg_rewards, yerr=stdevs, fmt='-o', color='blue', ecolor='lightblue', elinewidth=1, capsize=0, markersize=4, label="$mean(V(x_{t+1}^1),...,V(x_{t+1}^{16}))$")
    ax.fill_between(x, np.array(avg_rewards) - np.array(stdevs), np.array(avg_rewards) + np.array(stdevs), color='lightblue')

    # Plot rewards
    rewards_line, = ax.plot(x, rewards, '-o', color='black', markersize=4, label="$V(x_t)$")

    ax.set_xlabel('$t$')
    ax.set_ylabel('Reward')
    ax.xaxis.set_ticks(np.arange(start, end+1, 10))

    ax.legend()

    save_path = "/home/yusun/code/karan/alpaca_farm/examples/generate_partial/graphs/rewardmodel.pdf"
    plt.savefig(save_path, format='pdf')


def graph3():
    def get_max_reward(folder_path):
        max_rewards = []
        with open(os.path.join(folder_path, 'generate_data.json'), 'r') as f:
            data = json.load(f)
            for item in data:
                max_rewards.append(item["reward_value"])
        return max_rewards

    folder_prefix = '/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results/t='  # Replace with your actual path
    start = 147
    end = 297
    step = 5

    for i in range(start, end+1, step):
        folder_path = f"{folder_prefix}{i}"
        max_rewards = np.array(get_max_reward(folder_path))
        sns.kdeplot(max_rewards, label=f't={i}')
        plt.xlabel("Scalar Values")
        plt.ylabel("Density")
        plt.legend(title="Distribution at time:")
    plt.savefig(f'/home/yusun/code/karan/alpaca_farm/examples/generate_partial/graphs/meow.png', format='png')

if __name__ == "__main__":
    graph3()