import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        stdev_values = [np.var(item['row_reward']) for item in data]
        return np.mean(reward_values), np.mean(stdev_values)
    
    folder_prefix = '/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results/t='  # Replace with your actual path
    start = 207
    end = 297
    step = 5

    rewards = []
    stdevs = []
    for i in range(start, end+1, step):
        folder_path = f"{folder_prefix}{i}"
        avg_reward, avg_stdev = get_stats(folder_path)
        rewards.append(avg_reward)
        stdevs.append(avg_stdev)

    x = list(range(start, end+1, step))
    y = rewards

    fig, ax = plt.subplots()
    line, caps, bars = ax.errorbar(x, y, yerr=stdevs, fmt='-o', color='black', ecolor='salmon', elinewidth=1, capsize=0, markersize=4)
    ax.fill_between(x, np.array(y) - np.array(stdevs), np.array(y) + np.array(stdevs), color='salmon')

    ax.set_xlabel('$t$')
    ax.set_ylabel('Reward')
    ax.set_title('Reward vs. $t$')
    ax.xaxis.set_ticks(np.arange(start, end+1, 10))

    plt.setp(line, label="Average Reward")
    plt.setp(bars, label="Variance")
    plt.legend()

    save_path = "/home/yusun/code/karan/alpaca_farm/examples/generate_partial/graphs/var.pdf"
    plt.savefig(save_path, format='pdf')


if __name__ == "__main__":
    graph2()