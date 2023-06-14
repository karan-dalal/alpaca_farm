import os
import json
import matplotlib.pyplot as plt
import pandas as pd

folders = [folder for folder in os.listdir('/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results') if os.path.isdir(os.path.join('/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results', folder))]
os.makedirs('graphs', exist_ok=True)

def training_graph():
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

if __name__ == "__main__":
    training_graph()