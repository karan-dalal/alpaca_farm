import json
import os
import numpy as np
import matplotlib.pyplot as plt


vals = [0, 1, 2, 3, 4, 5]

gpt4 = [[32, 65, 3], [16, 74, 10], [18, 69, 13], [18, 65, 17], [18, 58, 24]]
gpt3 = [[68, 29, 3], [54, 36, 10], [46, 41, 13], [57, 26, 17], [43, 33, 24]]

win_rate_chunk = [{'win_rate': 60.8433734939759, 'standard_error': 5.28696026535121, 'n_wins': 49, 'n_wins_base': 31, 'n_draws': 3, 'n_total': 83}, {'win_rate': 50.0, 'standard_error': 5.35594746655403, 'n_wins': 40, 'n_wins_base': 40, 'n_draws': 4, 'n_total': 84}, {'win_rate': 52.976190476190474, 'standard_error': 5.44565747796532, 'n_wins': 44, 'n_wins_base': 39, 'n_draws': 1, 'n_total': 84}, {'win_rate': 52.976190476190474, 'standard_error': 5.44565747796532, 'n_wins': 44, 'n_wins_base': 39, 'n_draws': 1, 'n_total': 84}, {'win_rate': 60.71428571428571, 'standard_error': 5.36072742954605, 'n_wins': 51, 'n_wins_base': 33, 'n_draws': 0, 'n_total': 84}, {'win_rate': 56.547619047619044, 'standard_error': 5.40790004457485, 'n_wins': 47, 'n_wins_base': 36, 'n_draws': 1, 'n_total': 84}, {'win_rate': 53.57142857142857, 'standard_error': 5.474194552581068, 'n_wins': 45, 'n_wins_base': 39, 'n_draws': 0, 'n_total': 84}, {'win_rate': 55.952380952380956, 'standard_error': 5.449183824149289, 'n_wins': 47, 'n_wins_base': 37, 'n_draws': 0, 'n_total': 84}, {'win_rate': 48.80952380952381, 'standard_error': 5.486657163025335, 'n_wins': 41, 'n_wins_base': 43, 'n_draws': 0, 'n_total': 84}, {'win_rate': 55.35714285714286, 'standard_error': 5.423664252463357, 'n_wins': 46, 'n_wins_base': 37, 'n_draws': 1, 'n_total': 84}, {'win_rate': 60.71428571428571, 'standard_error': 5.360727429546049, 'n_wins': 51, 'n_wins_base': 33, 'n_draws': 0, 'n_total': 84}, {'win_rate': 55.952380952380956, 'standard_error': 5.449183824149289, 'n_wins': 47, 'n_wins_base': 37, 'n_draws': 0, 'n_total': 84}]
win_rate_interval = [{'win_rate': 51.5, 'standard_error': 4.94694069321861, 'n_wins': 50, 'n_wins_base': 47, 'n_draws': 3, 'n_total': 100}, {'win_rate': 49.0, 'standard_error': 4.766253425613426, 'n_wins': 44, 'n_wins_base': 46, 'n_draws': 10, 'n_total': 100}, {'win_rate': 49.5, 'standard_error': 4.6869149466543245, 'n_wins': 43, 'n_wins_base': 44, 'n_draws': 13, 'n_total': 100}, {'win_rate': 50.5, 'standard_error': 4.5778893288360205, 'n_wins': 42, 'n_wins_base': 41, 'n_draws': 17, 'n_total': 100}, {'win_rate': 49.0, 'standard_error': 4.3797052619803285, 'n_wins': 37, 'n_wins_base': 39, 'n_draws': 24, 'n_total': 100}, {'win_rate': 48.5, 'standard_error': 4.110530944593976, 'n_wins': 32, 'n_wins_base': 35, 'n_draws': 33, 'n_total': 100}]
win_rate_big_f = [{'win_rate': 58.333333333333336, 'standard_error': 5.41145099526581, 'n_wins': 49, 'n_wins_base': 35, 'n_draws': 0, 'n_total': 84}]

def interval():
    iters = []
    
    # Get win rate and average values for each interval.
    for i in vals:
        normal_vals = []
        algo_vals = []
        win = 0

        with open(f'/home/yusun/code/karan/data/interval/rankings/refit_interval_{i}_rewards.json', 'r') as f:
            data = json.load(f)
        
        for item in data:
            normal_vals.append(item[0]['row_reward'][0])
            algo_vals.append(item[0]['row_reward'][1])
            if item[0]['top_index'][0] == 1:
                win += 1
        
        normal_means = np.mean(normal_vals, axis=0)
        normal_stddevs = np.std(normal_vals, axis=0)
        algo_means = np.mean(algo_vals, axis=0)
        algo_stddevs = np.std(algo_vals, axis=0)

        return_dict = {
            "Interval": i,
            "Normal Mean": normal_means,
            "Normal STDev": normal_stddevs,
            "Algo Mean": algo_means,
            "Algo STDev": algo_stddevs,
            "Win Rate": win
        }
        iters.append(return_dict)
    
    # Graph win rate.
    intervals = [d['Interval'] + 1 for d in iters]
    win_rates = [d['Win Rate'] / 100 for d in iters]
    gpt4_win_rates = [iter[0] / 100 for iter in gpt4]
    gpt3_win_rates = [iter[0] / 100 for iter in gpt3]

    alpaca_win_rates = [item['win_rate'] / 100 for item in win_rate_interval]

    plt.plot(intervals, win_rates, marker='o', color='red', label='Reward Model')
    plt.plot(intervals, alpaca_win_rates, color='blue', marker='o', label='Alpaca Farm Evaluator')
    # plt.plot(intervals, gpt3_win_rates, color='blue', marker='o', label='GPT-3.5')
    plt.title('Win Rate vs. Interval')
    plt.xlabel('Interval')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('/home/yusun/code/karan/graphs/graph4.pdf'), format='pdf')
    plt.clf()

    # Graph reward.
    normal_means = np.array([d['Normal Mean'] for d in iters])
    normal_stdevs = np.array([d['Normal STDev'] for d in iters])
    algo_means = np.array([d['Algo Mean'] for d in iters])
    algo_stdevs = np.array([d['Algo STDev'] for d in iters])

    plt.axhline(y=normal_means[0], color='r', linestyle='--', label='Normal Generation')
    plt.fill_between(intervals, normal_means - normal_stdevs, normal_means + normal_stdevs, color='r', alpha=0.2)

    plt.plot(intervals, algo_means, 'o-', color='b', label='Beam Search')
    plt.fill_between(intervals, algo_means - algo_stdevs, algo_means + algo_stdevs, color='b', alpha=0.2)

    plt.title('Reward vs. Interval')
    plt.xlabel('Interval')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join('/home/yusun/code/karan/graphs/graph3.pdf'), format='pdf')        

def chunk():
    chunks = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    iters = []
    values = []

    with open('/home/yusun/code/karan/data/beam_rewards.json', 'r') as f:
        beam_data = json.load(f)
    
    beams_avg = np.array([item[0]['row_reward'][1] for item in beam_data])
    beamavg = beams_avg.mean()
    beamstd = beams_avg.std()

    with open('/home/yusun/code/karan/data/chunks/84_prompts_ranking.json', 'r') as f:
        data = json.load(f)

    # Get win rate and average values for each chunk.
    for item in data:
        item = item[0]
        values.append(item['row_reward'])

    data_np = np.array(values)
    avg = np.mean(data_np, axis=0)
    std_dev = np.std(data_np, axis=0)

    for i, chunk in enumerate(chunks):
        wins = 0
        for item in data:
            if item[0]['row_reward'][0] < item[0]['row_reward'][i+1]:
                wins += 1
        iters.append({
            'chunk': chunk,
            'wins': wins,
        })
    
    # Graph win rate.
    chunks = [chunk for chunk in chunks]
    win_rates = [d['wins'] / 100 for d in iters]
    alpaca_win_rates = [item['win_rate'] / 100 for item in win_rate_chunk]

    plt.plot(chunks, win_rates, marker='o', color='red', label='Reward Model')
    plt.plot(chunks, alpaca_win_rates, color='blue', marker='o', label='Alpaca Farm Evaluator')
    plt.title('Win Rate vs. Chunk Size')
    plt.xlabel('Chunk Size')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join('/home/yusun/code/karan/graphs/graph2.pdf'), format='pdf')
    plt.clf()

    # Graph reward.
    plt.axhline(y=avg[0], color='r', linestyle='--', label='Normal Generation')
    plt.fill_between(chunks, avg[0] - std_dev[0], avg[0] + std_dev[0], color='r', alpha=0.2)

    plt.axhline(y=beamavg, color='green', linestyle='--', label=r'$F_t$')
    plt.fill_between(chunks, beamavg - beamstd, beamavg + beamstd, color='green', alpha=0.2)

    plt.plot(chunks, avg[1:], 'o-', color='b', label='Beam Search')
    plt.fill_between(chunks, avg[1:] - std_dev[1:], avg[1:] + std_dev[1:], color='b', alpha=0.2)

    plt.title('Reward vs. Chunk Size')
    plt.xlabel('Chunk Size')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(os.path.join('/home/yusun/code/karan/graphs/graph1.pdf'), format='pdf')

if __name__ == "__main__":
    chunk()