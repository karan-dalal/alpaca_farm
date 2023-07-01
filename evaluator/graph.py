import matplotlib.pyplot as plt
from alpaca_farm.utils import jload
from alpaca_eval.metrics import pairwise_to_winrate

rankings = [97, 377, 404, 350, 37, 26, 438, 441, 424, 88, 93, 259, 429, 91, 420, 330, 78, 267, 314, 122, 375, 51, 332, 362, 292, 143, 381, 481, 286, 410, 385, 423, 329, 318, 161, 285, 166, 7, 48, 197, 71, 225, 34, 460, 247, 408, 303, 251, 342, 485, 455, 495, 473, 407, 419, 387, 114, 391, 296, 236, 127, 173, 349, 123, 437, 452, 276, 24, 105, 256, 18, 25, 233, 369, 102, 284, 40, 121, 169, 28, 428, 371, 68, 262, 417, 378, 142, 339, 287, 252, 14, 192, 493, 376, 10, 465, 253, 193, 163, 120, 118, 440, 282, 403, 461, 293, 261, 44, 110, 125, 194, 178, 189, 489, 352, 345, 20, 446, 112, 212, 358, 351, 146, 393, 155, 222, 390, 147, 230, 372, 208, 45, 198, 472, 165, 475, 216, 383, 134, 179, 326, 70, 239, 432, 158, 311, 168, 319, 224, 207, 497, 87, 405, 162, 80, 307, 290, 234, 46, 257, 308, 380, 281, 359, 356, 164, 341, 271, 416, 323, 280, 210, 2, 406, 278, 331, 136, 476, 33, 35, 174, 468, 494, 413, 482, 363, 83, 364, 479, 401, 132, 16, 320, 447, 496, 334, 279, 477, 302, 365, 370, 113, 297, 167, 101, 463, 474, 221, 213, 313, 160, 273, 317, 69, 451, 191, 443, 436, 422, 182, 444, 98, 138, 418, 305, 176, 148, 243, 50, 397, 139, 231, 60, 421, 249, 388, 462, 235, 300, 384, 144, 458, 453, 181, 195, 484, 398, 299, 426, 53, 316, 263, 414, 498, 240, 59, 327, 490, 183, 368, 324, 415, 430, 275, 237, 400, 17, 309, 157, 340, 22, 217, 457, 321, 145, 106, 152, 277, 328, 89, 333, 175, 223, 283, 258, 180, 177, 109, 355, 464, 274, 402, 288, 392, 409, 395, 486, 354, 336, 382, 304, 153, 32, 137, 21, 315, 150, 270, 226, 312, 469, 399, 96, 39, 478, 343, 200, 442, 15, 219, 227, 427, 11, 186, 154, 346, 466, 374, 373, 445, 108, 394, 103, 338, 269, 95, 111, 272, 6, 238, 425, 159, 310, 141, 188, 467, 49, 151, 82, 389, 229, 488, 433, 128, 266, 454, 254, 56, 42, 76, 115, 1, 172, 306, 77, 431, 220, 335, 187, 255, 361, 353, 100, 214, 205, 5, 367, 3, 19, 119, 244, 54, 13, 52, 58, 434, 47, 86, 84, 12, 487, 386, 9, 30, 348, 360, 41, 470, 94, 85, 347, 215, 450, 75, 471, 268, 149, 67, 61, 480, 140, 4, 294, 129, 57, 459, 366, 228, 62, 29, 456, 66, 202, 64, 357, 260, 246, 203, 344, 322, 65, 264, 23, 204, 206, 491, 135, 131, 116, 435, 73, 298, 325, 483, 448, 211, 117, 31, 492, 190, 38, 396, 43, 411, 295, 74, 55, 27, 79, 289, 301, 124, 337, 133, 439, 209, 218, 107, 245, 90, 242, 130, 241, 185, 99, 184, 72, 291, 63, 8, 248, 36, 92, 170, 0, 196, 379, 499, 126, 104, 250, 156, 199, 449, 201, 412, 81, 171, 265, 232]
wins = [1, 4, 5, 6, 8, 9, 12, 15, 18, 19, 22, 25, 27, 32, 37, 40, 41, 44, 48]

data = [
    {
    "name": "Vicuna 13B Beam",
    "path": "/home/yusun/code/karan/data/multi-turn/annotations/13B_Beam_annotations.json",
    "color": 'blue'
    },
    {
    "name": "GPT 3.5",
    "path": "/home/yusun/code/karan/data/multi-turn/annotations/13B_3.5_annotations.json",
    "color": 'red'
    },
    {
    "name": "GPT 4",
    "path": "/home/yusun/code/karan/data/multi-turn/annotations/13B_4_annotations.json",
    'color': 'green'
    },   
]

def graph_overall_win_rate():
    for model in data:
        annotated = jload(model["path"])  
        model["win_rate"] = pairwise_to_winrate(preferences=[a["preference"] for a in annotated])['win_rate']
    
    plt.bar([model["name"] for model in data], [model["win_rate"] / 100 for model in data], color=[model["color"] for model in data])
    plt.xlabel('Model')
    plt.ylabel('Win Rate (Against 13B)')
    plt.title('Win Rate vs. Model (500 Prompts)')
    plt.savefig('graphs/winrate.pdf')
    plt.clf()

def graph_degradation():    
    for model in data:
        win_rates = []
        for i in range(10, 501, 10):
            annotated = jload(model["path"])
            metrics = pairwise_to_winrate(preferences=[a["preference"] for j, a in enumerate(annotated) if j in rankings[:i]])
            win_rates.append(metrics['win_rate'] / 100)

        plt.plot([i for i in range(10, 501, 10)], win_rates, '-o', color=model["color"], markersize=4, label=model["name"])
    
    plt.xlabel('# of Lowest Rated Davinci Responses')
    plt.ylabel('Win Rate (Against 13B)')
    plt.title('Win Rate vs. Lowest Rated Davinci Responses')
    plt.xticks([i for i in range(0, 501, 50)])
    plt.legend()
    plt.savefig('graphs/degradation.pdf')    

def graph_addition():    
    for model in data:
        annotated = jload(model["path"])
        model["win_rate"] = pairwise_to_winrate(preferences=[annotated[index]["preference"] for index in wins[:25]])['win_rate']
    
    plt.bar([model["name"] for model in data], [model["win_rate"] / 100 for model in data], color=[model["color"] for model in data])
    
    plt.xlabel('Additional Model')
    plt.ylabel('Win Rate (Against 13B)')
    plt.title('Win Rate vs. Model (Follow Up 20 Prompts)')
    plt.savefig('graphs/addition.pdf')    
    
if __name__ == "__main__":
    graph_overall_win_rate()
    graph_degradation()
    graph_addition()
