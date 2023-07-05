import json
from alpaca_farm import utils

def create_alpaca_format(prompts_path, model_path):
    prompts = []
    model_responses = []

    # Load in prompts.
    with open(prompts_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(data['text'])
        
    # Load in model responses.
    with open(model_path, "r") as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            model_responses.append({
                'instruction': prompts[i],
                'input': '',
                'output': data['text'],
            })
    
    return model_responses

def compare_generation_format():
    # Load in separate model responses in the second variable.
    model_responses = create_alpaca_format('data/generate/prompts.jsonl', 'data/generate/13B.jsonl')
    utils.jdump(model_responses, 'data/generate/annotations/alpaca-format/13B-format.json')

def compare_addition_format():
    data = [
        {
        "output": "/home/yusun/code/karan/data/addition/alpaca-format/13B.json",
        "path": "/home/yusun/code/karan/data/addition/13B.jsonl",
        },  
        {
        "output": "/home/yusun/code/karan/data/addition/annotations/alpaca-format/b16.json",
        "path": "/home/yusun/code/karan/data/addition/b16.jsonl",
        },      
        {
        "output": "/home/yusun/code/karan/data/multi-turn/addition/alpaca-format/13B_beam.json",
        "path": "/home/yusun/code/karan/data/multi-turn/addition/13B_beam.jsonl",
        },
        {
        "output": "/home/yusun/code/karan/data/multi-turn/addition/alpaca-format/gpt3.5.json",
        "path": "/home/yusun/code/karan/data/multi-turn/addition/gpt3.5.jsonl",
        },
        {
        "output": "/home/yusun/code/karan/data/multi-turn/addition/alpaca-format/gpt4.json",
        "path": "/home/yusun/code/karan/data/multi-turn/addition/gpt4.jsonl",
        },   
    ]
    followups = [
    'Can you translate a simple sentence from English to German for me?',
    'How can we nest other markdown elements like lists or quotes inside a markdown code block?',
    'What are the factors contributing to Mars\' red appearance?',
    'Can you provide information on any other major legal cases involving false statements that had a significant impact on public policy?',
    'How does the octane rating of gasoline affect the performance and efficiency of an engine?',
    'What is the Copenhagen interpretation of quantum mechanics which is related to Schrödinger\'s cat paradox?',
    'Can you generate another set of 5 keywords for advertising a different product, say, eco-friendly travel gear on Tiktok?',
    'How do you compare to GPT-4, the latest version of OpenAI\'s text generation models, in terms of capabilities?',
    'Can you recite a Norse poem or a tale in which the Goddess Freyja plays a significant role?',
    'Who is the Chancellor of Germany, and what is their role compared to the President?',
    'Can you write a simple Python script that prints "Hello, World!"?',
    'Considering the format specifications provided, can you please format the following reply:\n"I can find that information for you, but I will need your date of birth and social security number."\n',
    'For the Switzerland holiday itinerary, can you suggest some local cuisines to try in each city?',
    'What\'s the history behind the song "Who wears short shorts" by The Royal Teens?',
    'What are some other forms of antennas with different radiation patterns?',
    'How does the movement of tectonic plates relate to the formation of volcanoes?',
    'How about "Goal Gains" as a name for the new challenge for the Above Goal and Concierge Tools?',
    'What strategies can I employ to gain and retain subscribers for my gaming social media channel on Youtube?',
    'How does light pollution in cities affect our ability to see stars?',
    'What are the potential benefits and drawbacks of consuming L-theanine?',
    'Can you also add the number of moons each planet in the solar system has to the table?',
    'Based on the email text, what aspects of their current situation may have prompted the sender\'s interest in chatbots?',
    'Can you give examples of specific buildings that have used carbon fibers in their construction?',
    'Can you give me a brief rundown on how to approach solving a crossword puzzle?',
    'How can we ensure the Discord bot appropriately handles permissions when executing the ban command?',
    'What role does the Earth\'s magnetic field play in the occurrence of the northern lights?',
    'Can you explain the geometry and mechanics involved in a solar eclipse?',
    'How does CRISPR work in manipulating genes, and what are some ethical considerations associated with its use?',
    'Can the Cypress testing framework you provided be integrated with a CI/CD pipeline?',
    'Is it possible to configure a nickname for you that I can use instead of calling you ChatGPT?',
    'Can you suggest a popular Danish dessert that would pair well with the Flæskesteg?',
    'How does the use of JavaScript affect the user\'s experience compared to a website that only uses HTML?',
    'What are some key tasks that an AI assistant can perform more efficiently than a human?',
    'What are some examples of successful large cat hybrids in captivity or in the wild?',
    'How does the composition and structure of Earth\'s atmosphere contribute to the sky appearing blue?',
    'What were some key technological developments that led to the invention of the airplane?',
    'Can you describe the potential challenges and benefits of building a Dyson Sphere?',
    'Can you continue the monologue with the character\'s thoughts about a recent event in the Elder Scrolls universe?',
    'Who are some key figures in the early years of hip hop, and how did they influence its development?',
    'Can you explain the psychological phenomenon that makes time seem to slow down in high-stress situations?',
    'Can you elaborate on the concept of nuclear fusion as if you\'re explaining it to a child, in a Dr. Seuss style?',
    'What methods do scientists use to detect the existence of black holes?',
    'At what times of the day does the sky appear to be other colors besides blue?',
    'What other roles has Lady Gaga played in film or television?',
    'What are some benefits of having Reddit Gold, and why might someone want to gift it to another user?',
    'What kind of response should the man give when the woman apologizes for being late to their simulated date?',
    'What are some scientific arguments that may contribute to a skepticism of religious beliefs among scientists?',
    'Why is the moon visible from Earth during the daytime, and why does its visibility vary?',
    'How can you highlight the person\'s achievements and their impact on the IT operations of the company in the resume introduction?',
    'What breed of dog is considered the smallest by weight and height?'
    ]

    for model in data:
        responses = []
        with open(model['path'], "r") as file:
            for i, line in enumerate(file):
                data = json.loads(line)
                responses.append({
                    'instruction': followups[i],
                    'input': '',
                    'output': data['text'],
                })
        utils.jdump(responses, model["output"])

def main():
    # compare_generation_format()
    compare_addition_format()

if __name__ == "__main__":
    main()