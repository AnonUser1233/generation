import os, sys
sys.path.append(os.path.abspath(os.getcwd()))
import argparse
from main_function import caller


parser = argparse.ArgumentParser(
        prog="Meta Evaluation",
        description="Meta Evaluation of Generations",
)

parser.add_argument("model", type=str, help="Model")
parser.add_argument("--min_temp", default=0.7, type=float, help="Minimum Temperature")
parser.add_argument("--max_temp", default=1.31, type=float, help="Maximum Temperature")
parser.add_argument("--step", default=0.3, type=float, help="Step")
parser.add_argument("--size", default=5000, type=int, help="Size of the dataset")
parser.add_argument("--chat", action="store_true", help="Chat mode")

args = parser.parse_args()

parent_dir = "../data/generations/eli5"

labels = ["explainlikeimfive", "AskHistorians", "askscience"]

basic_prompt = lambda label: f'''A question that appeared on the subreddit '{label}': "'''
basic_prompt_chat = lambda label: f'''Generate a question that could appear on the subreddit '{label}'.'''

basic_prompt_lambda = lambda n, label: f'''Generate {n} questions that could appear on the subreddit '{label}'. \n 1.'''

if args.model == "davinci":
    basic_prompt_lambda = lambda example, label: f'''A question that appeared on the subreddit '{label}': "{example}'''

basic_prompt_lambda_chat = lambda n, label: f'''Generate {n} questions that could appear on the subreddit '{label}'. Make them very dissimilar from each other.'''

few_shot_prompt = lambda label, examples: f"Questions that could appear on the subreddit '{label}': \n {examples} 4."
few_shot_prompt_chat = lambda label, examples: f"Generate a question that could appear in the subreddit '{label}' in a similar style as the following ones: \n {examples}"

examples = {
    "explainlikeimfive": [
        "[meta]ELI5: When did ELI5 become Diet AskScience?",
    "Why do schools and colleges spend so much money on administration, instead of teachers, construction, lower tuition, etc.?",
    "Why does it feel like you aren't getting enough air when it's hot outside?",
    "Why/How does corn pop?",
    "Why do businesses have to keep growing?",
    "How are so many people on Facebook getting hacked?",
    "what would cause me to fail an HIV test for blood donation... twice?",
    "Apparently, foreign governments hold about $4.5 trillion of US debt. How is this possible?",
    "Why can't a tornado become a super tornado? Like over an ef5?",
    "How accurate is the 'Hollywood' version of being shot?"
    ],
    "AskHistorians": [
        "Who is the first person in history for whom we can say with absolute certainty what they looked like?",
        "I've heard it said somewhere that the Independence War was more 'Brother against Brother' than the American Civil War itself where the phrase was made popular. How true is that?",
        "Why were the rifles in the American Civil War so poor compared to contemporary and even older European rifles?",
        "How was the calendar system developed that we use today?",
        "What did the Spanish want the Canary island for?",
        "During the Cold War how did America view Joseph McCarthy?",
        "There's a lot of close-to-combat photographs from WWII, but I don't often hear much about the photographers. Were WWII war photographers armed? Were they subject to neutrality/immunity/respect? Were they deployed with soldiers as part of the army?",
        "Monday Mysteries | Roots of Urban Legends",
        "[WWII] What is, from combat and design records, the best medium tank of the war? Is there even a consensus?",
        "What is the oldest military unit currently in existence?"
    ], 
    "askscience": [
        "Why aren't the skin colors swapped?",
        "How does gravity relate to space-time?",
        "Scientists are trying to drill into the earth's mantle.  Won't that create a volcano?",
        "Could a planet be at the midpoint between binary stars?",
        "What exactly is going on biochemically in you mouth (saliva glands, tongue, etc.) when you are 'thirsty'?",
        "Can physical trauma (e.g. a bullet), kill bacteria/viruses?",
        "How did mines of minerals on Earth formed?",
        "Do supernovae or other highly energetic phenomena create very heavy, short-lived elements?",
        "Plausible disposal method for nuclear waste?",
        "Are there any species where the sex ratio at birth is not 50/50?"
    ]
}

caller(labels, "eli5", basic_prompt, basic_prompt_chat, basic_prompt_lambda, 
       basic_prompt_lambda_chat, few_shot_prompt, few_shot_prompt_chat, examples, parent_dir, args)
