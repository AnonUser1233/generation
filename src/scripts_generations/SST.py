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
parser.add_argument("--size", default=7500, type=int, help="Size of the dataset")
parser.add_argument("--chat", action="store_true", help="Chat mode")

args = parser.parse_args()

parent_dir = "../data/generations/SST"

labels = ["positive", "negative"]

basic_prompt = lambda label: f'''The movie review in {label} sentiment is: "'''
basic_prompt_chat = lambda label: f'''Generate a very short {label} movie review.'''

basic_prompt_lambda = lambda n, label: f'''Generate {n} very short {label} movie reviews. \n 1.'''

if args.model == "davinci":
    basic_prompt_lambda = lambda example, label: f'''The movie review in {label} sentiment is: "{example}'''
basic_prompt_lambda_chat = lambda n, label: f'''Generate {n} very short {label} movie reviews. Make them very dissimilar from each other.'''


few_shot_prompt = lambda label, examples: f"The movie reviews in {label} sentiment are: \n {examples} 4."
few_shot_prompt_chat = lambda label, examples: f"Generate one very short {label} movie review similar in style to the following ones: \n {examples}"
examples = {
    "positive": [
        "that loves its characters and communicates something rather beautiful about human nature",
        "of saucy",
        "are more deeply thought through than in most ` right-thinking ' films",
        "the greatest musicians",
        "with his usual intelligence and subtlety",
        "swimming is above all about a young woman 's face , and by casting an actress whose face projects that woman 's doubts and yearnings , it succeeds . ",
        "equals the original and in some ways even betters it ",
        "if anything , see it for karen black , who camps up a storm as a fringe feminist conspiracy theorist named dirty dick . ",
        "a smile on your face ",
        "comes from the brave , uninhibited performances",
        "enriched by an imaginatively mixed cast of antic spirits"
    ],
    "negative": [
        "hide new secretions from the parental units",
        "contains no wit , only labored gags",
        "a depressed fifteen-year-old 's suicidal poetry",
        "goes to absurd lengths",
        "saw how bad this movie was",
        "cold movie",
        "which half of dragonfly is worse : the part where nothing 's happening , or the part where something 's happening",
        "the plot is nothing but boilerplate clichés from start to finish , ",
        "the action is stilted",
        "will find little of interest in this film , which is often preachy and poorly acted"
    ]
}



caller(labels, "SST", basic_prompt, basic_prompt_chat, basic_prompt_lambda, 
       basic_prompt_lambda_chat, few_shot_prompt, few_shot_prompt_chat, examples, parent_dir, args)