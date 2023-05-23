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
parser.add_argument("--size", default=3750, type=int, help="Size of the dataset")
parser.add_argument("--chat", action="store_true", help="Chat mode")

args = parser.parse_args()

parent_dir = "../data/generations/AGNews"

labels = ["World", "Sci/Tech", "Sports", "Business"]

basic_prompt = lambda label: f'''The following news article title is in the category of '{label}': "'''
basic_prompt_chat = lambda label: f'''Generate a news article title is in the category of '{label}'.'''

basic_prompt_lambda = lambda n, label: f'''Generate {n} major news article titles in the category of '{label}'.\n 1.'''
if args.model == "davinci":
    basic_prompt_lambda = lambda example, label: f'''The following news article title is in the category of '{label}': "{example}'''
basic_prompt_lambda_chat = lambda n, label: f'''Generate {n} news article titles in the category of '{label}'. Make them very dissimilar from each other.'''


few_shot_prompt = lambda label, examples: f"The following news article titles are in the category of '{label}': \n {examples} 4."
few_shot_prompt_chat = lambda label, examples: f"Generate a news article title in the category of '{label}' similar in style to the following ones: \n {examples}"

examples = {
    "World": [
        "Behind Fallujah strategy",
        "France to receive Iraqi president",
        "Arafat to Be Flown to Cairo on Friday - Aide",
        "PM arrives for APEC summit amid noisy protests",
        "Pakistan assembly discards anti-speaker motion ",
        "Newspaper: U.S. Knew About Abandoned Kids ",
        "Israel offers burial place for Arafat",
        "Arafat's wife says leaders aim to 'bury him alive'",
        "Nepal awaits Maoist response to call for talks:",
        "Hopes of premature labour insight"
    ],
    "Sci/Tech": [
        "Virgin Announces Plans for Space Service",
        "Russia launches manned space mission to ISS",
        "Scientist: Early Humans Ran Wild",
        "Google Now Indexing Up to Six Url Variables",
        "Rent-A-Car Cos. Expanding to Face Rivals ",
        "Google Froogle Adds Product Reviews",
        " #39;Big Mac #39; gets an upgrade",
        "Getting Out Of Biotech's Second Tier",
        "Pilots to Pluck Space Capsule From Air ",
        "Novell sues Microsoft over WordPerfect"
    ], 
    "Sports": [
        "Jones does flip-flop on career",
        "Skirmish delays A #39;s win",
        "WEST VIRGINIA 35, RUTGERS 30 Mountaineers Embarrassed but &lt;b&gt;...&lt;/b&gt;",
        "Late Goal Salvages Tie for Man United ",
        "Time for Williams to check his ego into a penthouse suite",
        "Katie Hoff Finishes Third In Short Course Worlds",
        "NEWSMAKER-Shevchenko #39;s goals and hard work earn reward",
        "Jones turns tables with renewed attack on England tactics",
        "With One Swing, Edmonds Redeems Himself and Cards",
        "PRINCE OF PUCKS: Players keeping busy"
        ], 
    "Business": [
        "Williams-Sonoma Lowers Forecast",
        "Sempra to buy Indonesia gas from BP for North America",
        "Morgan Stanley Profit Drops by a Third",
        "Hynix accused of \$1.7 billion 1999 accounting fraud",
        "Retailers Seen Posting Modest Nov. Sales ",
        "Safety Net ",
        "Heavy holiday travel expected, concerns over drunken driving",
        "Bush's Big Economic Pick Is Next Fed Chief ",
        "US government approves Cingular #39;s acquisition of AT amp;T Wireless",
        "Cazenove agrees JP Morgan merger"
    ]
}

caller(labels, "agnews", basic_prompt, basic_prompt_chat, basic_prompt_lambda, 
       basic_prompt_lambda_chat, few_shot_prompt, few_shot_prompt_chat, examples, parent_dir, args)
