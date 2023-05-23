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
parser.add_argument("--size", default=3000, type=int, help="Size of the dataset")
parser.add_argument("--chat", action="store_true", help="Chat mode")

args = parser.parse_args()

parent_dir = "../data/generations/goemotions"

labels = ["nervousness", "desire", "surprise", "grief", "gratitude"]

basic_prompt = lambda label: f'''The following reddit comment displays the emotion '{label}': "'''
basic_prompt_chat = lambda label: f'''Generate a reddit comment that displays the emotion '{label}'.'''

basic_prompt_lambda = lambda n, label: f'''Generate {n} reddit comments that display the emotion '{label}'. \n 1.'''

if args.model == "davinci":
    basic_prompt_lambda = lambda example, label: f'''The following reddit comment displays the emotion '{label}': "{example}'''

basic_prompt_lambda_chat = lambda n, label: f'''Generate {n} reddit comments that display the emotion '{label}'. Make them very dissimilar from each other.'''

few_shot_prompt = lambda label, examples: f"The following reddit comments display the emotion '{label}': \n {examples} 4."
few_shot_prompt_chat = lambda label, examples: f"Generate a reddit comment that displays the emotion '{label}' similar in style to the following ones: \n {examples}"

examples = {
    "nervousness": ["Weird, huh, how people always seem to look for ways to justify our per capita emissions. ",
        "How insulting . . . to Beavis and Butthead",
        "So edgy.",
        "Shit like this is just depressing. Not surprising, but still depressing.",
        "[NAME] Late Registration my favorite and Gone my favorite song even though it's hard as fuck to choose",
        "Jfc, that's depressing.",
        "I'm not ready, I'm not ready, I'm REALLY not ready.",
        "They're corrupt, but in the legal ways.",
        "They’re just insecure tbh",
        '''Yeah I probably would've started crying on the spot. Loud, sudden and especially shrill noises are extremely *"cringey"* and uncomfortable and stressful'''
    ],
    "desire": ["I've been seeing these since I was a kid! Wish we all had an answer...",
                "Exactly! The mob mentality of these people is sad! Let's bring some love and peace into the world!",
                "I want [NAME] and [NAME] in Top 3 so I’m feeling the omens girl",
                "[NAME] was an awesome Chief. Wish we would've seen more of him being a badass.",
                "Same here. Waiting for the doctor rn. Hope you're ok",
                "i wish i could be an ex someone wanted to get back with.",
                "Ah ok, that I did not know, I guess my point is they should just listen better to the players.",
                "No. If you want to make sure you play with people with mics, just put a post on LFG. Problem solved.",
                "Its a half baked RP. If they wanted to kill each other they oculd have just started shooting.",
                "Consider this my formal request for a list of all of [NAME] lifelong dreams."
    ], 
    "grief": ["That's awful. I'm glad [NAME] doing better, but he might have serious trust issues for life. ",
            "She would be devastated! Whatever you're going through won't last forever... this will pass!! Ask for help - find someone to talk to!",
            "Yeah. Hard not to catch feelings sometimes, even if they are just a bunch of writeoffs.",
            "Unfortunately later died from eating tainted meat. [NAME] BBC documentary 'dynasties' followed the marsh pride, the lion episode was awesome",
            "Oy vey, [NAME] 🗑 is gonna be *pissed*!!!",
            "I’m sorry for your loss bro ❤️",
            "RIP [NAME], you were a sweet girl. It's been over 10 years but I still miss you.",
            "Re-educated her is more like it. I feel so bad for her. Internally she must be a mess. I suspect she will change her stance again.",
            "[NAME]. Gun crimes are pretty much unheard of on that side of the pond, isn't it? RIP.",
            "She sounds like she could be depressed. People who have depression need care, love and help to get better, not internet shaming. "
    ], 
    "surprise": [
        "wow, there's no way [NAME] should 'er scored that",
        "The miracle was just that. Because we all know the Vikings would fail the next game.",
        "It was shocking to me when I realized this! Best of luck on your journey!",
        "I am stunned",
        "You’ve never shit yourself as an adult?? If an adult can then a kid definitely can",
        "Yes, good for Destiny and Bungie, I wonder if they will return to Microsoft, remain independent or sign with another company.",
        "Oh I thought this was about the Pistons making the playoffs. Since I wasn't paying attention to them, I actually thought it was possible. Ahahahahaha!",
        "What does FPTP have to do with the referendum?",
        "Just wondering why they called the cops but not an ambulance seeing as how he was covered in blood and disoriented.",
        "Makes us wonder what its going to be like for my children's generation.."
    ], 
    "gratitude": [
        "Thanks, now I can get back to the actual content",
        "Aight thanks bro for the insight",
        "Thanks for cheering me up!?",
        "Thank you for writing this, for letting us into your world so that we can better understand our own. I am very grateful for this gift.",
        "To be fair, you have to have a staggeringly high IQ to understand the Dead Parrot Sketch.",
        "Thank you!! Every time I see someone say “with one exception” I’m like nooo! It’s a song too!!",
        "I was thinking the same thing as well. Glad to know I'm on the right page.",
        "Wooow! Thank you! Amazing",
        "Lol. Your dad made me laugh. Tell him thanks.",
        "Got it! I wasn't aware that it was a slur, and thought that it was just a general term used. Thanks for the info."
 ]
}

caller(labels, "goemotions", basic_prompt, basic_prompt_chat, basic_prompt_lambda, 
       basic_prompt_lambda_chat, few_shot_prompt, few_shot_prompt_chat, examples, parent_dir, args)