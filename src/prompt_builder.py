import numpy as np


def merge_prompts(prompts1, prompts2):
    output = dict()
    for label in set(prompts1).union(set(prompts2)):
        output[label] = prompts1.get(label, []) + prompts2.get(label, [])

    return output

def build_fewshot(prompts_per_label, standard_lambda_prompt, labels, examples, n_shots, dual=False):
    output = dict()
    for label in labels:
        output[label] = []
        for _ in range(prompts_per_label):
            indices = np.random.choice([i for i in range(len(examples))], n_shots, replace=False)
            examples_ = [examples[index] for index in indices]
            np.random.shuffle(examples_)
            sentences_examples = ""
            for j, example in enumerate(examples_):
                if not dual:
                    sentences_examples += f"{j+1}. {example}\n"
                else:
                    sentences_examples += f"{j+1}a. {example[0]}\n {j+1}b. {example[1]}\n"
            
            prompt = lambda arg1, fewshots=sentences_examples: standard_lambda_prompt(arg1, fewshots)
            output[label].append(prompt)
    
    return output