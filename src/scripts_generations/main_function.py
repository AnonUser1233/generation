import os, sys
sys.path.append(os.path.abspath(os.getcwd()))
from generation import Generator
from pipeline import MultiLevelPipeline
import pandas as pd
import numpy as np
from prompt_builder import build_fewshot, merge_prompts
from prompts import LabeledPrompts, LambdaPrompt, OpenAIPrompt
from query import OpenAIQuery
from parse import NormalParser, NumberedParser
from loguru import logger
import asyncio
from dotenv import load_dotenv
from prompts import StandardPrompt
from rewrite import Rewrite
from converter import MultiConverter
load_dotenv()


def caller(labels, dataset_name, basic_prompt, basic_prompt_chat, basic_prompt_lambda, basic_prompt_lambda_chat, 
           few_shot_prompt, few_shot_prompt_chat, examples, parent_dir, args):
    prompts = {
        label: [StandardPrompt(basic_prompt(label))] for label in labels
    }

    prompts_chat = {
        label: [OpenAIPrompt(StandardPrompt(basic_prompt_chat(label)))] for label in labels
    }

    prompts_lambda = {
        label: [LambdaPrompt(lambda n, label=label: basic_prompt_lambda(n, label))] for label in labels
    }   

    prompts_lambda_chat = {
        label: [OpenAIPrompt(LambdaPrompt(lambda n, label=label: basic_prompt_lambda_chat(n, label)))] for label in labels
    }


    few_shot_prompts_per_label = {
        label: [lambda arg1, examples, label=label: few_shot_prompt(label, examples)] for label in labels
    }

    few_shot_prompts_per_label_chat = {
        label: [lambda arg1, examples, label=label: few_shot_prompt_chat(label, examples)] for label in labels
    }

    logger.add(f"../../logs/call/{dataset_name}/{args.model}.log", rotation="1 day", level="DEBUG", backtrace=True, diagnose=True)

    few_shot = None
    for class_ in few_shot_prompts_per_label:
        if not args.chat:
            extra = build_fewshot(500, few_shot_prompts_per_label[class_][0], [class_], examples[class_], 3)
        else:
            extra = build_fewshot(500, few_shot_prompts_per_label_chat[class_][0], [class_], examples[class_], 3)
        if few_shot is None:
            few_shot = extra
        else:
            few_shot = merge_prompts(few_shot, extra)



    temperatures = np.arange(args.min_temp, args.max_temp, args.step)
    generators = []

    if args.chat:
        prompts = prompts_chat
        prompts_lambda = prompts_lambda_chat
        for label in few_shot:
            few_shot[label] = [OpenAIPrompt(LambdaPrompt(prompt)) for prompt in few_shot[label]]

    prompts = LabeledPrompts(prompts)
    few_shot_prompts = LabeledPrompts(few_shot)

    for temp in temperatures:
        rlhf_model = f"text-{args.model}-001"
        if args.chat:
            rlhf_model = args.model

        if not args.chat:
            generators.append(
                Generator(
                    prompts=prompts,
                    querier=OpenAIQuery(model=f"{args.model}", max_tokens=128, temperature=temp, stop=["\n"]),
                    parser=NormalParser(strip_chars=['"', "'", " "]),
                    generations_per_prompt=1
                )
            )

        generators.append(
            Generator(
                prompts=prompts,
                querier=OpenAIQuery(model=rlhf_model, max_tokens=128, temperature=temp, stop=["\n"]),
                parser=NormalParser(strip_chars=['"', "'", " "]),
                generations_per_prompt=1
            )
        )

        generators.append(
            Generator(
                prompts=few_shot_prompts,
                querier=OpenAIQuery(model=rlhf_model, max_tokens=128, temperature=temp, stop=["\n"]),
                parser=NormalParser(strip_chars=['"', "'", " "]),
                generations_per_prompt=1
            )
        )

    if args.model == "davinci":
        for temp in temperatures:
            generators.append(
                Generator(
                    prompts=prompts,
                    querier=OpenAIQuery(model="text-davinci-003", max_tokens=128, temperature=temp, stop=["\n"]),
                    parser=NormalParser(strip_chars=['"', "'", " "]),
                    generations_per_prompt=1
                )
            )
        generators.append(
            Generator(
                prompts=few_shot_prompts,
                querier=OpenAIQuery(model="text-davinci-003", max_tokens=128, temperature=1.0, stop=["\n"]),
                parser=NormalParser(strip_chars=['"', "'", " "]),
                generations_per_prompt=1
            )
        )

        generators.append(
            MultiConverter(
                    [Generator(
                        prompts=prompts,
                        querier=OpenAIQuery(model="text-davinci-003", max_tokens=2, temperature=1.3, tpm=5000, stop=["\n"]),
                        parser=NormalParser(strip_chars=['"', "'", " "]),
                        generations_per_prompt=1
                    ),
                    Rewrite(
                        prompts_per_level=[LabeledPrompts(prompts_lambda)],
                        querier=OpenAIQuery(model="text-davinci-003", max_tokens=128, temperature=1.0, stop=["\n"]),
                        parser=NormalParser(strip_chars=['"', "'"]), n_input_sentences=1,
                        generations_per_prompt=1, prepend=True
                    )
                    ]
            ) 
        )

        generators.append(
            MultiConverter(
                    [Generator(
                        prompts=prompts,
                        querier=OpenAIQuery(model="text-davinci-001", max_tokens=2, temperature=1.3, tpm=5000, stop=["\n"]),
                        parser=NormalParser(strip_chars=['"', "'", " "]),
                        generations_per_prompt=1
                    ),
                    Rewrite(
                        prompts_per_level=[LabeledPrompts(prompts_lambda)],
                        querier=OpenAIQuery(model="text-davinci-001", max_tokens=128, temperature=1.0, stop=["\n"]),
                        parser=NormalParser(strip_chars=['"', "'"]), n_input_sentences=1,
                        generations_per_prompt=1, prepend=True
                    )
                    ]
            ) 
        )

        generators.append(
                Generator(
                    prompts=prompts,
                    querier=OpenAIQuery(model=f"{args.model}", max_tokens=128, temperature=1.0, top_p=0.9, stop=["\n"]),
                    parser=NormalParser(strip_chars=['"', "'", " "]),
                    generations_per_prompt=1
                )
            )

        generators.append(
            Generator(
                prompts=prompts,
                querier=OpenAIQuery(model="text-davinci-001", max_tokens=128, temperature=1.0, top_p=0.9, stop=["\n"]),
                parser=NormalParser(strip_chars=['"', "'", " "]),
                generations_per_prompt=1
            )
        )

        generators.append(
            Generator(
                prompts=prompts,
                querier=OpenAIQuery(model="text-davinci-003", max_tokens=128, temperature=1.0, top_p=0.9, stop=["\n"]),
                parser=NormalParser(strip_chars=['"', "'", " "]),
                generations_per_prompt=1
            )
        )


    pipeline = MultiLevelPipeline(
        generators=generators, converter_levels=[]
    )

    with logger.catch():
        asyncr = pipeline.run_multiple(args.size, save_folder=os.path.join(parent_dir, args.model), reload=True, n_runs=1)

        res = asyncio.get_event_loop().run_until_complete(asyncr)   

