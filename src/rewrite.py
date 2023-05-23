import numpy as np
from converter import DatasetConverter
from parse import NumberedParser
from loguru import logger
from query import OpenAIQuery
from dataset import Dataset
from base import BaseClass
from copy import deepcopy

class Rewrite(DatasetConverter):
    def __init__(self, prompts_per_level, parser=NumberedParser(), querier=OpenAIQuery(), 
                 shuffle_data=True, probability_rewrite=1.0, n_input_sentences=20, extend=False, prepend=False, **kwargs) -> None:
        if isinstance(prompts_per_level, BaseClass):
            prompts_per_level = [prompts_per_level]

        super().__init__(prompts_per_level=prompts_per_level, parser=parser, shuffle_data=shuffle_data,
                          probability_rewrite=probability_rewrite, n_input_sentences=n_input_sentences,
                          querier=querier, extend=extend, prepend=prepend, **kwargs)

    def prepare_phase_0(self, dataset):
        if self.shuffle_data:
            dataset.shuffle()

        input_sentences = {
            label: [] for label in dataset.get_labels()
        }
        chosen_sentences = np.array([False for _ in range(dataset.size())])

        for label in dataset.get_labels():
            all_sentences, indices = dataset.get_label_sentences(label)
            rewrite_indices = np.random.uniform(0, 1, size=len(all_sentences)) < self.probability_rewrite
            all_sentences, indices = all_sentences[rewrite_indices], indices[rewrite_indices]
            chosen_sentences[indices] = True
            
            for i in range(0, len(all_sentences), self.n_input_sentences):
                sentences = ""
                length = 0
                for j, sentence in enumerate(all_sentences[i:i + self.n_input_sentences]):
                    sentences += f"{j + 1}. {sentence}\n"
                    length += 1

                if self.n_input_sentences == 1:
                    sentences = all_sentences[i]
                input_sentences[label].append(sentences)
        
        return input_sentences, chosen_sentences


    async def run_level(self, level, previous_answers, previous_prompts=None):
        logger.info(f"Running level {level}")
        new_prompts = dict()
        for label in previous_answers:
            new_prompts[label] = []
            for i, answer in enumerate(previous_answers[label]):
                random_prompt = self.prompts_per_level[level].get_arbitrary(label)
                if previous_prompts is None:
                    new_prompts[label].append(random_prompt.get(answer))
                else:
                    new_prompts[label].append(random_prompt.get(previous_prompts[label][i], answer))
        
        flatted_prompts = []
        for label in new_prompts:
            flatted_prompts.extend(new_prompts[label])

        output = await self.querier.run_string_prompts(flatted_prompts)

        answers = dict()
        i = 0
        for label in new_prompts:
            answers[label] = []
            for _ in range(len(new_prompts[label])):
                answers[label].append(output[i])
                i += 1

        return answers, new_prompts
        
    async def run(self, dataset):
        original_answers, chosen_sentences = self.prepare_phase_0(dataset)
        answers = deepcopy(original_answers)
        new_prompts = None
        for level in range(len(self.prompts_per_level)):
            answers, new_prompts = await self.run_level(level, answers, previous_prompts=new_prompts)
        
        new_sentences = []
        new_labels = []
        if self.extend:
            new_sentences, new_labels = dataset.get_all()
            new_sentences = list(new_sentences)
            new_labels = list(new_labels)

        for i in range(len(chosen_sentences)):
            if not chosen_sentences[i]:
                sentence, label = dataset.get(i)
                new_sentences.append(sentence)
                new_labels.append(label)

        for label in answers:
            for i, answer in enumerate(answers[label]):
                parsed = self.parser.run(answer)
                for parsed_answer in parsed:
                    if len(parsed_answer) > 0:
                        if not self.prepend:
                            new_sentences.append(parsed_answer)
                        else:
                            new_sentences.append(f"{original_answers[label][i]}{parsed_answer}")
                        new_labels.append(label)

        return Dataset(new_sentences, new_labels)