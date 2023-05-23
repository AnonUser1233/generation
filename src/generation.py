from dataset import Dataset
from converter import DatasetConverter
from parse import NumberedParser
from loguru import logger
from query import OpenAIQuery

class Generator(DatasetConverter):
    def __init__(self, prompts=None, parser=NumberedParser(), querier=OpenAIQuery(), generations_per_prompt=20, **kwargs) -> None:
        super().__init__(prompts=prompts, parser=parser, querier=querier, generations_per_prompt=generations_per_prompt, **kwargs)

    async def run(self, n_sentences, dataset=None, retry_case=0, n_retries=5, *args, **kwargs):
        if dataset is None:
            dataset = Dataset()
        
        n_sentences_per_label = dict()
        for label in self.prompts.get_labels():
            n_sentences_per_label[label] = max(n_sentences - dataset.size(label), 0)

        all_labels = []
        all_prompts = []
        n_sentences_all = []

        for label in self.prompts.get_labels():
            for i in range(0, n_sentences_per_label.get(label, 0), self.generations_per_prompt):
                prompt = self.prompts.get_arbitrary(label)
                all_prompts.append(prompt)
                all_labels.append(label)
                if retry_case == 0:
                    n_sentences_all.append((min(self.generations_per_prompt, n_sentences_per_label.get(label, 0) - i),))
                else:
                    n_sentences_all.append((self.generations_per_prompt,))  # Avoids NumberedParser from failing (generating not enough sentences)

        generations = await self.querier.run(all_prompts, n_sentences_all)

        for label, generation in zip(all_labels, generations):
            answers = self.parser.run(generation)
            dataset.append(answers, [label for _ in range(len(answers))])

        # if too little data was generated, try again
        to_do = 0
        for label in self.prompts.get_labels():
            to_do += max(n_sentences - dataset.size(label), 0)
        if to_do > 0 and retry_case < n_retries:
            logger.info(f"Not enough data was generated, running again. Still {to_do} sentences to generate.")
            await self.run(dataset=dataset, n_sentences=n_sentences, retry_case=retry_case+1, n_retries=n_retries, *args, **kwargs)

        return dataset
