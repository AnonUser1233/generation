from base import BaseClass
from inspect import iscoroutinefunction


class DatasetConverter(BaseClass):
    def __init__(self, prompts=None, parser=None, **kwargs) -> None:
        self.prompts = prompts
        self.parser = parser
        super().__init__(parser=parser, **kwargs, prompts=prompts)

    def run(self, dataset, *args, **kwargs):
        return


class MultiConverter(DatasetConverter):
    def __init__(self, converters=None, **kwargs) -> None:
        self.converters = converters
        super().__init__(**kwargs, converters=converters)

    async def run_possible_assync(self, func, *args, **kwargs):
        if iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    async def run(self, dataset=None, n_sentences=None):
        for i, converter in enumerate(self.converters):
            if i == 0 and n_sentences is not None:
                dataset = await self.run_possible_assync(converter.run, dataset=dataset, n_sentences=n_sentences)
            else:
                dataset = await self.run_possible_assync(converter.run, dataset=dataset)
        return dataset
