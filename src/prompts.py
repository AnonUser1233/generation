from loguru import logger
from inspect import signature
from base import BaseClass
import numpy as np
import types
import numpy as np

class Prompt(BaseClass):
    def __init__(self, prompt, include_as_kwarg=True, **kwargs) -> None:
        self.prompt = prompt
        if include_as_kwarg:
            super().__init__(prompt=prompt, **kwargs)
        else:
            super().__init__(**kwargs)

    def get(self, *args):
        raise NotImplementedError
    
class StandardPrompt(Prompt):
    def __init__(self, prompt) -> None:
        super().__init__(prompt)
    
    def get(self, *args):
        return self.prompt

class LambdaPrompt(Prompt):
    def __init__(self, prompt, parameters=None) -> None:
        super().__init__(prompt, include_as_kwarg=False)
        self.parameters = parameters

    def get(self, *args):
        return self.prompt(*args)
    
    @staticmethod
    def load_from_settings(settings):
        prompt = settings["prompt"]
        parameters = []
        if isinstance(prompt, str):
            parameters = prompt.split("{{")
        elif isinstance(prompt, list):
            for subdict in prompt:
                for key in subdict:
                    parameters += subdict[key].split("{{")

        parameters = [param.split("}}")[0] for param in parameters]
        parameters = [param for param in parameters if param != ""]

        #NOTE: this is a bit cheesy, since the number of parameters cannot be recovered etc. but works.
        def func(*args, prompt=prompt, parameters=parameters):
            if isinstance(prompt, str):
                for i in range(len(parameters)):
                    prompt = prompt.replace("{{" + parameters[i] + "}}", args[i])
                return prompt
            elif isinstance(prompt, list):
                for i in range(len(parameters)):
                    for subdict in prompt:
                        for key in subdict:
                            subdict[key] = subdict[key].replace("{{" + parameters[i] + "}}", str(args[i]))
                return prompt
            else:
                logger.error(f"Prompt {prompt} is not a string or list of dictionaries.")

        return LambdaPrompt(func, parameters=parameters)
    
    def generate_settings(self):
        settings = super().generate_settings()
        if self.parameters is not None:
            string_args = ["{{" + str(arg) + "}}" for arg in self.parameters]
        else:
            parameters = signature(self.prompt).parameters
            args = list(parameters)
            string_args = []
            i = 1
            for arg in args:
                if parameters[arg].default is not parameters[arg].empty:
                    string_args.append(f"{parameters[arg].default}")
                else:
                    string_args.append("{{" + str(arg) + "}}")
                    i += 1
        
        settings["prompt"] = self.get(*string_args)
    
        return settings

class OpenAIPrompt(Prompt):
    def __init__(self, prompt : Prompt) -> None:
        super().__init__(prompt)
    
    def get(self, *args):
        prompt_val = self.prompt.get(*args)
        if isinstance(prompt_val, str):
            prompt_val = [
                {"content": "You are a helpful assistant.", "role": "system"}, 
                {"content": prompt_val, "role": "user"}
            ]
        elif isinstance(prompt_val, dict):
            prompt_val = [
                {"content": "You are a helpful assistant.", "role": "system"}, 
                prompt_val
            ]
        elif isinstance(prompt_val, list) and prompt_val[0]["role"] != "system":
            prompt_val = [
                {"content": "You are a helpful assistant.", "role": "system"}, 
                *prompt_val
            ]

        return prompt_val

    
class OpenAIMultiPrompt(Prompt):
    def __Init__(self, prompt : Prompt) -> None:
        super().__init__(prompt=prompt)

    def get(self, previous_prompt, answer):
        new_prompt = self.prompt.get()
        if isinstance(new_prompt, str):
            new_prompt = {"content": new_prompt, "role": "user"}
        if isinstance(new_prompt, dict):
            new_prompt = [new_prompt]
        return previous_prompt + [answer["message"]] + new_prompt


class Prompts(BaseClass):
    def __init__(self, prompts, **kwargs):
        super().__init__(prompts=prompts, **kwargs)

    def get_arbitrary(self):
        raise NotImplementedError
    
    def get(self, label=None):
        raise NotImplementedError
    
    def get_labels(self):
        raise NotImplementedError
    
    def get_at(self, index, label=None):
        raise NotImplementedError


class LabeledPrompts(Prompts):
    def __init__(self, prompts):
        for label in prompts:
            for i, prompt in enumerate(prompts[label]):
                if isinstance(prompt, types.FunctionType):
                    prompts[label][i] = LambdaPrompt(prompt)
                elif not isinstance(prompt, Prompt):
                    prompts[label][i] = StandardPrompt(prompt)
        
        super().__init__(prompts=prompts)
        self.prompts = prompts

    def get_arbitrary(self, label=None):
        if label is None:
            label = np.random.choice(list(self.prompts.keys()))
        return np.random.choice(self.prompts[label])
    
    def get(self, label):
        return self.prompts[label]
    
    def get_labels(self):
        return list(self.prompts.keys())

    def get_at(self, index, label):
        return self.prompts[label][index % len(self.prompts[label])]
