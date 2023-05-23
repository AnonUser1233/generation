from loguru import logger
from base import BaseClass
import numpy as np
import os

import aiohttp

import json
import time
import asyncio
import numpy as np
import asyncio


class Query(BaseClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def run_string_prompts(self, string_prompts):
        raise NotImplementedError
    
    async def run(self, prompts, args_for_prompts):
        for i in range(len(args_for_prompts)):
            if not isinstance(args_for_prompts[i], (tuple, list)):
                args_for_prompts[i] = (args_for_prompts[i],)
        
        string_prompts = [prompt.get(*args) for prompt, args in zip(prompts, args_for_prompts)]
        return await self.run_string_prompts(string_prompts)

class OpenAIQuery(Query):
    def __init__(self, model="gpt-3.5-turbo", tpm=30000, timeout=100, temperature=1.2, max_tokens=1000, error_stop=10 ** 8, **kwargs) -> None:
        super().__init__(model=model, tpm=tpm, timeout=timeout, temperature=temperature, max_tokens=max_tokens, error_stop=error_stop, **kwargs)
    
    async def run_string_prompts(self, string_prompts):
        kwarg = self.kwargs.copy()
        del kwarg["tpm"]
        del kwarg["timeout"]
        del kwarg["error_stop"]
        openai_queries = []
        for prompt in string_prompts:
            if isinstance(prompt, str):
                openai_queries.append({"prompt": prompt, **kwarg})
            else:
                openai_queries.append({"messages": prompt, **kwarg})

        return await self.get_completions(openai_queries)

    async def get_completion_async(self, arguments, session):
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        try:
            url = "https://api.openai.com/v1/chat/completions"
            if "prompt" in arguments:
                url = "https://api.openai.com/v1/completions"
            async with session.post(
                url, 
                headers={
                    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json=arguments
            ) as response:
                resp = await response.read()
                return resp
        except Exception as e:
            logger.warning(f"Error occurred while posting to openai API: {e}. Posted: {arguments}")
            return None
        
    async def get_completions_async(self, list_arguments):
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            ret = await asyncio.gather(*[self.get_completion_async(argument, session) for argument in list_arguments])
            
        return ret

    async def get_completions(self, list_arguments):
        succeeded_requests = [False for _ in range(len(list_arguments))]
        outputs = [None for _ in range(len(list_arguments))]
        generated_tokens = []
        n_errors = 0
        n_parse_errors = 0
        n_new_errors = 0
        while not all(succeeded_requests) and n_errors < self.error_stop and n_parse_errors < self.error_stop:
            start_time = time.time()
            generated_tokens_last_min = sum([usage[1] for usage in generated_tokens if start_time - usage[0] < 60])
            async_requests = (self.tpm - min(generated_tokens_last_min, self.tpm)) // self.max_tokens
            if async_requests == 0:
                time.sleep(0.2)
                continue

            indices = np.where(np.logical_not(succeeded_requests))[0][:async_requests]
            arguments_async = [list_arguments[index] for index in indices]
            logger.debug(f"Running {len(arguments_async)} requests to openai API. tokens last minute: {generated_tokens_last_min}. percentage done: {np.count_nonzero(succeeded_requests) / len(succeeded_requests) * 100:.2f}%")
            if asyncio.get_event_loop().is_running():
                ret = await self.get_completions_async(arguments_async)
            else:
                ret = await asyncio.run(self.get_completions_async(arguments_async))

            for results, index in zip(ret, indices):
                if results is not None:
                    try:
                        outputs[index] = json.loads(results)
                        if "error" not in outputs[index]:
                            succeeded_requests[index] = True
                            generated_tokens.append((start_time, outputs[index]["usage"]["total_tokens"]))
                            outputs[index] = outputs[index]["choices"][0]
                        else: 
                            logger.warning(f"OpenAI API returned an error: {outputs[index]} \n On parameters {list_arguments[index]}")
                            n_errors += 1
                            n_new_errors += 1
                    except Exception:
                        logger.warning(f"OpenAI API returned invalid json: {results} \n On parameters {list_arguments[index]}")
                        n_parse_errors += 1
                else:
                    n_errors += 1
                    n_new_errors += 1

            if n_new_errors >= 20:
                time.sleep(10)
                n_new_errors = 0
                    
        if n_errors >= self.error_stop or n_parse_errors >= self.error_stop:
            raise ValueError("OpenAI API returned too many errors. Stopping requests.")

        return outputs
