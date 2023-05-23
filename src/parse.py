import re
from loguru import logger
from base import BaseClass

class Parser(BaseClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self, openai_text):
        return NotImplementedError
    
class NormalParser(Parser):
    def __init__(self, strip_chars=[], exclude_strings=["AI language model"]) -> None:
        super().__init__(strip_chars=strip_chars, exclude_strings=exclude_strings)

    def exclude_answer(self, answer):
        for string in self.exclude_strings:
            if string in answer:
                return True
        return False

    def run(self, openai_text):
        if "text" in openai_text:
            text = openai_text["text"]
        elif "message" in openai_text:
            text = openai_text["message"]["content"]
        else:
            logger.error(f"Failed to parser {openai_text}")
            return []

        if len(self.strip_chars) > 0:
            text = text.strip("".join(self.strip_chars))
        
        if len(text) == 0 or self.exclude_answer(text):
            return []

        return [text]
    
class ClassificationParser(Parser):
    def __init__(self, classes) -> None:
        super().__init__(classes=classes)

    def run(self, input_text):
        max_class = None
        max_occurrences = 0
        for class_ in self.classes:
            count = input_text["message"]["content"].lower().count(class_.lower())
            if count > max_occurrences:
                max_class = class_
                max_occurrences = count
        
        if max_occurrences > 0:
            return [max_class]
        else:
            return ["None"]
        
class DualNumberedParser(Parser):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def numbered_sentence(self, string):
        patterna = r"^\d+a+[.:)]\s?"
        patternb = r"^\d+b+[.:)]\s?"
        if bool(re.match(patterna, string)):
            return 0, re.match(patterna, string).end()
        if bool(re.match(patternb, string)):
            return 1, re.match(patternb, string).end()
        return None, 0
    
    def extract_sentence(self, string):
        start_index = 0
        end_index = len(string)
        in_apostrophes = None
        for i, char in enumerate(string):
            if char in ['"', "'"]:
                in_apostrophes = char
                start_index = i + 1
                break
            elif char not in [".", ":", " ", ")"]:
                start_index = i
                break

        for i, char in enumerate(string[::-1]):
            if (char not in [" "] and in_apostrophes is None):
                end_index = len(string) - i
                break
            if char == in_apostrophes:
                end_index = len(string) - i - 1
                break

        if end_index - 1 <= start_index and in_apostrophes: # LLM forgets to close apostrophes
            return string[start_index:]
        
        return string[start_index:end_index]
    
    def run(self, input_text):
        if "message" in input_text:
            text = input_text["message"]["content"]
        else:
            text = input_text["text"]
        finish_reason = input_text["finish_reason"]
        text = text.replace("\n\n", "\n")
        answers = text.split("\n")

        parsed_answers = []
        types = [None, None]
        for answer in answers:
            type_, answer_begin = self.numbered_sentence(answer)
            if type_ is not None:
                extracted = self.extract_sentence(answer[answer_begin:])
                if len(extracted) > 0:
                    types[type_] = extracted
                    if types[0] is not None and types[1] is not None:
                        parsed_answers.append(list(types))
                        types = [None, None]
                else:
                    types = [None, None]
                    logger.warning(f"Empty extractions for sentence '{answer}'.")

        if finish_reason == "length":
            parsed_answers = parsed_answers[:-1]
            logger.warning(f"Finish reason was length, increase the max_tokens parameter for optimal performance.")

        if len(parsed_answers) == 0:
            logger.warning(f"No answers found in text '{text}'.")

        return parsed_answers

class NumberedParser(Parser):
    def __init__(self, only_numbered_sentences=True, exclude_strings=["AI language model"]) -> None:
        super().__init__(only_numbered_sentences=only_numbered_sentences, exclude_strings=exclude_strings)

    def numbered_sentence(self, string):
        pattern = r"^\d+[.:)]"
        return bool(re.match(pattern, string))

    def extract_sentence(self, string):
        start_index = 0
        end_index = len(string)
        in_apostrophes = None
        for i, char in enumerate(string):
            if char in ['"', "'"]:
                in_apostrophes = char
                start_index = i + 1
                break
            elif not char.isnumeric() and char not in [".", ":", " ", ")"]:
                start_index = i
                break

        for i, char in enumerate(string[::-1]):
            if (char not in [" "] and in_apostrophes is None):
                end_index = len(string) - i
                break
            if char == in_apostrophes:
                end_index = len(string) - i - 1
                break

        if end_index - 1 <= start_index and in_apostrophes: # LLM forgets to close apostrophes
            return string[start_index:]
        
        return string[start_index:end_index]
    
    def exclude_answer(self, answer):
        for string in self.exclude_strings:
            if string in answer:
                return True
        return False

    def run(self, input_text):
        if "message" in input_text:
            text = input_text["message"]["content"]
        else:
            text = input_text["text"]
        finish_reason = input_text["finish_reason"]
        text = text.replace("\n\n", "\n")
        answers = text.split("\n")

        parsed_answers = []
        for answer in answers:
            if (not self.only_numbered_sentences or self.numbered_sentence(answer)) and not self.exclude_answer(answer):
                extracted = self.extract_sentence(answer)
                if len(extracted) > 0:
                    parsed_answers.append(extracted)
                else:
                    logger.warning(f"Empty extractions for sentence '{answer}'.")

        if finish_reason == "length":
            parsed_answers = parsed_answers[:-1]
            logger.warning(f"Finish reason was length, increase the max_tokens parameter for optimal performance.")

        if len(parsed_answers) == 0:
            logger.warning(f"No answers found in text '{text}'.")

        return parsed_answers


class LMQLParser(Parser):
    def __init__(self, variables_start=["OUTPUT"]) -> None:
        super().__init__(variables_start=variables_start)

    def run(self, input_text):
        output = []
        for variable in input_text:
            if any([variable.startswith(start) for start in self.variables_start]):
                text = input_text[variable].replace("\xa0", "").replace("\n", "")
                if len(text) > 0:
                    output.append(text)

        return output
