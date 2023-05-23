import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import os
import json
import re
from inspect import signature
import finetune
from utils import get_best_device
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from utils import unnest_dictionary
from datetime import datetime
import mauve

class Evaluator:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.results = dict()

    def set_results(self, results):
        self.results = results

    def save(self, filename):
        if os.path.dirname(filename) != "":
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.isdir(filename):
            file = len(os.listdir(filename))
            filename = os.path.join(filename, f"{file}.json")

        output = {
            "dataset_path": self.dataset.save_path,
            "results": self.results
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2, sort_keys=True)

    def store_result(self, key_name, result, sub_key_name=None):
        logger.debug(f"Storing result {key_name}, {sub_key_name} for {str(self.dataset)} with result {result}")
        if sub_key_name is None or isinstance(result, dict):
            self.results[key_name] = result
        elif key_name in self.results:
            self.results[key_name][sub_key_name] = result
        else:
            self.results[key_name] = {sub_key_name: result}

    def in_result(self, function_name):
        for result in self.results:
            if result.startswith(function_name):
                return True

        return False

    def calculate_all(self, n_runs=1, exclude=[], release_memory=False, save_path=None, results=None, force=[], **kwargs):
        if results is not None:
            self.results = results

        for method in dir(self):
            if self.in_result(method) and results is not None and not method in force:
                continue
                
            if callable(getattr(self, method)) and not method in dir(Evaluator) and not method in exclude:
                sign = signature(getattr(self, method))
                kwargs_method = dict()
                todo = True
                for parameter in sign.parameters:
                    if sign.parameters[parameter].default == sign.parameters[parameter].empty:
                        todo = False
                        break
                    if sign.parameters[parameter].name in kwargs:
                        kwargs_method[sign.parameters[parameter].name] = kwargs[sign.parameters[parameter].name]

                if todo:
                    logger.debug(f"Calculating {method} for {str(self.dataset)}")
                    if not isinstance(self, SupervisedEvaluator) or n_runs == 1:
                        getattr(self, method)(**kwargs_method)
                    else:
                        print("going correct")
                        train_datasets, valid_datasets = self.dataset.split(n_runs)
                        evaluators = [SupervisedEvaluator(train_dataset, self.evaluation_dataset, valid_dataset) for train_dataset, valid_dataset in zip(train_datasets, valid_datasets)]
                        for evaluator in evaluators:
                            getattr(evaluator, method)(**kwargs_method)
                        
                        for key in evaluators[0].results:
                            if isinstance(evaluators[0].results[key], dict):
                                self.results[key] = dict()
                                for sub_key in evaluators[0].results[key]:
                                    if not isinstance(evaluators[0].results[key][sub_key], str):
                                        self.store_result(key, np.mean([evaluator.results[key][sub_key] for evaluator in evaluators]), sub_key)
                                        self.store_result(key, np.std([evaluator.results[key][sub_key] for evaluator in evaluators]) / len(evaluators), str(sub_key) + "_std")
                            else:
                                if not isinstance(evaluators[0].results[key], str):
                                    self.store_result(key, np.mean([evaluator.results[key] for evaluator in evaluators]))
                                    self.store_result(key + "_std", np.std([evaluator.results[key] for evaluator in evaluators]) / len(evaluators))

        if save_path is not None:
            logger.info(f"Saving metrics to {save_path}")
            unnested = unnest_dictionary(self.results)
            if os.path.isfile(save_path):
                with open(save_path, "r") as f:
                    reloading = json.load(f)
                for key in reloading:
                    if key not in unnested:
                        unnested[key] = reloading[key]

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(unnested, f, indent=2, sort_keys=True)
        
        if release_memory:
            self.dataset.release_memory()
        return self.results

class UnsupervisedEvaluator(Evaluator):
    def __init__(self, dataset) -> None:
        super().__init__(dataset)

    def distinctness(self, n=3, store=True, sentences=None):
            # From "A Diversity-Promoting Objective Function for Neural Conversation Models"
            # Don't like this for longer texts since the longer a text, the more words will be repeated even though it might be very diverse
            # Actually, if the distribution over words is uniform, the number of unique n-grams scales about linearly with length -> it does work!
            # emprirically I find something different though, in fact it even drops exponentially with length
            if sentences is None:
                sentences = self.dataset.get_sentences()
            pattern = re.compile(r"[^a-zA-Z ]")
            stripped_text = pattern.sub("", " ".join(sentences))
            ngrams = stripped_text.split(" ")
            ngrams_disinct = np.unique(ngrams)
            if len(ngrams) > 0:
                result = len(ngrams_disinct) / len(ngrams)
                if store:
                    self.store_result("distinctness", result, n)
                return result
            else:
                return None
            
    def average_distinctness(self, n=3, group_size=5000, repetitions=50, store=True, sentences=None, doc=None, spacy=False):
        if sentences is None and not spacy:
            sentences = self.dataset.get_sentences()
            sentences = [sentence.split(" ") for sentence in sentences]
            sentences = [item for sublist in sentences for item in sublist]
        elif doc is not None and spacy:
            sentences = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        elif sentences is None and spacy:
            sentences = [token.lemma_ for token in self.dataset.get_doc() if not token.is_stop and not token.is_punct]

        n_sentences = len(sentences)
        if n_sentences < group_size:
            logger.warning(f"Dataset {self.dataset} has less than {group_size} sentences.")
            result = self.distinctness(n, store=False)
        else:
            total = 0
            for _ in range(repetitions):
                selected_sentences = np.random.choice(sentences, group_size, replace=False)
                total += self.distinctness(n, store=False, sentences=selected_sentences)
            result = total / repetitions
        
        if store:
            self.store_result("average_distinctness", result, n)

        return result
    
    def average_distinctness_spacy(self, n=1, group_size=5000, repetitions=1000, store=True, doc=None):
        result = self.average_distinctness(n, group_size, repetitions, False, doc=doc, spacy=True)
        if store:
            self.store_result("average_distinctness_spacy", result, n)
        
        return result

class SemiSupervisedEvaluator(UnsupervisedEvaluator):
    def __init__(self, dataset, evaluation_dataset, valid_dataset=None) -> None:
        super().__init__(dataset)
        self.evaluation_dataset = evaluation_dataset
        self.valid_dataset = valid_dataset


class SupervisedEvaluator(SemiSupervisedEvaluator):
    def __init__(self, dataset, evaluation_dataset, valid_dataset=None) -> None:
        super().__init__(dataset, evaluation_dataset, valid_dataset=valid_dataset)

    def mauve(self, vectors=None, eval_vectors=None, store=True, max_sentences=4000):
        if vectors is None:
            vectors = self.dataset.get_vectors()
        if eval_vectors is None:
            eval_vectors = self.evaluation_dataset.get_vectors()
        try:
            result = mauve.compute_mauve(vectors[:max_sentences], eval_vectors[:max_sentences])
            if store:
                self.store_result("mauve", float(result.mauve))
        except RuntimeError as e:
            logger.warning(f"RuntimeError {e} mauve for dataset {self.dataset}.")
            return None
        return result.mauve
    
    def finetune(self, batch_size=16, max_length=64, training_steps=2000, lr=1e-5, label_smoothing=0.1, alpha_temporal=0.9, lambda_temporal=0.0, device=None, store_result_name="finetune", 
                       persist_temp_folder=False, num_warmup_steps=600, store=True, reload_best_model=True, only_store_model=False, max_epochs=5):
        if device is None:
            device = torch.device(get_best_device())

        time_folder = datetime.now().strftime("%Y%m%d-%H%M%S%f")

        classes = list(set(self.dataset.get_labels()).union(set(self.evaluation_dataset.get_labels())))
        label_encoder = LabelEncoder()
        label_encoder.fit(classes)
        if self.valid_dataset is None:
            train, test = train_test_split(self.dataset.df, test_size=0.3)
        else:
            train, test = self.dataset.df, self.valid_dataset.df
        model, tokenizer, label_encoder, store_folder_model, store_folder_label_encoder = finetune.run(train, batch_size, max_length, training_steps, lr, label_smoothing, device=device, 
                                                       alpha_temporal=alpha_temporal, lambda_temporal=lambda_temporal, label_encoder=label_encoder, validation_data=test, 
                                                       temp_folder=os.path.join("temp", time_folder), persist_temp_folder=persist_temp_folder, 
                                                       num_warmup_steps=num_warmup_steps, reload_best_model=reload_best_model, max_epochs=max_epochs)
        
        if persist_temp_folder:
            self.store_result("model", store_folder_model, store_result_name)
            self.store_result("label_encoders", store_folder_label_encoder, store_result_name)

        metrics_real = finetune.evaluate_metrics(model, tokenizer, label_encoder, self.evaluation_dataset.df, device, max_length)
        if store or only_store_model:
            self.store_result(store_result_name + "_fake_to_real", metrics_real)

        metrics_fake = finetune.evaluate_metrics(model, tokenizer, label_encoder, test, device, max_length)

        if store or only_store_model:
            self.store_result(store_result_name + "_fake_to_fake", metrics_fake)

        del model, tokenizer
        torch.cuda.empty_cache()

        return {
            "fake": metrics_fake,
            "real": metrics_real
        }
    
    def finetune_temporal(self, batch_size=16, max_length=64, training_steps=2000, lr=1e-5, label_smoothing=0.1, alpha_temporal=0.9, lambda_temporal=1.0, 
                                device=None, persist_temp_folder=False, num_warmup_steps=600, store=True, reload_best_model=True, only_store_model=False, max_epochs=5):
        return self.finetune(batch_size=batch_size, max_length=max_length, training_steps=training_steps, lr=lr, 
                            label_smoothing=label_smoothing, alpha_temporal=alpha_temporal, lambda_temporal=lambda_temporal, 
                            device=device, store_result_name="finetune_temporal", persist_temp_folder=persist_temp_folder, 
                            num_warmup_steps=num_warmup_steps, store=store, reload_best_model=reload_best_model, only_store_model=only_store_model, max_epochs=max_epochs)