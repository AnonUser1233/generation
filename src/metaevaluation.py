from evaluation import UnsupervisedEvaluator, SupervisedEvaluator, SemiSupervisedEvaluator
from dataset import Dataset
from pipeline import MultiPipeline
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import run_parallel
from datetime import datetime
import numpy as np
import json
from loguru import logger
import shutil
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from scipy.spatial import cKDTree
from utils import get_best_device
from transformers import DistilBertForSequenceClassification
import finetune
import pickle
from utils import unnest_dictionary
sns.set_theme(style="whitegrid")

class MetaEvaluator:
    def __init__(self, pipeline_datasets, evaluation_dataset=None, y="linear_fake_to_real.accuracy") -> None:
        self.pipeline_datasets = pipeline_datasets
        self.evaluation_dataset = evaluation_dataset
        self.y = y

    def deduplicate_vectors(self, array, dataset_indices, labels_indices, epsilon=1e-8):
        kdtree = cKDTree(array)
        dedup_mask = np.ones(array.shape[0], dtype=bool)
        new_dataset_indices = []
        
        for i, vector in enumerate(array):
            datasets_index = [dataset_indices[i]]
            if dedup_mask[i]:
                neighbors = kdtree.query_ball_point(vector, r=epsilon)
                for neighbor in neighbors[1:]:
                    dedup_mask[neighbor] = False
                    datasets_index.append(dataset_indices[neighbor])
            new_dataset_indices.append(datasets_index)

        unique_vectors = array[dedup_mask]
        unique_labels = labels_indices[dedup_mask]
        real_new_dataset_indices = [index for i, index in enumerate(new_dataset_indices) if dedup_mask[i]]
        return unique_vectors, real_new_dataset_indices, unique_labels

    def calculate_global_evaluation_metrics(self, dict_real_metrics, list_dict_metrics, n_cores, reload_=True):
        # real to fake metrics
        key_names = list(dict_real_metrics.keys())
        for model_folders in key_names:
            if model_folders.startswith("model."):
                metric_name = model_folders.split(".")[1]
                if (metric_name + "_real_to_fake" in key_names and reload_) or not os.path.exists(dict_real_metrics[model_folders]):
                    continue
                tokenizer = finetune.get_tokenizer()
                model = DistilBertForSequenceClassification.from_pretrained(dict_real_metrics[model_folders])
                label_encoder = pickle.load(open(dict_real_metrics["label_encoders." + metric_name], "rb"))
                device = get_best_device()
                model = model.to(device)
                eval_funcs = []
                for i, (_, dataset) in enumerate(self.pipeline_datasets):
                    if metric_name + "_real_to_fake" not in list_dict_metrics[i]:
                        eval_funcs.append(lambda dataset=dataset: finetune.evaluate_metrics(model, tokenizer, label_encoder, dataset.df, device))

                results = run_parallel(eval_funcs, n_cores)
                # shutil.rmtree(dict_real_metrics[model_folders])

                for i in range(len(results)):
                    list_dict_metrics[i][metric_name + "_real_to_fake"] = results[i]
                    list_dict_metrics[i] = unnest_dictionary(list_dict_metrics[i])

                key_names = list(dict_real_metrics.keys())
                for metric in key_names:
                    if metric.startswith(metric_name + "_fake_to_fake"):
                        dict_real_metrics[metric.replace("_fake_to_fake", "_real_to_fake")] = dict_real_metrics[metric]

        return dict_real_metrics, list_dict_metrics
    
    def calculate_evaluation_metrics(self, eval1, eval2, folder, first_phase_n_cores, second_phase_n_cores, exclude, dataset_range, n_runs=5, include_eval=True, 
                                     reload_=True, force=[], **kwargs):
        list_dict_metr, dict_eval_metr = self.load_all_results(folder)
        if not reload_:
            dict_eval_metr = None
            list_dict_metr = [None for _ in range(len(self.pipeline_datasets))]

        real_evaluator = SupervisedEvaluator(eval1, eval2)
        all_funcs = []
        if include_eval:
            eval_metrics_path = os.path.join(folder, "results", f"dict_real_metrics.json")
            models_unnecessary_functions = dir(real_evaluator)
            models_unnecessary_functions = [func for func in models_unnecessary_functions if not func.startswith("finetune") or func in exclude]
            all_funcs.append(lambda save_path=eval_metrics_path: real_evaluator.calculate_all(exclude=models_unnecessary_functions, n_runs=1, persist_temp_folder=True, save_path=save_path, 
                                                                                              results=dict_eval_metr, **kwargs))
            all_funcs.append(lambda save_path=eval_metrics_path: real_evaluator.calculate_all(exclude=exclude, n_runs=n_runs, save_path=save_path, results=dict_eval_metr,
                                                                                             force=force, 
                                                                                                **kwargs))
            

        for i, (_, dataset) in enumerate(self.pipeline_datasets[dataset_range[0]:dataset_range[1]]):
            list_metrics_path = os.path.join(folder, "results", f"list_dict_metrics_{i + dataset_range[0]}-{i+dataset_range[0]+1}.json")
            evaluator = None
            if self.evaluation_dataset is None:
                evaluator = UnsupervisedEvaluator(dataset)
            elif len(self.evaluation_dataset.get_labels()) <= 1:
                evaluator = SemiSupervisedEvaluator(dataset, self.evaluation_dataset)
            else:
                evaluator = SupervisedEvaluator(dataset, self.evaluation_dataset)

            all_funcs.append(lambda evaluator=evaluator, save_path=list_metrics_path, i=i: evaluator.calculate_all(exclude=exclude, n_runs=n_runs, save_path=save_path, 
                                                                                                                results=list_dict_metr[i + dataset_range[0]], **kwargs))

        if second_phase_n_cores > 1 and len(all_funcs) > 1:
            metrics = run_parallel(all_funcs, n_cores=second_phase_n_cores)
        else:
            metrics = [func() for func in all_funcs]
    
    def generate_eval_file(self, list_dict_metrics, dict_real_metrics, folder, title, template_folder):
        all_keys_in_metrics = set()
        for keys in list_dict_metrics:
            all_keys_in_metrics = all_keys_in_metrics.union(set(keys.keys()))
        
        for key in all_keys_in_metrics:
            for dict_metrics in list_dict_metrics:
                dict_metrics[key] = dict_metrics.get(key, 0)
        

        y = [list_dict_metrics[i][self.y] for i in range(len(list_dict_metrics))]
        y_real = dict_real_metrics[self.y]

        metrics = []
        metric_names = set(dict_real_metrics)
        for list_metric in list_dict_metrics:
            metric_names = metric_names.union(set(list_metric))

        metric_names = list(metric_names)
        metric_names.sort()
        metrics_good = []

        generators_list = []
        generators_jinja = []
        average_score_per_generator = dict()

        converters_list = []
        converters_jinja = []
        average_score_per_converter = dict()

        logger.info("Generating generator and converter jinja templates.")
        for i, (pipeline, _) in enumerate(self.pipeline_datasets):
            if pipeline.generator not in generators_list:
                kwargs = pipeline.generator.generate_settings()
                if kwargs.get("prompts") is None:
                    kwargs["prompts"] = []

                kwargs_no_prompt = kwargs.copy()
                del kwargs_no_prompt["prompts"]
                del kwargs_no_prompt["class"]

                generators_jinja.append({
                    "id": len(generators_list), 
                    "class": kwargs["class"],
                    "kwargs": kwargs_no_prompt, 
                    "prompts": kwargs["prompts"],
                })
                generators_list.append(pipeline.generator)
                
                average_score_per_generator[len(generators_list) - 1] = [y[i], 1]
            else:
                index = generators_list.index(pipeline.generator)
                average_score_per_generator[index][0] += y[i]
                average_score_per_generator[index][1] += 1
                

            for converter in pipeline.converters:
                if converter not in converters_list:
                    kwargs = converter.generate_settings()
                    if kwargs.get("prompts_per_level") is None:
                        kwargs["prompts_per_level"] = []

                    kwargs_no_prompt = kwargs.copy()
                    del kwargs_no_prompt["prompts_per_level"]
                    del kwargs_no_prompt["class"]
                    converters_jinja.append({
                        "id": len(converters_list), 
                        "class": kwargs["class"], 
                        "prompts": kwargs["prompts_per_level"],
                        "kwargs": kwargs_no_prompt
                    })
                    converters_list.append(converter)
                    average_score_per_converter[len(converters_list) - 1] = [y[i], 1]
                else:
                    index = converters_list.index(converter)
                    average_score_per_converter[index][0] += y[i]
                    average_score_per_converter[index][1] += 1

        sorted_indices = np.argsort(y)[::-1]

        all_locations = [
            self.pipeline_datasets[i][1].save_path for i in range(len(sorted_indices))
        ]
        existing_locations = [loc for loc in all_locations if loc is not None]
        try:
            common_prefix = next((existing_locations[0][:i] for i,(p,*r) in enumerate(zip(*existing_locations)) 
                                        if any(p!=c for c in r)),min(existing_locations,key=len))
            common_prefix = common_prefix[:common_prefix.rfind("/") + 1]
        except ValueError:
            common_prefix = ""

        all_locations = [
            loc if loc is not None else "-" for loc in all_locations
        ]

        logger.debug(f"Removing common prefix {common_prefix} from locations")

        datasets = []

        datasets.append({
            "id": "real",
            "y": dict_real_metrics[self.y],
            "size": self.evaluation_dataset.size(), 
            "generator": "-",
            "converters": "-", 
            "location": "-", 
            "metrics": {
                key: dict_real_metrics.get(key, 0) for key in metrics_good
            }
        })

        for i, index in enumerate(sorted_indices):
            logger.debug(f"Generating plots for dataset {str(self.pipeline_datasets[index][1])}")
            metrics_index = {
                key: list_dict_metrics[index].get(key, 0) for key in metrics_good
            }
            datasets.append({
                "id": i,
                "y": list_dict_metrics[index][self.y],
                "size": self.pipeline_datasets[index][1].size(), 
                "generator": generators_list.index(self.pipeline_datasets[index][0].generator),
                "converters": [converters_list.index(converter) for converter in self.pipeline_datasets[index][0].converters], 
                "location": all_locations[index][len(common_prefix):], 
                "metrics": metrics_index
            })

        info = {
            "title": title, 
            "date": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
            "folder": common_prefix, 
            "n_datasets": len(self.pipeline_datasets),
        }

        real = {
            "y": dict_real_metrics[self.y],
            "size": self.evaluation_dataset.size()
        }

        env = Environment(
            loader=FileSystemLoader(template_folder),
        )

        
        logger.info("Generating final evaluation html file")
        template = env.get_template("evaluation_template.html")
        rendered = template.render(info=info, datasets=datasets, generators=generators_jinja, converters=converters_jinja, 
                                   metrics=metrics, real=real, metric_names=metrics_good)
        with open(os.path.join(folder, "evaluation.html"), "w", encoding="utf-8") as f:
            f.write(rendered)

    def load_all_results(self, folder):
        results_folder = os.path.join(folder, "results")
        real_metrics_path = os.path.join(results_folder, "dict_real_metrics.json")
        dict_real_metrics = None
        if os.path.exists(real_metrics_path):
            with open(real_metrics_path, "r") as f:
                dict_real_metrics = json.load(f)


        list_dict_metrics = [None for _ in range(len(self.pipeline_datasets))]
        all_result_path = f"list_dict_metrics_0-{len(list_dict_metrics) - 1}.json"
        if os.path.exists(os.path.join(results_folder, all_result_path)):
            with open(os.path.join(results_folder, all_result_path), "r") as f:
                list_dict_metrics = json.load(f)
                if isinstance(list_dict_metrics, dict):
                    list_dict_metrics = [list_dict_metrics]

        for file in os.listdir(results_folder):
            if file.startswith("list_dict_metrics_") and file != all_result_path:
                with open(os.path.join(results_folder, file), "r") as f:
                    index = int(file[len("list_dict_metrics_"):file.find("-")])
                    metrics = json.load(f)
                    if isinstance(metrics, dict):
                        metrics = [metrics]
                    for i, metrics in enumerate(metrics):
                        list_dict_metrics[i + index] = metrics

        return list_dict_metrics, dict_real_metrics
    
    def store_all_results(self, folder, list_dict_metrics, dict_real_metrics):
        results_folder = os.path.join(folder, "results")
        shutil.rmtree(results_folder)
        os.makedirs(results_folder, exist_ok=True)
        with open(os.path.join(results_folder, "dict_real_metrics.json"), "w") as f:
            json.dump(dict_real_metrics, f, indent=2, sort_keys=True)
        
        with open(os.path.join(results_folder, f"list_dict_metrics_0-{len(list_dict_metrics) - 1}.json"), "w") as f:
            json.dump(list_dict_metrics, f, indent=2, sort_keys=True)
    
    def check_results_present(self, folder, dataset_range=None, include_eval=True):
        list_dict_metrics, dict_real_metrics = self.load_all_results(folder)
        if dict_real_metrics is None and include_eval:
            return False
        
        if dataset_range is None:
            dataset_range = (0, len(self.pipeline_datasets))
        
        for i in range(*dataset_range):
            if i < len(list_dict_metrics) and list_dict_metrics[i] is None:
                return False
        
        return True

    def run(self, template_folder, folder="./metaevaluation/", n_runs=5, first_phase_cores=1, second_phase_cores=1, 
            exclude=[], force=[], title="Emotions", dataset_range=None, include_eval=True, reload_=True, real_metrics_loc=None, **kwargs):
        if dataset_range is None:
            dataset_range = (0, len(self.pipeline_datasets))
        
        for i, (pipeline, _) in enumerate(self.pipeline_datasets):
            pipeline.save(os.path.join(folder, "pipelines", f"pipeline_{i}.json"))

        os.makedirs(os.path.join(folder, "results"), exist_ok=True)

        if real_metrics_loc is not None:
            with open(real_metrics_loc, "r") as f:
                dict_real_metrics = json.load(f)

            with open(os.path.join(folder, "results", "dict_real_metrics.json"), "w") as f:
                json.dump(dict_real_metrics, f, indent=2, sort_keys=True)

        sentences, labels = self.evaluation_dataset.get_all()
        size = len(sentences) // 2
        eval1 = Dataset(sentences[:size], labels[:size])
        eval2 = Dataset(sentences[size:], labels[size:])
        logger.info("Starting calculation evaluators")
        self.calculate_evaluation_metrics(eval1, eval2, folder, first_phase_cores, second_phase_cores, exclude, dataset_range, include_eval=include_eval, reload_=reload_, force=force, n_runs=n_runs, **kwargs)
        logger.success("Done calculating metrics.")

        list_dict_metrics, dict_real_metrics = self.load_all_results(folder)

        if dict_real_metrics is not None and all([metrics is not None for metrics in list_dict_metrics]):
            logger.info("Calculation final metrics...")
            dict_real_metrics, list_dict_metrics = self.calculate_global_evaluation_metrics(dict_real_metrics, list_dict_metrics, 
                                                                                            second_phase_cores, reload_=reload_)
            self.store_all_results(folder, list_dict_metrics, dict_real_metrics)
            logger.success("Done calculating final metrics.")
            logger.info("Generating evaluation file")
            self.generate_eval_file(list_dict_metrics, dict_real_metrics, folder, title, template_folder)
            logger.success("Done generating evaluation file.")
            logger.info("Removing temporary files")
            if os.path.exists(os.path.join(folder, "temp")):
                shutil.rmtree(os.path.join(folder, "temp"))

    @staticmethod
    def load_from_multipipeline(file, evaluation_dataset, y=None):
        multipipeline = MultiPipeline.load_datasets(file)
        if y is None:
            return MetaEvaluator(multipipeline, evaluation_dataset)
        else:
            return MetaEvaluator(multipipeline, evaluation_dataset, y)
