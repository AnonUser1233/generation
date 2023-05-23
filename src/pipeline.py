from inspect import iscoroutinefunction
import os
from dataset import Dataset
import json
from loguru import logger
import shutil
from base import BaseClass

class Pipeline(BaseClass):
    def __init__(self, generator=None, converters=None):
        self.generator = generator
        self.converters = converters
        super().__init__(generator=generator, converters=converters)

    async def run_possible_assync(self, func, *args, **kwargs):
        if iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def select(self, start=0, end=None):
        if end == None:
            end = len(self.converters) + 1
        
        if start > 0:
            return Pipeline(None, converters=self.converters[start - 1:end - 1])
        else:
            return Pipeline(self.generator, converters=self.converters[:end - 1])

    async def run(self, n_sentences, save_folder=None, dataset=None, 
                  save_intermediate_datasets=False, skip=0, return_folder_structure=False, overwrite=False, reload=False, index_run=0):
        folder_structure = {
            "datasets": [], 
            "settings": None
        }
        intermediate_folder = os.path.join(save_folder, "intermediate_datasets")
        final_folder = os.path.join(save_folder, "final_datasets")
        index_run = str(index_run)
        if save_folder is not None:
            logger.debug(f"Saving pipeline settings in {save_folder}/pipeline.json.")
            self.save(os.path.join(save_folder, "pipeline.json"))
            folder_structure["settings"] = os.path.join(save_folder, "pipeline.json")
            os.makedirs(final_folder, exist_ok=True)

            if not overwrite and not reload:
                index_run = str(len(os.listdir(final_folder)))

            if save_intermediate_datasets:
                os.makedirs(intermediate_folder, exist_ok=True)

        if dataset is None:
            dataset = Dataset()

        reload_new = reload and os.path.exists(os.path.join(intermediate_folder, index_run, "dataset_0.csv"))
        if len(self.converters) == 0:
            reload_new = reload and os.path.exists(os.path.join(final_folder, f"final_dataset_{index_run}.csv"))

        if self.generator is not None and skip == 0 and not reload_new:
            logger.info(f"Running generator {self.generator}.")
            dataset = await self.run_possible_assync(self.generator.run, n_sentences=n_sentences)
            
            if save_folder is not None and save_intermediate_datasets and len(self.converters) > 0:
                save_path = os.path.join(intermediate_folder, index_run, "dataset_0.csv")
                folder_structure["datasets"].append(save_path)
                dataset.save(save_path)
        elif self.generator is not None and reload_new:
            logger.info(f"Reloading generator {self.generator}.")
            if len(self.converters) > 0:
                save_path = os.path.join(intermediate_folder, index_run, "dataset_0.csv")
                folder_structure["datasets"].append(save_path)
            else:
                save_path = os.path.join(final_folder, f"final_dataset_{index_run}.csv")
            
            dataset = Dataset.load(save_path)
        else:
            logger.info(f"Skipping generator {self.generator}.")
        
        assert dataset is not None, "Dataset is None."

        for i, converter in enumerate(self.converters):
            if i < skip - 1:
                logger.info(f"Skipping converter {i}: {converter}.")
                continue
            
            save_path = os.path.join(intermediate_folder, index_run, f"dataset_{i + 1}.csv")
            reload_new = reload and os.path.exists(save_path)

            if not reload_new:
                logger.info(f"Running converter {i}: {converter}.")
                dataset = await self.run_possible_assync(converter.run, dataset=dataset)
            else:
                logger.info(f"Reloading converter {i}: {converter}.")
                save_path = os.path.join(intermediate_folder, index_run, f"dataset_{i + 1}.csv")
                dataset = Dataset.load(save_path)

            if save_folder is not None and save_intermediate_datasets and i != len(self.converters) - 1:
                dataset.save(save_path)
                folder_structure["datasets"].append(save_path)
                    

        if save_folder is not None:
            save_path = os.path.join(final_folder, f"final_dataset_{index_run}.csv")
            folder_structure["datasets"].append(save_path)
            if not reload_new:
                logger.info(f"Saving final dataset to {save_path}.")
                dataset.save(save_path)

        if return_folder_structure:   
            return dataset, folder_structure
        return dataset


class MultiPipeline:
    def __init__(self, pipelines) -> None:
        self.pipelines = pipelines
    
    @staticmethod
    def load_datasets(config_file):
        locations = json.load(open(config_file, "r"))
        base_path = os.path.dirname(config_file)
        pipeline_datasets = []
        locations_loaded = []
        for pipeline_loc in locations:
            pipeline = Pipeline.load(os.path.join(base_path, pipeline_loc["settings"]))
            for j, dataset in enumerate(pipeline_loc["datasets"]):
                if dataset not in locations_loaded:
                    pipeline_datasets.append((pipeline.select(end=j + 1), Dataset.load(os.path.join(base_path, dataset))))
                    locations_loaded.append(dataset)

        return pipeline_datasets

    async def run_multiple(self, n_sentences, save_folder, n_runs=1, origin_dataset=None, 
                  already_generated=None, overwrite=False, reload=False):
        if overwrite and os.path.exists(save_folder) and not reload:
            shutil.rmtree(save_folder)
        os.makedirs(save_folder, exist_ok=True)

        for i in range(n_runs):
            await self.run(n_sentences, save_folder, origin_dataset=origin_dataset, 
                  already_generated=already_generated, overwrite=False, reload=reload, index_run=i)

    async def run(self, n_sentences, save_folder, origin_dataset=None, 
                  already_generated=None, overwrite=False, reload=False, index_run=0):
        if already_generated is None:
            already_generated = []

        if overwrite and os.path.exists(save_folder) and not reload:
            shutil.rmtree(save_folder)

        os.makedirs(save_folder, exist_ok=True)

        all_pipelines_locations = [dict() for _ in range(len(self.pipelines))]
        for j, pipeline in enumerate(self.pipelines):
            all_pipelines_locations[j]["datasets"] = []
            skip = 0
            start_dataset_loc = None
            for generated in already_generated:
                possible_skip = 0
                if generated["generator"] == pipeline.generator:
                    possible_skip += 1

                    if len(all_pipelines_locations[j]["datasets"]) == 0:
                        all_pipelines_locations[j]["datasets"].append(generated["dataset"])

                    for i, converter in enumerate(generated["converters"]):
                        if len(pipeline.converters) > i and converter == pipeline.converters[i]:
                            possible_skip += 1

                            if len(all_pipelines_locations[j]["datasets"]) < possible_skip:
                                all_pipelines_locations[j]["datasets"].append(generated["dataset"])
                        else:
                            break
                if possible_skip > skip:
                    skip = possible_skip
                    start_dataset_loc = generated["dataset"]
            
            
            if start_dataset_loc is None:
                logger.info(f"Running pipeline {j} with {n_sentences} sentences.")
                _, save_folder_structure = await pipeline.run(n_sentences, dataset=origin_dataset, 
                                   save_folder=os.path.join(save_folder, f"pipelines/pipeline_{j}"),
                                   save_intermediate_datasets=True, skip=skip, return_folder_structure=True, reload=reload, index_run=index_run)
            else:
                start_dataset = Dataset.load(os.path.join(save_folder, start_dataset_loc))
                logger.info(f"Running pipeline {j} with {n_sentences} sentences, skip={skip} and dataset {start_dataset}.")
                
                _, save_folder_structure = await pipeline.run(n_sentences, dataset=start_dataset, 
                                   save_folder=os.path.join(save_folder, f"pipelines/pipeline_{j}"),
                                   save_intermediate_datasets=True, skip=skip, return_folder_structure=True, reload=reload, index_run=index_run)
                
            all_pipelines_locations[j]["settings"] = save_folder_structure["settings"][len(save_folder) + 1:]

            for dataset_loc in save_folder_structure["datasets"]:
                all_pipelines_locations[j]["datasets"].append(dataset_loc[len(save_folder) + 1:])

            settings_pipeline = {
                "generator": pipeline.generator, 
                "dataset": all_pipelines_locations[j]["datasets"][0], 
                "converters": []
            }

            if skip == 0:
                already_generated.append(settings_pipeline.copy())

            for i, converter in enumerate(pipeline.converters):
                settings_pipeline["converters"].append(converter)
                settings_pipeline["dataset"] = all_pipelines_locations[j]["datasets"][i + 1]
                if skip - 1 <= i:
                    already_generated.append(settings_pipeline.copy())

        save_loc_folder = os.path.join(save_folder, "pipelines.json")
        if not overwrite and os.path.isfile(save_loc_folder):
            with open(save_loc_folder, "r") as f:
                all_pipelines_locations = json.load(f) + all_pipelines_locations
        with open(save_loc_folder, "w") as f:
            json.dump(all_pipelines_locations, f, indent=2, sort_keys=True)

        logger.info(f"Multi pipeline finished.")

        return all_pipelines_locations


class MultiLevelPipeline(MultiPipeline):
    def __init__(self, generators, converter_levels, skip_levels_allowed=True) -> None:
        converter_possibilities = [[]]
        converter_pipelines = []
        for i, level in enumerate(converter_levels):
            new_possibilities = []
            for possibility in converter_possibilities:
                for converter in level:
                    new_possibility = possibility + [converter]
                    new_possibilities.append(new_possibility)
                    if i == len(converter_levels) - 1 and (skip_levels_allowed or len(new_possibility) == len(converter_levels)):
                        converter_pipelines.append(new_possibility)
            
            converter_possibilities += new_possibilities

        if len(converter_pipelines) == 0:
            converter_pipelines = [[]]
            
        pipelines = []
        for generator in generators:
            for converter_pipeline in converter_pipelines:
                pipelines.append(Pipeline(generator, converter_pipeline))

        super().__init__(pipelines)