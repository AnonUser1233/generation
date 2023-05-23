import os, sys
sys.path.append(os.path.abspath(os.getcwd()))
import argparse
import subprocess
from loguru import logger
import json
import time
from utils import set_seed
set_seed()

def get_n_datasets(file):
    # get the number of datasets, without using MetaEvaluator in order to not make imports that allocate gpu resources
    pipelines = json.load(open(file))
    datasets = set()
    for pipeline in pipelines:
        datasets = datasets.union(set(pipeline["datasets"]))

    return len(datasets)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Meta Evaluation",
        description="Meta Evaluation of Generations",
    )

    parser.add_argument("pipeline_location", type=str, help="Location of the pipeline to evaluate")
    parser.add_argument("--cores", type=int, default=2, help="Number of processes to use")
    parser.add_argument("--title", type=str, default="SST", help="Title of the meta evaluation")
    parser.add_argument("--parent_dir1", type=str, default="../data/generations", help="Parent directory of the pipeline to evaluate")
    parser.add_argument("--parent_dir2", type=str, default="SST", help="Parent directory of the pipeline to evaluate")
    parser.add_argument("--evaluation_loc", type=str, default="eval_data", help="Location of the evaluation dataset")
    parser.add_argument("--no-real", action='store_true', help="Whether to include real evaluation in the meta evaluation")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of the pipeline to evaluate")
    parser.add_argument("--end_index", type=int, default=None, help="End index of the pipeline to evaluate")
    parser.add_argument("--gpus", type=int, default=1, help="Number of gpus to use for the evaluation")
    parser.add_argument("--max_length", type=int, default=128, help="Max tokens for the evaluation")
    parser.add_argument("--training_steps", type=int, default=5000, help="Number of training steps for the evaluation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the evaluation")
    parser.add_argument("--save_loc", type=str, default="meta_eval", help="Location to save the meta evaluation")
    parser.add_argument("--real_metric_loc", type=str, default="real_metrics/dict_real_metrics.json", help="Location of the real metrics")

    args = parser.parse_args()
    parent_dir = os.path.join(args.parent_dir1, args.parent_dir2)

    logger.add(f"logs/meta/{args.title}/{args.pipeline_location}.log", rotation="1 day", level="DEBUG", backtrace=True, diagnose=True)

    with logger.catch():

        evaluation_loc = os.path.join(parent_dir, args.evaluation_loc)
        save_folder = os.path.join(parent_dir, args.save_loc, args.pipeline_location)
        parent_dir_pipeline = os.path.join(parent_dir, args.pipeline_location)
        evaluation_loc = os.path.join(parent_dir, args.evaluation_loc)
        save_folder = os.path.join(parent_dir, args.save_loc, args.pipeline_location)
        EXCLUDE = ["finetune", "finetune_no_smoothing", "spelling_grammar", "vector_label_distinctness", "mauve_labeled", "linear", "average_closest_vector_distance", "labeled", "average_inverse_closest_vector_distance",
                "euclid_vector_distinctness", "labeled_vector_distinctness", "euclid_labeled_vector_distinctness", "vendi_score", "distinctness", "distinctness_labeled", 
                "self_blue_labeled", "self_bleu", "spacy_analysis", "vector_distinctness", "euclid_labeled_vector_distinctness", "perplexity", "average_vector_distance", "euclidean_average_vector_distance", "mauve_labeled", 
                "neural", "labeled_spacy_normalized_distinctness", "distinctness_averaged_labeled", "distinctness_averaged_spacy_labeled", 
                "euclidean_average_inverse_closest_vector_distance", "labeled_normalized_distinctness", "labeled_JS_divergence", "labeled_KL_divergence"]

        os.makedirs(save_folder, exist_ok=True)

        if args.gpus > 1 and args.end_index is None:
            logger.info("Running on multiple GPUs.")
            logger.info("Splitting the pipeline into multiple parts.")
            
            gpus = [i for i in range(args.gpus)]
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                gpus = [int(id_) for id_ in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]

            running_processes = [None for _ in range(args.gpus)]
            command = f"CUDA_VISIBLE_DEVICES={gpus[0]} python scripts/meta.py {args.pipeline_location} --cores {args.cores} --title {args.title} --parent_dir1 {args.parent_dir1} --parent_dir2 {args.parent_dir2} --evaluation_loc {args.evaluation_loc} --start_index 0 --end_index {args.cores - 1} --gpus 1 --max_length {args.max_length} --training_steps {args.training_steps} --batch_size {args.batch_size} --save_loc {args.save_loc} --real_metric_loc {args.real_metric_loc}"
            logger.info(f"Running command: {command}")
            running_processes[0] = subprocess.Popen(command, shell=True)
            current_dataset = args.cores - 1
            total_datasets = get_n_datasets(os.path.join(parent_dir_pipeline, "pipelines.json"))
            for i in range(1, args.gpus):
                command = f"CUDA_VISIBLE_DEVICES={gpus[i]} python scripts/meta.py {args.pipeline_location} --cores {args.cores} --title {args.title} --parent_dir1 {args.parent_dir1} --parent_dir2 {args.parent_dir2} --evaluation_loc {args.evaluation_loc} --no-real --start_index {current_dataset} --end_index {current_dataset + args.cores} --gpus 1  --max_length {args.max_length} --training_steps {args.training_steps} --batch_size {args.batch_size} --save_loc {args.save_loc} --real_metric_loc {args.real_metric_loc}"
                logger.info(f"Running command: {command}")
                running_processes[i] = subprocess.Popen(command, shell=True)
                current_dataset += args.cores
                if current_dataset >= total_datasets:
                    break

            while True:
                time.sleep(10)
                for i in range(args.gpus):
                    if running_processes[i] is not None and running_processes[i].poll() is not None:
                        command = f"CUDA_VISIBLE_DEVICES={gpus[i]} python scripts/meta.py {args.pipeline_location} --cores {args.cores} --title {args.title} --parent_dir1 {args.parent_dir1} --parent_dir2 {args.parent_dir2} --evaluation_loc {args.evaluation_loc} --no-real --start_index {current_dataset} --end_index {current_dataset + args.cores} --gpus 1 --max_length {args.max_length} --training_steps {args.training_steps} --batch_size {args.batch_size} --save_loc {args.save_loc} --real_metric_loc {args.real_metric_loc}"
                        logger.info(f"GPU {gpus[i]} finished. Running command: {command}")
                        running_processes[i] = subprocess.Popen(command, shell=True)
                        current_dataset += args.cores
                        if current_dataset > total_datasets:
                            break

                if current_dataset >= total_datasets:
                    break
                
            exit()
        
        dataset_range = None

        from dataset import Dataset
        from rewrite import *
        from generation import *
        from evaluation import *
        from pipeline import *
        from metaevaluation import MetaEvaluator
        from prompts import *

        evaluation_dataset = Dataset.load(evaluation_loc)
        parent_dir_pipeline = os.path.join(parent_dir, args.pipeline_location)
        evaluator = MetaEvaluator.load_from_multipipeline(os.path.join(parent_dir_pipeline, "pipelines.json"), 
                                                        evaluation_dataset, "finetune_temporal_fake_to_real.accuracy")
        
        if args.end_index is not None:
            dataset_range = (args.start_index, args.end_index)
        else:
            dataset_range = (args.start_index, len(evaluator.pipeline_datasets))

        real_metric_path = os.path.join(parent_dir, args.real_metric_loc)

        if not os.path.exists(real_metric_path):
            logger.warning(f"Real metric path {real_metric_path} does not exist. Not using real metrics.")
            real_metric_path = None

        evaluator.run("./templates", folder=save_folder, exclude=EXCLUDE, 
            title=args.title, first_phase_cores=args.cores, second_phase_cores=args.cores, include_eval=not args.no_real, dataset_range=dataset_range, 
            max_length=args.max_length, training_steps=args.training_steps, batch_size=args.batch_size, real_metrics_loc=real_metric_path)
