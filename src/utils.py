import multiprocessing as mp
from joblib import Parallel, delayed
import multiprocessing as mp
import torch
from loguru import logger
import numpy as np
import random
import tensorflow as tf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

def get_best_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def clear_memory_function(function, index, total, log_file=None, log_level=None):
    added = False
    handler_id = None
    if log_file is not None and len(logger._core.handlers) == 1:
        handler_id = logger.add(log_file, level=log_level, backtrace=True, diagnose=True)
        added = True

    logger.info(f"Starting job {index + 1}/{total}")
    output = function()
    logger.success(f"Ending job {index + 1}/{total}")
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        logger.warning(f"Unable to clear cache because of error {e}")
        pass
    if added:
        logger.remove(handler_id)
    return output

def run_parallel(functions, n_cores=1):
    logger.debug(f"Running {len(functions)} calculations in parallel on {n_cores} cores.")
    log_file = None
    log_level = None
    if len(logger._core.handlers) > 1:
        handler_id = list(logger._core.handlers)[-1]
        log_file = logger._core.handlers[handler_id]._sink._path
        log_level = logger._core.handlers[handler_id]._levelno

    with Parallel(n_jobs=n_cores) as parallel:
        output = parallel(delayed(clear_memory_function)(function, i, len(functions), 
                                                         log_file=log_file, log_level=log_level) for i, function in enumerate(functions))
    return output


def unnest_dictionary(dict_):
    result_dict = dict()
    for key in dict_:
        if isinstance(dict_[key], dict):
            unnested_subdict = unnest_dictionary(dict_[key])
            for subkey in unnested_subdict:
                result_dict[str(key) + "." + str(subkey)] = unnested_subdict[subkey]
        else:
            result_dict[key] = dict_[key]
    
    return result_dict