## Source Code for "Understanding Dataset Generation with Foundation Models"

This repository contains the source code for "Understanding Dataset Generation with Foundation Models". Please follow the following instructions to reproduce all results presented in the paper along with the associated plots.

### Create environment
Create the environment by installing Conda and running the following command:
```bash
conda create -n foundation python=3.10
```
Then install all requirements by activating the environment in this folder and installing the requirements:
```bash
conda activate foundation
pip install -r requirements.txt
```
Additionally, install the `en_core_web_sm` package of Spacy:
```bash
python -m spacy download en_core_web_sm
```

If you want to run the dataset generation process, the environment variable `OPENAI_API_KEY` needs to exist and point to your correct API key. You can for example do this by putting a `.env` file in the `src` folder and adding the following line:
```bash
OPENAI_API_KEY=[YOUR_API_KEY]
```

### Using previously generated datasets
In order to get the exact raw results from our paper, one can extract all files in the `data/generations` subfolder and run in the `src` folder:
```bash
python scripts/plots.py
```

This script will put all plots in the `plots` subfolder and the raw processed results in the `processed` subfolder.


### Generating datasets
Again, from the `src` folder generate all required dataset by running:
```bash
bash scripts_generations/main.sh
```
The generated datasets will be uploaded to `data/generations`
This command assumes that the environment variable `OPENAI_API_KEY` exists and we note that generation can take several days and costs 1000 USD. We have uploaded our datasets on TODO PROVIDE LINK. The folder should be put in the main folder with the name `data`.
### Preprocessing the evaluation data

First, download the datasets from the following sources and put them in the folder `data/datasets/[DATASET_NAME]`:
- [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset)
- SST-2: Nothing to do, the datasets is automatically downloaded
- eli5: Unfortunately, due to a change in the [license](https://www.reddit.com/r/reddit/comments/12qwagm/an_update_regarding_reddits_api/) the dataset is not available for download anymore. Follow news on [this webpage](https://huggingface.co/datasets/eli5) for possible updates.
- AGNews: Download from [this link](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)

Then, in thr `src` folder, run 
```
bash scripts/preprocess.py
```
This might take a while.
### Running the evaluation
For running the evaluation process, run from the `src`
```bash
python scripts/main.py
```
Before running this file, change the parameters at the beginning to settings appropriate for your setup. `GPUS` and `taskset_range` should be lists of the same size that can be used for each job. `python` should point to the absolute path where the python version of your conda environment is located, 
you can run  `which python` on Linux when the environment is activated to find out where this path should point. Finally, `cores` refers to the amount of parallel processes that should be run on each GPU. Note that the peak memory usage is 4500MB and that your GPUs should have `cores * 4500` MB memory.

Note that results are automatically reloaded. Remove all previous information from the data/generations/**/meta_eval subfolders if you wish to recalculate everything.
### Creating the plots

In the `src` folder, run
```bash
python scripts/plots.py
```
This script will put all plots in the `plots` subfolder and the raw processed results in the `processed` subfolder