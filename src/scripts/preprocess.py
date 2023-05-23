import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.getcwd()))
from dataset import Dataset
from datasets import load_dataset
import numpy as np

def preprocess_agnews():
    data = pd.read_csv("../data/datasets/AGNews/train.csv")
    data = pd.DataFrame(data)
    mapping = {
        1: "World",
        2: "Sports",
        3: "Business",
        4: "Sci/Tech",
        
    }
    data["label"] = data["Class Index"].apply(lambda x: mapping[x])
    texts = []
    for text in data["Title"]:
        if text.endswith(")"):
            text = "(".join(text.split("(")[:-1])
            texts.append(text)
        else:
            texts.append(text)

    data["text"] = texts
    dataset = Dataset(data["text"].tolist(), data["label"].tolist())
    dataset.calculate_all()
    dataset.save("../data/generations/AGNews/eval_data", save_data_only=False)

def preprocess_eli5():
    data = pd.read_csv("../data/datasets/eli5/eli5_train.csv")
    data.drop(columns=["document"], inplace=True)
    data.dropna(inplace=True)
    dataset = Dataset(data["title"].tolist(), data["subreddit"].tolist())
    dataset.save("../data/generations/eli5/eval_data", save_data_only=False, calculate_all=True)

def preprocess_goemotions():
    data = pd.concat([pd.read_csv(f"../data/datasets/GoEmotions/goemotions_{i}.csv") for i in range(1, 4)], ignore_index=True) 
    data = data[np.logical_not(data["example_very_unclear"])]
    emotions = [
        "gratitude", 
        "nervousness",
        "desire",
        "grief", 
        "surprise"
    ]
    emotion_data = []
    for emotion in emotions:
        em = data[data[emotion].astype(bool)]
        em["label"] = emotion
        emotion_data.append(em)
    data_new = pd.concat(emotion_data, ignore_index=True)
    dataset = Dataset(sentences=data_new["text"], labels=data_new["label"])
    dataset.save("../data/generations/goemotions/eval_data", save_data_only=False, calculate_all=True)

def preprocess_SST():
    data = load_dataset("sst2", split="train")
    data = pd.DataFrame(data)
    data["label"] = data["label"].apply(lambda x: "positive" if x == 1 else "negative")
    data = Dataset(data["sentence"], data["label"])
    data.save("../data/generations/SST/eval_data", save_data_only=False, calculate_all=True)

if __name__ == "__main__":
    try:
        preprocess_agnews()
    except Exception as e:
        print(f"AGNews failed: {e}")
    try:
        preprocess_eli5()
    except Exception as e:
        print(f"Eli5 failed: {e}")

    try:
        preprocess_goemotions()
    except Exception as e:
        print(f"GoEmotions failed: {e}")

    try:
        preprocess_SST()
    except Exception as e:
        print(f"SST failed: {e}")