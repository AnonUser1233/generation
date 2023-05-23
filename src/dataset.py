import numpy as np
import pandas as pd
import pickle
import spacy
from transformers import BertTokenizer, BertModel
import os
from copy import deepcopy
from loguru import logger
import gc
import torch
from utils import get_best_device
import re
from pandas.errors import ParserError
from sklearn.model_selection import StratifiedKFold

class Dataset:
    def __init__(self, sentences=[], labels=[]) -> None:
        if len(sentences) > 0 and isinstance(sentences[0], (list, tuple)):
            self.df = pd.DataFrame({"text": [sentence[0] for sentence in sentences], "label": labels, 
                                    "other_text": [sentence[1] for sentence in sentences]})
        elif len(sentences) > 0:
            self.df = pd.DataFrame({"text": sentences, "label": labels})
            self.remove_empty()
        else:
            self.df = pd.DataFrame(columns=["text", "label"])
        self.doc = None
        self.labels_doc = dict()
        self.vectors = None
        self.save_path = None
        self.other_text_vectors = None

    def remove_empty(self, min_length=2):
        self.df = self.df[self.df["text"] != ""].reset_index(drop=True)
        lengths = np.array([len(text) >= min_length for text in self.df["text"]])
        self.df = self.df[lengths].reset_index(drop=True)

    def is_dual_task(self):
        return "other_text" in self.df.columns

    def sort_by_labels(self):
        self.df = self.df.sort_values(by=["label"]).reset_index(drop=True)

    def get_doc(self, label=None, nlp=None, nlp_max_length=10 ** 7):
        if label is None:
            if self.doc is None:
                logger.trace(f"Calculating doc  for {str(self)}...")
                if nlp is None:
                    nlp = spacy.load("en_core_web_sm")
                    nlp.max_length = nlp_max_length
                text = " ".join(self.df["text"])
                if len(text) > nlp_max_length:
                    logger.warning(f"Dataset is too long for spacy, truncating from {len(text)} to {nlp_max_length} characters")
                self.doc = nlp(text[:nlp_max_length])
        
            return self.doc
        
        if self.labels_doc.get(label) is None:
            logger.trace(f"Calculating doc for label {label}  for {str(self)}...")
            if nlp is None:
                nlp = spacy.load("en_core_web_sm")
                nlp.max_length = nlp_max_length
            
            text = " ".join(self.df[self.df["label"] == label]["text"])
            if len(text) > nlp_max_length:
                logger.warning(f"Dataset is too long for spacy, truncating from {len(text)} to {nlp_max_length} characters")
            self.labels_doc[label] = nlp(text[:nlp_max_length])
        
        return self.labels_doc[label]
    
    def vectorize_sentences(self, sentences, batch_size):
        tokenizer = BertTokenizer.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')
        device = torch.device(get_best_device())
        model = BertModel.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased').to(device)
        vectors = None

        for i in range(0, len(sentences), batch_size):
            encoded_sentences = tokenizer(list(sentences[i:i+batch_size]), 
                                          truncation=True, padding='max_length', max_length=512, return_tensors='pt')
            input_ids = encoded_sentences['input_ids']
            attention_mask = encoded_sentences['attention_mask']

            with torch.no_grad():
                output = model(input_ids.to(device), attention_mask=attention_mask.to(device))
                last_hidden_states = output.last_hidden_state
            # Calculate the mean of the token embeddings to get the sentence embedding
            sentence_embeddings = last_hidden_states.mean(axis=1).detach().cpu().numpy()
            # Append the sentence embeddings to the `vectors` list
            if i == 0:
                vectors = sentence_embeddings
            else:
                vectors = np.vstack((vectors, sentence_embeddings))
        
        del model, tokenizer
        torch.cuda.empty_cache()
        return vectors
    
    def get_vectors(self, batch_size=64, label=None, include_other=False):
        if self.vectors is None:
            logger.trace(f"Calculating vectors for {str(self)}...")
            self.vectors = self.vectorize_sentences(self.df["text"], batch_size)

        return_vectors = self.vectors
        if label is not None:
            return_vectors = self.vectors[np.array(self.df["label"]) == label]

        if not include_other:
            return return_vectors
        
        if self.is_dual_task() and self.other_text_vectors is None:
            logger.trace(f"Calculating other vectors for {str(self)}...")
            self.other_text_vectors = self.vectorize_sentences(self.df["other_text"], batch_size)

        return_other_vectors = self.other_text_vectors
        if label is not None and self.is_dual_task():
            return_other_vectors = self.other_text_vectors[np.array(self.df["label"]) == label]

        if self.is_dual_task():
            return return_vectors, return_other_vectors
        return return_vectors, None

    def split(self, n_parts):
        fold = StratifiedKFold(n_splits=n_parts, shuffle=True, random_state=42)
        train_data, test_data = [], []
        for test_index, train_index in fold.split(self.df, self.df["label"]):
            used_test_index = test_index[:len(train_index)]
            train = Dataset(sentences=list(self.df.iloc[train_index]["text"]), labels=list(self.df.iloc[train_index]["label"]))
            train.save_path = self.save_path
            test = Dataset(sentences=list(self.df.iloc[used_test_index]["text"]), labels=list(self.df.iloc[used_test_index]["label"]))
            test.save_path = self.save_path
            if self.vectors is not None:
                train.vectors = self.vectors[train_index]
                test.vectors = self.vectors[used_test_index]
            
            train_data.append(train)
            test_data.append(test)

        return train_data, test_data

    def copy(self):
        copy = Dataset()
        copy.df = self.df.copy()
        copy.doc = deepcopy(self.doc)
        copy.labels_doc = deepcopy(self.labels_doc)
        copy.vectors = deepcopy(self.vectors)
        return copy
    
    def reset_calculation(self, except_vectors=False):
        self.doc = None
        self.labels_doc = dict()
        if not except_vectors:
            self.vectors = None
            self.other_text_vectors = None

    def release_memory(self):
        logger.debug(f"Releasing memory for dataset {str(self)}")
        del self.doc, self.labels_doc, self.vectors, self.other_text_vectors
        gc.collect()
        self.reset_calculation()

    def get_label_col(self):
        return np.array(self.df["label"])

    def calculate_all(self):
        logger.debug(f"Calculating all for {str(self)}...")
        self.get_doc()
        self.get_vectors(include_other=self.is_dual_task())
        for label in self.get_labels():
            self.get_doc(label)

    def append(self, sentences, labels, other_sentences=None):
        if len(sentences) == 0:
            return
        if isinstance(sentences[0], (list, tuple)):
            other_sentences = [sentence[1] for sentence in sentences]
            sentences = [sentence[0] for sentence in sentences]
        if other_sentences is not None:
            self.append_df(pd.DataFrame({"text": sentences, "label": labels, "other_text": other_sentences}))
        else:
            self.append_df(pd.DataFrame({"text": sentences, "label": labels}))

    def get_labels(self):
        return np.unique(self.df["label"])

    def get_sentences(self, label=None, include_other=False):
        if not include_other:
            if label is None:
                return np.array(self.df["text"])
            return np.array(self.df[self.df["label"] == label]["text"])
        
        out_right = None
        if self.is_dual_task():
            if label is None:
                out_right = np.array(self.df["other_text"])
            else:
                out_right = np.array(self.df[self.df["label"] == label]["other_text"])
            
        if label is None:
            return np.array(self.df["text"]), out_right
        return np.array(self.df[self.df["label"] == label]["text"]), out_right

    def shuffle(self):
        logger.debug(f"Shuffling dataset {str(self)}...")
        random_perm = np.random.permutation(self.df.index)
        self.df = self.df.reindex(random_perm).reset_index(drop=True)
        if self.vectors is not None:
            self.vectors = self.vectors[random_perm]
        if self.other_text_vectors is not None:
            self.other_text_vectors = self.other_text_vectors[random_perm]

    def get_all(self, include_other=False):
        if self.is_dual_task() and include_other:
            return np.array(self.df["text"]), np.array(self.df["other_text"]), np.array(self.df["label"])
        if include_other:
            return np.array(self.df["text"]), None, np.array(self.df["label"])
        return np.array(self.df["text"]), np.array(self.df["label"])
    
    def append_df(self, df):
        self.df = pd.concat([self.df, df], ignore_index=True)
        self.reset_calculation()

    def change_sentence(self, index, new_sentence):
        self.df.at[index, "text"] = new_sentence
        self.reset_calculation()

    def change_other_sentence(self, index, new_sentence):
        self.df.at[index, "other_text"] = new_sentence
        self.reset_calculation()

    def size(self, label=None):
        if label is None:
            return len(self.df)
        else:
            return np.count_nonzero(self.df["label"] == label)

    def remove(self, index):
        self.df.drop(index, inplace=True)
        if self.vectors is not None:
            self.vectors = np.delete(self.vectors, index, axis=0)
        self.reset_calculation(except_vectors=True)
        self.df.reset_index(drop=True, inplace=True)
 
    def remove_sentence(self, sentence):
        self.df = self.df[self.df["text"] != sentence]
        self.reset_calculation()

    def get(self, index, include_other=False):
        if self.is_dual_task():
            sentence, other_sentence, label = self.df.iloc[index]
            return sentence, other_sentence, label
        sentence, label = self.df.iloc[index]

        if include_other:
            return sentence, None, label
        return sentence, label
    
    def get_label_sentences(self, label, include_other=False):
        if not include_other:
            return np.array(self.df[self.df["label"] == label]["text"]), np.where(self.df["label"] == label)[0]
        if not self.is_dual_task():
            return np.array(self.df[self.df["label"] == label]["text"]), None, np.where(self.df["label"] == label)[0]
        return np.array(self.df[self.df["label"] == label]["text"]), \
                np.array(self.df[self.df["label"] == label]["other_text"]), np.where(self.df["label"] == label)[0]
    
    def clean_text(self, text):
        text = re.sub(r'([a-z])-([a-z])', r'\1 \2', text) # Replace hyphens between words with space
        text = re.sub(r'[^a-z\s]', '', text) # Remove other punctuations
        text = re.sub(r'\s{2,}', ' ', text) # Replace double spaces with single space
        return text.strip()

    def clean(self):
        logger.debug(f"Cleaning dataset text for {self}")
        original_text = self.df["text"].copy()
        self.df["text"] = self.df["text"].str.lower()
        self.df['text'] = self.df['text'].apply(self.clean_text)
        if not np.all(original_text == self.df["text"]):
            self.reset_calculation()
        
    def save(self, folder_or_file=None, save_data_only=True, calculate_all=False, clean=False):
        if clean:
            self.clean()
        if calculate_all:
            self.calculate_all()
        if save_data_only:
            os.makedirs(os.path.dirname(folder_or_file), exist_ok=True)
            self.df.to_csv(folder_or_file, index=False, escapechar='\\')
        else:
            os.makedirs(folder_or_file, exist_ok=True)
            self.df.to_csv(os.path.join(folder_or_file, "dataset.csv"), index=False)
            
            with open(os.path.join(folder_or_file, "doc.pkl"), "wb") as f:
                pickle.dump([self.doc, self.labels_doc], f)

            with open(os.path.join(folder_or_file, "vectors.pkl"), "wb") as f:
                pickle.dump(self.vectors, f)

        self.save_path = folder_or_file
        logger.debug(f"Dataset saved to {folder_or_file}.")
    
    @staticmethod
    def load(folder_or_file, shuffle=True, clean=False):
        dataset = Dataset()
        if not os.path.exists(folder_or_file):
            raise ValueError(f"Path {folder_or_file} does not exist")
                
        if os.path.isfile(os.path.join(folder_or_file)):
            try:
                dataset.df = pd.read_csv(folder_or_file, escapechar='\\')
            except ParserError:
                dataset.df = pd.read_csv(folder_or_file)
        else:
            try:
                dataset.df = pd.read_csv(os.path.join(folder_or_file, "dataset.csv"), escapechar='\\')
            except ParserError:
                dataset.df = pd.read_csv(os.path.join(folder_or_file, "dataset.csv"))

            if os.path.isfile(os.path.join(folder_or_file, "doc.pkl")):
                with open(os.path.join(folder_or_file, "doc.pkl"), "rb") as f:
                    dataset.doc, dataset.labels_doc = pickle.load(f)

            if os.path.isfile(os.path.join(folder_or_file, "vectors.pkl")):
                with open(os.path.join(folder_or_file, "vectors.pkl"), "rb") as f:
                    dataset.vectors = pickle.load(f)

        dataset.df = dataset.df[dataset.df["label"].notna()]
        dataset.df.fillna("", inplace=True)

        dataset.save_path = folder_or_file

        dataset.remove_empty()

        if shuffle:
            dataset.shuffle()

        if clean:
            dataset.clean()

        logger.debug(f"Loaded {dataset}")

        return dataset
    
    def __str__(self):
        return f"Dataset(size={self.size()}, labels={self.get_labels()}, save_path={self.save_path})"
