import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from loguru import logger 
from utils import get_best_device
import os, shutil
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pickle

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text, label = (row['text'],), row['label']
        if 'other_text' in self.dataframe.columns:
            text = (row['text'], row['other_text'])
        encoding = self.tokenizer.encode_plus(
            *text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long), 
            'index': index
        }
    
class LabelSmoothedTemporalVarianceCrossEntropyLoss(nn.Module):
    # https://arxiv.org/pdf/1512.00567.pdf
    # and https://arxiv.org/pdf/1610.02242.pdf, or better: https://arxiv.org/pdf/2211.03044.pdf
    def __init__(self, num_samples, num_classes, device, smoothing=0.0, alpha_temporal=0.9, lambda_temporal=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.alpha_temporal = alpha_temporal
        self.lambda_temporal = lambda_temporal
        self.ensemble_targets = torch.zeros(num_samples, num_classes, dtype=torch.float).to(device)
        self.num_classes = num_classes
        self.device = device
        
    def forward(self, logits, target, batch_indices):
        if self.ensemble_targets.size(0) < max(batch_indices) + 1:
            self.ensemble_targets = torch.cat((self.ensemble_targets, torch.zeros(max(batch_indices) + 1 - self.ensemble_targets.size(0), self.num_classes).to(self.device)), dim=0)

        probs = F.softmax(logits, dim=-1)
        log_softmax = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_softmax, target)
        smooth_loss = -log_softmax.mean(dim=-1).mean()
        try:
            ensemble_loss = F.kl_div(self.ensemble_targets[batch_indices], probs).mean()
            loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss + self.lambda_temporal * ensemble_loss
            self.ensemble_targets[batch_indices] = self.alpha_temporal * self.ensemble_targets[batch_indices] + (1 - self.alpha_temporal) * probs.detach()
        except RuntimeError as e:
            logger.warning(f"RuntimeError: {e} - settings loss to 0. Probably some problem with nll loss?")
            return None

        return loss

def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = TextDataset(df, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    return tokenizer

def run(data, batch_size=16, max_length=64, training_steps=5000, lr=1e-5, label_smoothing=0.1, device=None, alpha_temporal=0.9, 
        lambda_temporal=0.0, label_encoder=None, validation_data=None, temp_folder="./temp", max_epochs=10,
        persist_temp_folder=False, model_folder="model", label_encoder_filename="label_encoder.pkl", num_warmup_steps=600, reload_best_model=True):
    # Load your dataset
    os.makedirs(temp_folder, exist_ok=True)
    logger.debug(f"Running finetuning with parameters batch_size={batch_size}, max_length={max_length}, training_steps={training_steps}, lr={lr}, label_smoothing={label_smoothing}, device={device}, alpha_temporal={alpha_temporal}, lambda_temporal={lambda_temporal}")
    data = data.copy()
    label_count = len(data['label'].unique())
    if label_encoder is None:
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])
    else:
        data['label'] = label_encoder.transform(data['label'])

    if validation_data is not None:
        validation_data = validation_data.copy()
        validation_data['label'] = label_encoder.transform(validation_data['label'])
    # Initialize tokenizer and model
    tokenizer = get_tokenizer()
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=label_count)
    if persist_temp_folder:
        pickle.dump(label_encoder, open(os.path.join(temp_folder, label_encoder_filename), "wb"))

    data_loader = create_data_loader(data, tokenizer, max_length, batch_size)
    validation_data_loader = None
    loss_func_eval = None
    if validation_data is not None:
        validation_data_loader = create_data_loader(validation_data, tokenizer, max_length, batch_size)
        loss_func_eval = LabelSmoothedTemporalVarianceCrossEntropyLoss(len(validation_data_loader), 
                                                                       label_count, device, smoothing=0.0, alpha_temporal=0.0, lambda_temporal=0.0)
    loss_func = LabelSmoothedTemporalVarianceCrossEntropyLoss(len(data_loader), label_count, device, smoothing=label_smoothing, 
                                              alpha_temporal=alpha_temporal, lambda_temporal=lambda_temporal)
    

    # Set up training
    if device is None:
        device = torch.device(get_best_device())
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=training_steps)
    best_loss = None
    # Training loop
    model.train()
    steps = 0
    epochs = 0
    while steps < training_steps:
        sum_loss = 0
        count_loss = 0
        epochs += 1
        for batch in data_loader:
            steps += 1
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_func(outputs.logits, labels, batch["index"])
                if loss is None:
                    steps -= 1
                    continue
                else:
                    sum_loss += loss.item()
                    count_loss += 1

                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if steps > training_steps:
                    break
            except RuntimeError as e:
                logger.warning(f"RuntimeError: {e} - skipping step")
                steps -= 1
                continue

        
        logger.debug(f'Steps {steps}/{training_steps}, Loss: {sum_loss / max(count_loss, 1):.6f}')
        if validation_data_loader is not None:
            _, val_loss = evaluate(model, validation_data_loader, loss_func_eval, device, max_batches=len(data_loader))
            if reload_best_model and (best_loss is None or val_loss < best_loss):
                best_loss = val_loss
                model.save_pretrained(os.path.join(temp_folder, model_folder))

        if epochs >= max_epochs:
            break

    if best_loss is not None:
        model = DistilBertForSequenceClassification.from_pretrained(os.path.join(temp_folder, model_folder))
        model.to(device)
        if not persist_temp_folder:
            shutil.rmtree(temp_folder)

    return model, tokenizer, label_encoder, os.path.join(temp_folder, model_folder), os.path.join(temp_folder, label_encoder_filename)

def evaluate(model, data_loader, loss_func, device, cross_entropy=False, max_batches=None):
    model.eval()
    correct_predictions = 0
    loss_total = 0
    loss_count = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if max_batches is not None and i > max_batches:
                break
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                if cross_entropy:
                    loss = loss_func(outputs.logits, labels)
                else:
                    loss = loss_func(outputs.logits, labels, batch["index"])
                if loss is not None:
                    loss_total += loss.item()
                    loss_count += 1

                _, preds = torch.max(outputs.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
            except RuntimeError as e:
                logger.warning(f"RuntimeError: {e} - skipping step")
                continue

    accuracy = correct_predictions / len(data_loader.dataset)
    logger.debug(f'Validation Accuracy: {accuracy:.6f}')
    logger.debug(f'Validation Loss: {loss_total / (loss_count + 1):.6f}')
    return accuracy, loss_total / (loss_count + 1)


def predict(texts, model, tokenizer, label_encoder=None, device=None, max_length=64):
    if isinstance(texts, str):
        texts = [texts]
    
    # Encode all texts in the batch
    encoding = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    if device is None:
        device = torch.device(get_best_device())
        model.to(device)

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_labels = torch.argmax(probs, dim=1)
        if label_encoder is not None:
            predicted_labels_inverse = label_encoder.inverse_transform(predicted_labels.cpu().numpy())
        else:
            predicted_labels_inverse = predicted_labels.cpu().numpy()

    return predicted_labels_inverse, [probs[i, label].item() for i, label in enumerate(predicted_labels)]

def load_model(model_folder):
    model, tokenizer = DistilBertForSequenceClassification.from_pretrained(model_folder), DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

def evaluate_metrics(model, tokenizer, label_encoder, dataset_df, device, max_length=128, batch_size=8):
    predictions = []
    text, other_text, labels = dataset_df["text"], None, dataset_df["label"]
    if "other_text" in dataset_df.columns:
        other_text = dataset_df["other_text"]

    correct_indices = np.array([label in label_encoder.classes_ for label in labels])
    if not np.all(correct_indices):
        logger.warning(f"Found {len(correct_indices) - np.sum(correct_indices)} labels not in label encoder. Ignoring them.")

    text = text[correct_indices]
    if other_text is not None:
        other_text = other_text[correct_indices]
    labels = labels[correct_indices]

    df = dataset_df.copy()
    df["label"] = label_encoder.transform(df["label"])
    predictions = []
    complexity_score = []
    for i in range(0, len(text), batch_size):
        if other_text is None:
            pred, score = predict(list(text[i:i+batch_size]), model, tokenizer, label_encoder, device, max_length)
        else:
            pred, score = predict((list(text[i:i+batch_size]), list(other_text[i:i+batch_size])), model, tokenizer, label_encoder, device, max_length)

        predictions.extend(list(pred))
        complexity_score.extend(list(1 - np.array(score)))

    metrics = {
        "accuracy": float(accuracy_score(labels, np.array(predictions))), 
        "f1": float(f1_score(labels, np.array(predictions), average="macro")),
        "complexity_abs": np.mean(complexity_score),
    }
    return metrics
    