#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import json
import logging
import numpy as np
import yaml
import torch
from tqdm import trange
from torch.utils.data import Dataset
from seqeval.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config

def convert_goldparse(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines = []
        with open(dataturks_JSON_FilePath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if "\\t" in line:
              continue
            data = json.loads(line)
            text = data['content']
            entities = []
            data_annotations = data['annotation']
            if data_annotations is not None:
                for annotation in data_annotations:
                    point = annotation['points'][0]
                    labels = annotation['label']
                    if not isinstance(labels, list):
                        labels = [labels]

                    for label in labels:
                        point_start = point['start']
                        point_end = point['end']
                        point_text = point['text']

                        lstrip_diff = len(point_text) - \
                            len(point_text.lstrip())
                        rstrip_diff = len(point_text) - \
                            len(point_text.rstrip())
                        if lstrip_diff != 0:
                            point_start = point_start + lstrip_diff
                        if rstrip_diff != 0:
                            point_end = point_end - rstrip_diff
                        entities.append((point_start, point_end + 1, label))
            training_data.append((text, {"entities": entities}))
        
        return training_data
    except Exception as e:
        logging.exception("Unable to process " +
                          dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        
        cleaned_data.append([text, {'entities': valid_entities}])
   
    return cleaned_data


def get_label(offset, labels):
    if offset[0] == 0 and offset[1] == 0:
        return 'O'
    for label in labels:
        if offset[1] >= label[0] and offset[0] <= label[1]:
            return label[2]
    return 'O'

def process_resume(data, tokenizer, tag2idx, max_len, is_test=False):
    tok = tokenizer.encode_plus(
        data[0], max_length=max_len, return_offsets_mapping=True, truncation = True)
    curr_sent = {'labels': []}

    padding_length = max_len - len(tok['input_ids'])
   
    if not is_test:
        labels = data[1]['entities']
        labels.reverse()
        for off in tok['offset_mapping']:
            label = get_label(off, labels)
            curr_sent['labels'].append(tag2idx[label])
        curr_sent['labels'] = curr_sent['labels'] + ([0] * padding_length)

    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)
    return curr_sent


class ResumeDataset(Dataset):
    def __init__(self, resume, tokenizer, tag2idx, max_len, is_test=False):
        self.resume = resume
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.resume)

    def __getitem__(self, idx):
        data = process_resume(
            self.resume[idx], self.tokenizer, self.tag2idx, self.max_len, self.is_test)

        return {
            'input_ids': torch.tensor(data['input_ids'], dtype=torch.long),
            'token_type_ids': torch.tensor(data['token_type_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(data['labels'], dtype=torch.long)
        }


def get_hyperparameters(model, ff):

    # ff: full_finetuning
    if ff:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer]}]

    return optimizer_grouped_parameters


def get_special_tokens(tokenizer, tag2idx):
    vocab = tokenizer.get_vocab()
    print(vocab)
    pad_tok = vocab["[PAD]"]
    sep_tok = vocab["[SEP]"]
    cls_tok = vocab["[CLS]"]
    o_lab = tag2idx["O"]

    return pad_tok, sep_tok, cls_tok, o_lab


def annot_confusion_matrix(valid_tags, pred_tags):
    """
    Create an annotated confusion matrix by adding label
    annotations and formatting to sklearn's `confusion_matrix`.
    """

    header = sorted(list(set(valid_tags + pred_tags)))

    matrix = confusion_matrix(valid_tags, pred_tags, labels=header)

    mat_formatted = [header[i] + "\t\t\t" +
                     str(row) for i, row in enumerate(matrix)]
    content = "\t" + " ".join(header) + "\n" + "\n".join(mat_formatted)

    return content


def flat_accuracy(valid_tags, pred_tags):
    return (np.array(valid_tags) == np.array(pred_tags)).mean()


def train_and_val_model(
    model,
    tokenizer,
    optimizer,
    epochs,
    idx2tag,
    tag2idx,
    max_grad_norm,
    device,
    train_dataloader,
    valid_dataloader,
    output_path
):

    pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, tag2idx)
    best_acc = -1
    epoch = 0
    for _ in trange(epochs, desc="Epoch"):
        epoch += 1

        # Training loop
        print("Starting training loop.")
        model.train()
        tr_loss, tr_accuracy = 0, 0
        nb_tr_examples, nb_tr_steps = 0, 0
        tr_preds, tr_labels = [], []

        for step, batch in enumerate(train_dataloader):
            # Add batch to gpu

            # batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            b_input_ids, b_input_mask, b_labels = b_input_ids.to(
                device), b_input_mask.to(device), b_labels.to(device)

            # Forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss, tr_logits = outputs[:2]

            # Backward pass
            loss.backward()

            # Compute train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
            )

            tr_logits = tr_logits.cpu().detach().numpy()
            tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            preds_mask = preds_mask.cpu().detach().numpy()
            tr_batch_preds = np.argmax(tr_logits[preds_mask.squeeze()], axis=1)
            tr_batch_labels = tr_label_ids.to("cpu").numpy()
            tr_preds.extend(tr_batch_preds)
            tr_labels.extend(tr_batch_labels)

            # Compute training accuracy
            tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
            tr_accuracy += tmp_tr_accuracy

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=max_grad_norm
            )

            # Update parameters
            optimizer.step()
            model.zero_grad()

        tr_loss = tr_loss / nb_tr_steps
        tr_accuracy = tr_accuracy / nb_tr_steps

        # save weight
        if best_acc < tr_accuracy:
            print("_________Best model_________")
            torch.save(
                {
                    "model_state_dict": model.state_dict()
                },
                f'{output_path}/model-state.bin',
            )
            
            best_acc = tr_accuracy
        
        # Print training loss and accuracy per epoch
        print(f"Train loss: {tr_loss}")
        print(f"Train accuracy: {tr_accuracy}")
        continue
        """
        Validation loop
        """
        print("Starting validation loop.")
        
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []

        for batch in valid_dataloader:

            b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            b_input_ids, b_input_mask, b_labels = b_input_ids.to(
                device), b_input_mask.to(device), b_labels.to(device)

            with torch.no_grad():
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                tmp_eval_loss, logits = outputs[:2]

            # Subset out unwanted predictions on CLS/PAD/SEP tokens
            preds_mask = (
                (b_input_ids != cls_tok)
                & (b_input_ids != pad_tok)
                & (b_input_ids != sep_tok)
            )

            logits = logits.cpu().detach().numpy()
            label_ids = torch.masked_select(b_labels, (preds_mask == 1))
            preds_mask = preds_mask.cpu().detach().numpy()
            val_batch_preds = np.argmax(logits[preds_mask.squeeze()], axis=1)
            val_batch_labels = label_ids.to("cpu").numpy()
            predictions.extend(val_batch_preds)
            true_labels.extend(val_batch_labels)

            tmp_eval_accuracy = flat_accuracy(
                val_batch_labels, val_batch_preds)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        # Evaluate loss, acc, conf. matrix, and class. report on devset
        pred_tags = [idx2tag[i] for i in predictions]
        valid_tags = [idx2tag[i] for i in true_labels]
        cl_report = classification_report(valid_tags, pred_tags)
        conf_mat = annot_confusion_matrix(valid_tags, pred_tags)
        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_steps
        if best_loss > eval_loss:
            print("_________Best model_________")
            torch.save(
                {
                    "model_state_dict": model.state_dict()
                },
                f'model-state.bin',
            )
            
            best_loss = eval_loss

        # Report metrics
        print(f"Validation loss: {eval_loss}")
        print(f"Validation Accuracy: {eval_accuracy}")
        print(f"Classification Report:\n {cl_report}")
        print(f"Confusion Matrix:\n {conf_mat}")

def tokenize_resume(text, tokenizer, max_len):
    tok = tokenizer.encode_plus(
        text, max_length=max_len, return_offsets_mapping=True, truncation=True)

    curr_sent = dict()

    padding_length = max_len - len(tok['input_ids'])

    curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
    curr_sent['token_type_ids'] = tok['token_type_ids'] + \
        ([0] * padding_length)
    curr_sent['attention_mask'] = tok['attention_mask'] + \
        ([0] * padding_length)

    final_data = {
        'input_ids': torch.tensor(curr_sent['input_ids'], dtype=torch.long),
        'token_type_ids': torch.tensor(curr_sent['token_type_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(curr_sent['attention_mask'], dtype=torch.long),
        'offset_mapping': tok['offset_mapping']
    }

    return final_data


def predict(model, tokenizer, idx2tag, device, test_resume, max_len, resticted_lables):
    model.eval()
    data = tokenize_resume(test_resume, tokenizer, max_len)
    input_ids, input_mask = data['input_ids'].unsqueeze(
        0), data['attention_mask'].unsqueeze(0)
    labels = torch.tensor([1] * max_len,
                          dtype=torch.long).unsqueeze(0)

    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            token_type_ids=None,
            attention_mask=input_mask,
            labels=labels,
        )
        tmp_eval_loss, logits = outputs[:2]

    logits = logits.cpu().detach().numpy()
    label_ids = np.argmax(logits, axis=2)

    entities = []
    for label_id, offset in zip(label_ids[0], data['offset_mapping']):
        curr_id = idx2tag[label_id]
        curr_start = offset[0]
        curr_end = offset[1]
        if curr_id not in resticted_lables:
            if len(entities) > 0 and entities[-1]['entity'] == curr_id and curr_start - entities[-1]['end'] in [0, 1]:
                entities[-1]['end'] = curr_end
            else:
                entities.append(
                    {'entity': curr_id, 'start': curr_start, 'end': curr_end})
    for ent in entities:
        ent['text'] = test_resume[ent['start']:ent['end']]
    return entities
