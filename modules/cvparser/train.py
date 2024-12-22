import argparse
import numpy as np
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import Adam
from utils.util import trim_entity_spans, convert_goldparse, ResumeDataset, get_hyperparameters, train_and_val_model, load_config


parser = argparse.ArgumentParser(description='Train Bert-NER')
parser.add_argument('-epoch', type=int, default=100, help='number of epochs')
parser.add_argument('-input', type=str, default='./data/Resumes.json',
                    help='input path data')
parser.add_argument('-output', type=str, default='./weights',
                    help='output path to save model state')
parser.add_argument('-config', type=str, default='./configs/config.yaml',
                    help='config path to setup')
parser.add_argument('-pre', type=bool, default=True,
                    help='output path to save model state')

args = parser.parse_args().__dict__

input_data = args['input']
output_path = args['output']
EPOCHS = args['epoch']

## load config file
CONFIG_PATH = args['config']
config = load_config(CONFIG_PATH)
MAX_GRAD_NORM = config["max_grad_norm"]
MAX_LEN = config["max_len"]
NUM_LABELS = config["num_labels"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN = config["pretrain"]
TOKENIZER = BertTokenizerFast(config["vocab"])
BATCH_SIZE = config("batch_size")
RATIO_SPLIT = config("ratio_split")
LR = config("lr")

tags_vals = config["tags_vals"]
idx2tag = {i: t for i, t in enumerate(tags_vals)}
tag2idx = {t: i for i, t in enumerate(tags_vals)}

## load pretrain model
# load model to continue training
if args['pre']:
    STATE_DICT = torch.load(config["model"], map_location=DEVICE)
    model = BertForTokenClassification.from_pretrained(
        PRETRAIN, state_dict=STATE_DICT['model_state_dict'], num_labels=NUM_LABELS)
# load pretrain model
else:
    model = BertForTokenClassification.from_pretrained(
        PRETRAIN, num_labels=len(tag2idx))
model.to(DEVICE)

data = trim_entity_spans(convert_goldparse(input_data))
total = len(data)

train_data, val_data = data[:total], data[int(total*RATIO_SPLIT):]

train_d = ResumeDataset(train_data, TOKENIZER, tag2idx, MAX_LEN)
val_d = ResumeDataset(val_data, TOKENIZER, tag2idx, MAX_LEN)

train_sampler = RandomSampler(train_d)
train_dl = DataLoader(train_d, sampler=train_sampler, batch_size=BATCH_SIZE)
val_dl = DataLoader(val_d, batch_size=BATCH_SIZE)

# optimizer 
optimizer_grouped_parameters = get_hyperparameters(model, True)
optimizer = Adam(optimizer_grouped_parameters, lr=LR)

train_and_val_model(
    model,
    TOKENIZER,
    optimizer,
    EPOCHS,
    idx2tag,
    tag2idx,
    MAX_GRAD_NORM,
    DEVICE,
    train_dl,
    val_dl,
    output_path
)
