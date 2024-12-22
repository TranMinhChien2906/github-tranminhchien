#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
from half_json.core import JSONFixer
from clear_text import *
import argparse

# set up parameter
parser = argparse.ArgumentParser(description='Convert data to train')
parser.add_argument('-input', type=str, help='input path file jsonl')
parser.add_argument('-output', type=str, default='../data/Resumes.json',
                    help='output path to save data')

args = parser.parse_args().__dict__

input_path = args['input']
output_path = args['output']

f_w = open(output_path, "w", encoding="utf-8")
data = [json.loads(line, encoding='utf-8') for line in open(input_path, 'r', encoding="utf-8")] # data exported from doccano web

for line in data:
    new_data = {}
    new_s = line["data"]
    new_data["content"] = new_s
    labels = line["label"]
    new_data["annotation"] = []
    for label in labels:
        start, end, sub_label = label
        if sub_label=="Skill":
            continue
        sub_data = {}
        sub_data["label"] = []
        sub_data["points"] = []
        text = line['data'][start:end]
        sub_data["label"].append(sub_label)
        sub_point = {}
        sub_point["start"] = start
        sub_point["end"] = end
        sub_point["text"] = text
        sub_data["points"].append(sub_point)
        new_data["annotation"].append(sub_data)
    new_data["extras"] = "null"
    
    new_data = json.dumps(new_data, ensure_ascii=False)
    try:
        s = remove_special_character(new_data)
        json.loads(s)   
        f_w.writelines(s)
    except:
        print(s)
    