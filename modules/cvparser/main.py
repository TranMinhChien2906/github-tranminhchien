import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import uvicorn 
import jwt
from fastapi import FastAPI, Form,File, UploadFile
from starlette.requests import Request
from utils.util import load_config, predict
from utils.extract_text import doc_extract, pdf_extract

from pathlib import Path
from tempfile import NamedTemporaryFile
from requests import get
import re
import os 
import json
import time
import dlib 
import asyncio
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "https://histaff-ai-cv-extraction-information.histaff.online",
    "http://localhost:9000",
    "http:htp//172.16.1.129:9000",
    "http://0.0.0.0:9000",
    "http://127.0.0.1:9000"
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# load config file
CONFIG_PATH = "./configs/config.yaml"
config = load_config(CONFIG_PATH)
MAX_LEN = config["max_len"]
NUM_LABELS = config["num_labels"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN = config["pretrain"]
STATE_DICT = torch.load(config["model"], map_location=DEVICE)
TOKENIZER = BertTokenizerFast(config["vocab"], lowercase=True)
tags_vals = config["tags_vals"]
idx2tag = {i: t for i, t in enumerate(tags_vals)}
resticted_lables = config["resticted_lables"]
BASIC_INFOS = config["basic_infos"]
SKILLS = open(config["skill"], "r").read()
UPDATE_FILE = config["update_file"]
# face_landmarks = dlib.shape_predictor(config["landmarks"]) 
model = BertForTokenClassification.from_pretrained(
    PRETRAIN, state_dict=STATE_DICT['model_state_dict'], num_labels=NUM_LABELS)
model.to(DEVICE)
print('DEVICE',DEVICE)
def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

def resume(cv_url):
    t0 = time.time()
    ImageBase64 = ''
    suffix = Path(os.path.basename(cv_url)).suffix
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        download(cv_url, tmp_path)
    # print(tmp_path)
    print("Download ", time.time()-t0)
    t1 = time.time()
    if ".pdf" in str(tmp_path):
        resume_text,ImageBase64 = pdf_extract(tmp_path)
    else:
        resume_text = doc_extract(bytes(tmp_path))
    resume_text = resume_text.replace("\t", " \t")

    dicts = {}

    ### Output model
    entities = predict(model, TOKENIZER, idx2tag,
                        DEVICE, resume_text, MAX_LEN, resticted_lables)
    
    position = ""
    flag_position = False
    for entity in entities:
        label = entity["entity"]
        text = list(dict.fromkeys(entity["text"].split("\n")))
        text = " ".join(text)
        # clear output model (VD: "email" is not contain "@")
        if len(text)<=2:
            continue
        if label=="email":
            if "@" not in text:
                continue
        if label=="company position":
            if not flag_position:
                position = text
                flag_position = True
        if label=="phone":
            clear_text = re.sub(r"[^a-zA-Z0-9]","",text)
            if not clear_text.isdigit():
                continue
        if label not in dicts:
            dicts[label] = []
            dicts[label].append(text)
        else:
            dicts[label].append(text)

    ### If not Job Title => Job Title = position
    if "job title" not in dicts and "company position" in dicts:
        dicts["job title"] = []
        dicts["job title"].append(position)

    ### SKILL
    dicts["skill"]=[]
    for skill in SKILLS.split("\n"):
        if skill in resume_text.strip():
            if len(skill)>1:
                dicts["skill"].append(skill)

    for k, values in dicts.items():
        clear_values = list(dict.fromkeys(values))
        if len(clear_values)==1:
            dicts[k] = clear_values[0]
        if k in BASIC_INFOS:
            dicts[k] = " ".join(val for val in clear_values)
        else:
            dicts[k] = clear_values 
    dicts_new = {}
    for k,v in dicts.items():
        dicts_new[str(k).replace(' ','_')] = v
    dicts_new['imageBase64'] = ImageBase64
    print("Processing ", time.time()-t0)
    return dicts_new, resume_text

def update_label(dicts):
    update_data={}
    update_data["text"] = dicts["texts"]
    update_data["label"] = []
    new_text = dicts["texts"].replace("\n", " ").lower()
    annos = dicts["annos"]
    for k, vals in annos.items():
        sub_label = k
        if type(vals) is str:
            sub_text = vals.lower()
            if sub_text in new_text:
                subs = [sub.start() for sub in re.finditer(sub_text, new_text)] 
                for sub in subs:
                    start = sub
                    end = len(sub_text)+start
                    update_data["label"].append([start, end, sub_label])
        else:
            for val in vals:
                if val.lower() in new_text:
                    subs = [sub.start() for sub in re.finditer(val.lower(), new_text)] 
                    for sub in subs:
                        start = sub
                        end = len(val)+start
                        update_data["label"].append([start, end, sub_label])
    update_data = json.dumps(update_data, ensure_ascii=False)
    return update_data

@app.post("/predict_cv")
def predict_cv(request: Request, cv_url: str = Form(...)):
    if request.method == "POST":
        annos, texts = resume(cv_url)
        return {"predicted": annos, "text": texts}

@app.put("/update_cv")
async def update_cv(request: Request):
    print("request is ",request.json())
    input_data = await request.json()
    asyncio.sleep(delay)
    update_data = update_label(input_data)
    UPDATE_LABEL = open(UPDATE_FILE, "a")
    UPDATE_LABEL.writelines(update_data+"\n")
    UPDATE_LABEL.close()
    return {'alert': 'success'}
@app.get("/items/")
async def read_items():
    return [{"name": "Katana"}]
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    # if request.method == "POST":
    #     annos, texts = resume(file.filename)
    #     return {"predicted": annos, "text": texts}
    tmp_path = file.filename
    t0 = time.time()
    ImageBase64 = ''
    if ".pdf" in str(tmp_path):
        resume_text,ImageBase64 = pdf_extract(tmp_path)
    else:
        resume_text = doc_extract(bytes(tmp_path))
    resume_text = resume_text.replace("\t", " \t")

    dicts = {}

    ### Output model
    entities = predict(model, TOKENIZER, idx2tag,
                        DEVICE, resume_text, MAX_LEN, resticted_lables)
    
    position = ""
    flag_position = False
    for entity in entities:
        label = entity["entity"]
        text = list(dict.fromkeys(entity["text"].split("\n")))
        text = " ".join(text)
        # clear output model (VD: "email" is not contain "@")
        if len(text)<=2:
            continue
        if label=="email":
            if "@" not in text:
                continue
        if label=="company position":
            if not flag_position:
                position = text
                flag_position = True
        if label=="phone":
            clear_text = re.sub(r"[^a-zA-Z0-9]","",text)
            if not clear_text.isdigit():
                continue
        if label not in dicts:
            dicts[label] = []
            dicts[label].append(text)
        else:
            dicts[label].append(text)

    ### If not Job Title => Job Title = position
    if "job title" not in dicts and "company position" in dicts:
        dicts["job title"] = []
        dicts["job title"].append(position)

    ### SKILL
    dicts["skill"]=[]
    for skill in SKILLS.split("\n"):
        if skill in resume_text.strip():
            if len(skill)>1:
                dicts["skill"].append(skill)

    for k, values in dicts.items():
        clear_values = list(dict.fromkeys(values))
        if len(clear_values)==1:
            dicts[k] = clear_values[0]
        if k in BASIC_INFOS:
            dicts[k] = " ".join(val for val in clear_values)
        else:
            dicts[k] = clear_values 
    dicts_new = {}
    for k,v in dicts.items():
        dicts_new[str(k).replace(' ','_')] = v
    dicts_new['imageBase64'] = ImageBase64
    print("Processing ", time.time()-t0)
    os.unlink(file.filename)
    # input_data = await request.json()
    # status = await OKupdate_cv(input_data)
    return dicts_new, resume_text
# async def OKupdate_cv(input_data):
  
#     update_data = update_label(input_data)
#     UPDATE_LABEL = open(UPDATE_FILE, "a")
#     UPDATE_LABEL.writelines(update_data+"\n")
#     UPDATE_LABEL.close()
#     print ('alert', 'success')
#     return {'alert': 'success'}
if __name__ == "__main__":
	print("* Starting web service...")
	uvicorn.run(app, host=config["HOST"], port=9000 )#config["PORT"])

