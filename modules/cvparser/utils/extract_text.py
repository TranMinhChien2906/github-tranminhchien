# -*- coding: utf8 -*-
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams
from pdfminer.converter import PDFPageAggregator
from pdf2image import convert_from_path
from utils.extract_face_image import face_image_extract
import pdfminer
import cv2
import numpy as np
from utils.clear_text import *
import pdfplumber
import itertools
from tika import parser

def pdfminer_extract(path,param):
    if "LTTextBox" in param:
        param = pdfminer.layout.LTTextBox
    elif "LTTextLine" in param:
        param = pdfminer.layout.LTTextLine
    else:
        assert False,"False"
    fp = open(path, 'rb')

    # Create a PDF parser object associated with the file object.
    parser = PDFParser(fp)

    # Create a PDF document object that stores the document structure.
    # Password for initialization as 2nd parameter
    document = PDFDocument(parser)

    # Check if the document allows text extraction. If not, abort.
    if not document.is_extractable:
        raise PDFTextExtractionNotAllowed

    # Create a PDF resource manager object that stores shared resources.
    rsrcmgr = PDFResourceManager()

    # Create a PDF device object.
    device = PDFDevice(rsrcmgr)

    # BEGIN LAYOUT ANALYSIS
    # Set parameters for analysis.
    laparams = LAParams()

    # Create a PDF page aggregator object.
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)

    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    def parse_obj(lt_objs):
        arr = []
        # loop over the object list
        for obj in lt_objs:
            # if it's a textbox, print text and location
            if isinstance(obj, param):
                text = obj.get_text().replace('\n', ' ')
                text_clean = text.strip()

                start_index = text.find(text_clean)
                end_index = start_index + len(text_clean)
                if start_index > 0:
                    start_index-=1
                left, upper, right,lower = obj.bbox

                step = (right-left) / len(text)
                right = left + step*end_index
                left = left + step*start_index
                arr.append([left, upper, right,lower, obj.get_text()])

            # if it's a container, recurse
            else:
                try:
                    tmp_arr = parse_obj(obj._objs)
                    arr.extend(tmp_arr)
                except:
                    pass
        return arr
    result = []
    # loop over all pages in the document
    sizes = None
    for page in PDFPage.create_pages(document):
        if sizes is None:
            sizes = page.mediabox
           
        # read the page into a layout object
        interpreter.process_page(page)
        layout = device.get_result()
        # extract text from this object
        result.append(parse_obj(layout._objs))
        
    return result,sizes

def extract_box(path, param):
    pred_boxes = []
    try:
        boxes,sizes = pdfminer_extract(path, param)
        images = convert_from_path(path)
        
        if len(images)!= len(boxes):
            return None
        for i, img in enumerate(images):
            boxs_perI = []
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results = []
            for box in boxes[i]:
                if len(box[4].replace("\xa0","").strip())<1:
                    continue
                results.append([box[0],sizes[3] - box[1],box[2],sizes[3] - box[3],box[4].replace("\xa0","")])

            if len(results) == 0:
                continue
            for box in results:
                left, lower, right, upper, text  = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
                left -=5
                right +=5
                
                left,right = (left/sizes[2])*image.shape[1] ,(right/sizes[2])*image.shape[1] 
                upper,lower = (upper/sizes[3])*image.shape[0] ,(lower/sizes[3])*image.shape[0] 
                boxs_perI.append((left, upper, right, lower, text))

            pred_boxes.append(boxs_perI)
        
    except Exception as e:
        print(e,path)
        return None
    return pred_boxes, images

# use pdfplumber for CV 1 column
def pdfplumber_extract(path):
    result_texts = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text is None:
                continue
            ''' loai bo cac ky tu lien tiep trung nhau do thu vien
            VD: "TTHHÔÔNNGG  TTIINN  TTHHÊÊMM" => "THÔNG TIN THÊM"
            '''
            text = [t for t in text.split("\n") if len(t.strip())>0]
            text = " \n".join(text)
            result_texts += "\n" + text
    result_texts = remove_special_character(result_texts)
    return result_texts

# sort box, text follow box output from pdfminer
def sort(boundings, texts):
    arr_bounds = np.asarray(boundings)
    arr_bounds[:,2] = arr_bounds[:,2]-arr_bounds[:,0]
    arr_bounds[:,3] = arr_bounds[:,3]-arr_bounds[:,1]
    ind_list = np.lexsort((arr_bounds[:, 0], arr_bounds[:, 1]))
    y_box = arr_bounds[ind_list]
    texts = texts[ind_list]
    idx = 0
    sort = []
    for i, cnt in enumerate(y_box):
        x, y, w, h = y_box[i-1]
        x1, y1, w1, h1 = y_box[i]
        if((y1-(y+h))>0):
            idx += 1
        sort.append((idx, x1, y1, w1, h1))
    arr_sort= np.asarray(sort)
    ind_sort= np.lexsort((arr_sort[:, 1], arr_sort[:, 0]))
    x_box = arr_sort[ind_sort]
    texts = texts[ind_sort]
    return x_box, texts

# ratio of column
def ratio_(boxes, image):
    copy_image = np.zeros([image.shape[0],image.shape[1],3],dtype=np.uint8)
    copy_image.fill(255)
    Y_cor = image.shape[0]*0.2
    min_left = 1e6
    max_right = -1
    for box in boxes:
        left, upper, right, lower, text = int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(box[4])
        if lower<Y_cor:
            continue
        if min_left > left:
            min_left = left
        if max_right < right:
            max_right = right
        cv2.rectangle(copy_image, (left, 0), (right, copy_image.shape[0]), (0, 0, 0), -1)
   
    # draw black all of two offset column
    cv2.rectangle(copy_image, (0, 0), (int(min_left), copy_image.shape[0]), (0, 0, 0), -1)
    cv2.rectangle(copy_image, (int(max_right), 0), (copy_image.shape[1], copy_image.shape[0]), (0, 0, 0), -1)
    imgray = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours)==0:
        ratio = 1.0
    else:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_center = int(x+w/2)
            if x_center/copy_image.shape[1] < 0.15 or x_center/copy_image.shape[1] > 0.85:
                ratio = 1.0
                continue
            ratio = x_center/copy_image.shape[1]

    return ratio

# get text follow column
def detect_line(boxes, image_, ratio_column):
    pred_boxes = []
    columns = []
    texts = []

    for box in boxes:
        if (int(box[0])==int(box[2])) or (int(box[1])==int(box[2])):
            continue
        left, upper, right, lower = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        texts.append(box[4])
        pred_boxes.append([left, upper, right, lower])
        if ratio_column*image_.shape[1] < left:
            columns.append(1)
        elif ratio_column*image_.shape[1] > right:
            columns.append(0)
        else:
            if (ratio_column*image_.shape[1]-left)/(right-left)>0.3:
                columns.append(0)
            else:
                columns.append(1)
   
    pred_boxes = np.array(pred_boxes)
    columns = np.array(columns)
    texts = np.array(texts)
   
    boxes_left = pred_boxes[columns==0]
    boxes_right = pred_boxes[columns==1]
    texts_left = texts[columns==0]
    texts_right = texts[columns==1]
    
    if len(boxes_left)>0:
        boxes_left, texts_left = sort(boxes_left, texts_left)
    if len(boxes_right)>0:
        boxes_right, texts_right = sort(boxes_right, texts_right)

    return texts_left, texts_right

# read file pdf
def pdf_extract(path):
    print ('-------path------------',path)
    pred_boxes, images = extract_box(path, "LTTextBox")
    base64 = face_image_extract(convert_from_path(path, fmt='jpeg'),None)
    ratio = ratio_(pred_boxes[0], np.asarray(images[0]))
    if ratio==1.0:
        texts = pdfplumber_extract(path)
        return texts,base64
        
    else:
        texts_left = ""
        texts_right = ""
        for pred_box, image in zip(pred_boxes, images):
            image = np.asarray(image)
            text_left, text_right = detect_line(pred_box, image, ratio)
            texts_left = texts_left+"\n"+" ".join(text for text in text_left)
            texts_right = texts_right+"\n"+" ".join(text for text in text_right)
        texts_left = remove_special_character(texts_left)
        texts_right = remove_special_character(texts_right)
        texts = texts_left+"\n"+texts_right
        return texts,base64

# read file docx
def doc_extract(path):
    texts = ""
    parsed = parser.from_file(path)
    contents = parsed["content"]
    for text in contents.split("\n"):
        text = text.strip()
        if text!="":
            texts += text+"\n"
    return texts
