import dlib
import cv2
import math 
import base64
import numpy as np
from PIL import Image
from datetime import datetime
import os 

def face_image_extract(img,face_landmarks = None):
    try :
        faces = None
        for item in img:
            image =np.asarray(item)#cv2.imread(img)
            img_gray = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            faces = detector(img_gray)
            if faces == None or len(faces) == 0 :
                continue
            else:
                break

        
        x1 = faces[0] .left() # các điểm bên trái
        y1 = faces[0] .top() # các điểm bên trên
        x2 = faces[0] .right() # các điểm bên phải
        y2 = faces[0] .bottom() # các điểm bên dưới
        dc = math.sqrt(math.pow((x2-x1),2) + math.pow((y2-y1),2))
        crop_img = image[int(-dc/3)+y1:y2+int(dc/dc/3),int(-dc/10)+ x1:x2+int(dc/10)]
        im = Image.fromarray(crop_img)
        now = datetime.now() # current date and time
        filename ='docs/'+ now.strftime("%m%d%Y%H:%M:%S")+'.jpeg'
        im.save(filename)
        my_string = ''
        with open(filename, "rb") as img_file:
            my_string = base64.b64encode(img_file.read())
        print(my_string)

        # os.remove(filename)
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("The file does not exist")
        return my_string
    except:
        return ""
    
