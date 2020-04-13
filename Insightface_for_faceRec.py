# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
import sys
# from Face_Recongition_Edge import face_model
import face_model
import argparse
import threading
import time
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import base64

##———————————————————————————————————————————————————————
##  The parameter setup
imageDatasetPath='./Datasets/imgdata/'
configPath='./Datasets/label.txt'
modelPath='./models/model,0'
ageModelPath='./models_age/model,0'
rtspPath='rtsp://admin:qwer1234@192.168.20.14:554/h264/ch1/sub/av_stream'
headBoxPath="./Datasets/headframe/headFrame.png"
headBoxPath2="./Datasets/headframe/warning.png"
fontSize,fontSizePath=16,"./Datasets/platech.ttf"
gender_select={"男":'1','女':'0'}
##———————————————————————————————————————————————————————
##  read the configuration
def ReadTxtName(rootdir):
    """
    read the configuration which describe the the dataset , which consists of filenames, labels
    """
    filenames=[]
    labels=[]
    with open(rootdir, 'r',encoding="utf-8") as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            # strline = line.decode('gbk')
            part = line.strip().split(',')
            filename=part[0]
            label=part[1:-1]
            filenames.append(filename)
            labels.append(label)
        return filenames,labels
##———————————————————————————————————————————————————————
##  many thread to caputer the frame

# class myThread(threading.Thread):
#     """
#     through the rtsp , caputer the frame from the current environment
#     """
#     def __init__(self,rtspPath):
#         threading.Thread.__init__(self)
#         self.videoCapture = cv2.VideoCapture(rtspPath)
#         sucess, self.frame = self.videoCapture.read()
#     def run(self):
#         sucess, frame = self.videoCapture.read()
#         while (sucess):
#             sucess, frame = self.videoCapture.read()
#             if frame is None:
#                 continue
#             threadLock.acquire()
#             self.frame=frame
#             threadLock.release()
#     def get_img(self):
#         threadLock.acquire()
#         img=self.frame
#         threadLock.release()
#         return img

##———————————————————————————————————————————————————————
##  merge two images together
def mergeImg(image,src,bbox):
    """
    merge the src into the image, the zone is range from bbox
    """
    scale = 1.0
    locx,locy=int(bbox[0]),int(bbox[1])
    w,h=int(bbox[2]-bbox[0]),int(bbox[3]-bbox[1])
    dst=image[locy:locy+h,locx:locx+w]
    src=cv2.resize(src,(w,h))
    dst_channel=cv2.split(dst)
    src_channel=cv2.split(src)
    alpha=src_channel[3]*scale/255
    # alpha=scale*np.ones(dst_channel[0].shape)*0.2
    beta=1-alpha
    for i in range(3):
        dst_channel[i]=np.multiply(dst_channel[i],beta)
        dst_channel[i]+=np.multiply(src_channel[i],alpha)
    dst=cv2.merge(dst_channel,dst)
    return dst

##———————————————————————————————————————————————————————
##  show the result
def drawImg(image_base64,infos):
    # arr = base64.b64decode(image_base64)
    # nparr = np.fromstring(arr, np.uint8)
    # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image=image_base64

    img_h, img_w, _ = image.shape
    for info in infos:
        if info is None:
            continue

        box=info[0]
        xTopPos=max(0,np.floor(box[0]).astype(int))
        yTopPos=max(0,np.floor(box[1]).astype(int))
        xBotPos=min(img_w,np.floor(box[2]).astype(int))
        yBotPos = min(img_w, np.floor(box[3]).astype(int))
        box_w=xBotPos-xTopPos
        box_h=yBotPos-yTopPos
        # if info[-1] == False:
        #     headframe = cv2.imread(headBoxPath,-1)
        # elif info[-1] == True:
        #     headframe = cv2.imread(headBoxPath2,-1)
        headframe = cv2.imread(headBoxPath, -1)
        headframe = cv2.resize(headframe, (box_w, box_h))
        box = [xTopPos, yTopPos, xBotPos,yBotPos]
        dst = mergeImg(image, headframe, box)
        image[yTopPos:yBotPos,xTopPos:xBotPos] = dst

    img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype(fontSizePath, fontSize, 0)
    draw = ImageDraw.Draw(img_PIL)

    for text in infos:
        if text is None:
            continue
        name, age = str(text[1]),str(text[2])
        gender, max_pred = str(text[3]),str(text[4])
        max_len = np.max(len(name.encode()))
        max_cols = 16 * max_len
        loc_x,loc_y=int(text[0][0]),int(text[0][3])
        if (loc_y+76)>image.shape[0]:
            loc_y= image.shape[0] - 76
        if (loc_x+max_cols)>image.shape[1]:
            loc_x= image.shape[1] - max_cols

        # unicode_str = text[1].decode('utf-8')
        draw.text((loc_x,loc_y), name, (255, 255, 255), font=font)
        draw.text((loc_x,loc_y+20),age, (255, 255, 255), font=font)
        draw.text((loc_x,loc_y+40), gender, (255, 255, 255), font=font)
        draw.text((loc_x,loc_y+60), max_pred, (255, 255, 255), font=font)

    img_vis = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    cv2.imshow('drawimg', img_vis)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_vis


##———————————————————————————————————————————————————————
##  The main function for face recognition
class face_recognition():
    """
    init the face recognition, ready for detect the face and  recognition the target
    """
    def __init__(self):
        args =self.arg_parse()
        self.model = face_model.FaceModel(args)
        #提取带比对图片的特征
        self.tar_features,self.labels = self.extract_tar_feature()

    def arg_parse(self):
        # 1.2 set the parser
        parser = argparse.ArgumentParser(description='API for face-recognition')
        # general
        parser.add_argument('--image-size', default='112,112', help='')
        parser.add_argument('--model', default=modelPath, help='path to load model.')
        parser.add_argument('--ga-model', default=ageModelPath, help='path to load model.')
        parser.add_argument('--gpu', default=0, type=int, help='gpu id')
        parser.add_argument('--det', default=0, type=int,
                            help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        parser.add_argument('--threshold', default=0.709, type=float, help='ver dist threshold')
        return parser.parse_args()

    def extract_tar_feature(self):
        filenames, txtlabels = ReadTxtName(configPath)
        tarImgData = os.listdir(imageDatasetPath)
        numTarImg= len(tarImgData)

        tarFeauters,tarLabels = [],[]
        for i in range(numTarImg):
            selectImgName=tarImgData[i]
            imgPath=imageDatasetPath+selectImgName
            imgName = selectImgName.split(".")[0]
            j = filenames.index(imgName)
            label = txtlabels[j]
            # print("imgPath:{}".format(imgPath))
            # print("imgName:{}".format(imgName))

            frame = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), -1)
            # frame=cv2.imread(imgPath,-1)
            imgs, bboxs = self.model.get_input(frame)
            for img in imgs:
                if img is None:
                    continue
                f1 = self.model.get_feature(img)
                tarLabels.append(label)
                tarFeauters.append(f1)

        return tarFeauters,tarLabels

    def face_match(self, image_base64):
        # arr = base64.b64decode(image_base64)
        # nparr = np.fromstring(arr, np.uint8)
        # frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame=image_base64
        frames, bboxs = self.model.get_input(frame)

        info = []
        if  frames is not None:
            for index in range(len(frames)):
                img = frames[index]
                box = bboxs[index]
                f = self.model.get_feature(img)

                max_pred, max_index = 0, 0
                for tar_index in range(len(self.tar_features)):
                    # dist = np.sum(np.square(f - self.tar_f[tar_embs_index]))
                    sim = np.dot(f, self.tar_features[tar_index].T)
                    pred = sim * 0.5 + 0.5

                    if pred > max_pred:
                        max_pred = round(pred, 3)
                        max_index = tar_index

                # 3. 返回结果
                if max_pred > 0.75:
                    name=self.labels[max_index][0].split(':')[-1]
                    gender_label=self.labels[max_index][1].split(':')[-1]
                    gender = gender_select[gender_label]
                    age=self.labels[max_index][2].split(':')[-1]
                    midinfo = [box, name, gender, age, max_pred]
                    info.append(midinfo)
                else:
                    name = 'None'
                    gender, age = self.model.get_ga(img)
                    midinfo = [box, name, gender, age, round(pred, 3)]
                    info.append(midinfo)
        return info
        # return {'task_idx': 'face_match', 'face_info': info}

if __name__ == "__main__":
    ### Function 1: Data from the image
    ### process the single image
    face_model = face_recognition()
    frame = cv2.imread('./Datasets/test/test1.png')
    match_info = face_model.face_match(frame)
    print(match_info)
    drawImg(frame, match_info)

    ### Function 3: Read the image from the videos
    ### caputer the frame from the videos
    ### code fron the demo_sdb_test.py
    # face_model = face_recognition()
    # testVideoPath='./Datasets/1.mp4'
    # saveVideoPath='./Datasets/save-1.mp4'
    #
    # cap=cv2.VideoCapture(testVideoPath)
    # h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # writer = cv2.VideoWriter(saveVideoPath, fourcc, int(fps), (int(w), int(h)))
    # while True: # 循环读取视频帧
    #     rval, frame = cap.read()
    #     if frame is None:
    #         break
    #     match_info = face_model.face_match(frame)
    #     frame=drawImg(frame,match_info)
    #     writer.write(frame)
    # writer.release()