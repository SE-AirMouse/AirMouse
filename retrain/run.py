import cv2
import math
import numpy as np
import mediapipe as mp
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from mediapipe.framework.formats import landmark_pb2
from Net import *
train_with_cam = True #dont set this to false if its not lagging for better result
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image,s):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

  cv2.imshow(s, img)
  cv2.waitKey(1)


def apply_hue_offset(image, hue_offset):# 0 is no change; 0<=huechange<=180
    # convert img to hsv
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = img_hsv[:,:,0]
    s = img_hsv[:,:,1]
    v = img_hsv[:,:,2]
    # shift the hue
    img_hsv = cv2.add(h, hue_offset)
    # combine new hue with s and v
    img_hsv = cv2.merge([img_hsv,s,v])
    # convert from HSV to BGR
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    
def run():
    global time_s
    global points_depth
    global points_hand
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net()
    model.load_state_dict(torch.load('model1.ckpt'))
    model.to(device)
    model.eval()
    random_data = torch.rand((1, time_s * points_hand * points_depth)).to(device)
    result = model(random_data)

    cap = cv2.VideoCapture(0)
    num_of = 0
    Big_List=[]
    while(True):
        ret, frame = cap.read()

        resize_and_show(frame, 'n1')

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles


        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7) as hands:

                results = hands.process(cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1))

                image_hight, image_width, _ = frame.shape
                annotated_image = cv2.flip(frame.copy(), 1)
                
                if not results.multi_hand_landmarks:
                    if not train_with_cam:
                        continue
                    List = []
                    for j in range(points_hand):
                        List.append((0.5,0.5,0))
                    Big_List.append(List)
                    if len(Big_List) > 10:
                        Big_List.pop(0)
                    count=0
                    for j in range(len(Big_List)):
                        sum=0
                        for i in range(points_hand):
                            if Big_List[j][i][0]!=0.5 or Big_List[j][i][1]!=0.5 or Big_List[j][i][1]!=0.5:
                                sum += 1
                        if sum==0:
                            count+=1
                    if count<7 and len(Big_List)==10:
                        annotated_image = cv2.putText(annotated_image, '++++++', (50, 50),
                                                      cv2.FONT_HERSHEY_SIMPLEX,
                                                      1, (255, 0, 0), 2, cv2.LINE_AA)
                        BB = np.asarray(Big_List)
                        BB[:, :, 2] += 0.5
                        BB2 = torch.FloatTensor(BB)
                        BB2 = BB2.reshape(-1, time_s * points_hand * points_depth)
                        BB2 = BB2.to(device)
                        result = model(BB2)
                        l = []
                        res = result.cpu().detach().numpy()
                        res = np.reshape(res, (-1, points_hand, 3))
                        for j in range(points_hand):

                            nl = landmark_pb2.NormalizedLandmark()
                            nl.x = res[9][j][0]
                            nl.y = res[9][j][1]
                            nl.z = res[9][j][2]-0.5
                            l.append(nl)
                        landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark = l)

                        mp_drawing.draw_landmarks(
                            annotated_image,
                            landmark_subset,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                        resize_and_show(cv2.flip(annotated_image, 1), 'n2')

                        print(result)
                    continue

                for hand_landmarks in results.multi_hand_landmarks:

                    List = []
                    for L in mp_hands.HandLandmark:
                        List.append((hand_landmarks.landmark[L].x,hand_landmarks.landmark[L].y,hand_landmarks.landmark[L].z))
                    Big_List.append(List)
                    if len(Big_List) > 10:
                        Big_List.pop(0)



                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                resize_and_show(cv2.flip(annotated_image, 1), 'n2')
                
run()