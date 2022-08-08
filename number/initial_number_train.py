import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
### 본 폴더에 있는 digits.png 내에 있는 500개의 손글씨 이미지로 손글씨를 학습시킨 데이터를 npz파일로 저장합니다. ###
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
now_dir = os.path.dirname(os.path.abspath(__file__))

img = cv2.imread(now_dir + '/digits.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#20x20인 손 글씨를 50x70 크기로 확장합니다.
expand = cv2.resize(gray, None, fx=2.5, fy=3.5, interpolation=cv2.INTER_CUBIC)

#이진화(중간값)
ret, thresh = cv2.threshold(expand, 127, 255, cv2.THRESH_BINARY)

# 세로로 50줄, 가로로 100줄로 사진을 나눕니다.
cells = [np.hsplit(row, 100) for row in np.vsplit(thresh, 50)]
x = np.array(cells)
#print(x.shape) # (50, 100, 70, 50) 세로 50개, 가로 100개 -> 50x70의 데이터 (2차원 배열의 요소가 2차원 배열임
#print(x[0][0]) 
#print(cells[0][0])

# 각 (50 X 70) 크기의 사진을 한 줄(1 X 3500)으로 바꿉니다. 2차원 배열 -> 1차원 배열 (벡터)
train = x[:, :].reshape(-1, 3500).astype(np.float32)
#print(train.shape)

# 0이 500개, 1이 500개, ... 로 총 5,000개가 들어가는 (1 x 5000) 배열을 만듭니다.
k = np.arange(10)
train_labels = np.repeat(k, 500)[:, np.newaxis] 
#print(train_labels)

np.savez(now_dir + "/trained.npz", train=train, train_labels=train_labels) #정답 데이터 저장
