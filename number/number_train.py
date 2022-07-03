import cv2
import numpy as np
import os
# 미리 생성한 숫자당 500개의 손글씨 이미지로 손글씨를 학습시킨 데이터를 npz파일로 저장합니다.

now_dir = os.path.dirname(os.path.abspath(__file__))

img = cv2.imread(now_dir + '/digits.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 세로로 50줄, 가로로 100줄로 사진을 나눕니다.
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)
#print(x.shape) # (50, 100, 20, 20) 세로 50개, 가로 100개 -> 20x20의 데이터 (2차원 배열의 요소가 2차원 배열임
#print(x[0][0]) 

# 각 (20 X 20) 크기의 사진을 한 줄(1 X 400)으로 바꿉니다. 2차원 배역 -> 1차원 배열 (벡터)
train = x[:, :].reshape(-1, 400).astype(np.float32)
#print(train.shape)

# 0이 500개, 1이 500개, ... 로 총 5,000개가 들어가는 (1 x 5000) 배열을 만듭니다.
k = np.arange(10)
train_labels = np.repeat(k, 500)[:, np.newaxis] 
#print(train_labels.shape)

np.savez(now_dir + "/trained.npz", train=train, train_labels=train_labels) #정답 데이터 저장
