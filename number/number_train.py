import cv2
import numpy as np
import os
import glob
### 테스트를 통해 수집된 숫자 이미지로 추가 데이터에 대한 학습을 진행합니다. ###
now_dir = os.path.dirname(os.path.abspath(__file__))

# 머신러닝 된 데이터가 몇개인지 확인하는 코드
pre_data = np.load(now_dir + "/trained.npz")
p_train = np.array(pre_data['train'])
p_train_labels = np.array(pre_data['train_labels'])
print(p_train.shape)

# 어떤 숫자에 대한 데이터를 학습시킬 것인지 지정
for number in range(10):
    data_path = now_dir + '/image/{}'.format(number)

    # 축적된 숫자 데이터의 갯수
    data_num = len(os.listdir(data_path))
    # 데이터가 없으면 다음루프로 넘어간다
    if data_num == 0:
        print(f"{number}는 존재하지 않음")
        continue

    #train data를 저장할 빈 리스트 생성
    cells = []

    for file in glob.glob(data_path + '/*'):
        # 이미지 파일을 불러온다.
        img = cv2.imread(file)
        # 아래 과정을 거쳐주어야 50x70x1이 됨(rgb를 하나로)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 이미지 파일을 50x70 에서 1x3500으로 변형한다.
        img_res = gray.reshape(-1, 3500).astype(np.float32)
        #print(img_res.shape)
        cells.append(list(img_res))

    x = np.array(cells).astype(np.float32)
    #print(x)  
    n_train = x.reshape(data_num, -1)
    #print(n_train)  

    y = [number]*data_num
    y_ = np.array(y)
    n_train_labels = y_.reshape(data_num, -1)
    #print(n_train_labels)
    #print(n+train_labels.shape)
    
    #기존 데이터 불러오기
    pre_data = np.load(now_dir + "/trained.npz")
    p_train = np.array(pre_data['train'])
    p_train_labels = np.array(pre_data['train_labels'])
    
    #기존데이터와 합치기
    train = np.concatenate([p_train, n_train], axis=0)
    train_labels = np.concatenate([p_train_labels, n_train_labels], axis=0)
    #print(train.shape)
    #print(train_labels.shape)
    
    np.savez(now_dir + "/trained.npz", train=train, train_labels=train_labels) #정답 데이터 저장
    print(f'{number} 데이터 저장 완료')
    
    # 저장 완료된 이미지 파일들 삭제
    for file in glob.glob(data_path + '/*'):
        os.remove(file)
    print(f'{number} 데이터 삭제 완료')

