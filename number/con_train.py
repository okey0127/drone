import numpy as np
import os

# 학습데이터를 합칩니다.

now_dir = os.path.dirname(os.path.abspath(__file__))


# 머신러닝 된 데이터가 몇개인지 확인하는 코드
pre_data = np.load(now_dir + "/up_trained.npz")
p_train = np.array(pre_data['train'])
p_train_labels = np.array(pre_data['train_labels'])

# 머신러닝 된 데이터가 몇개인지 확인하는 코드
new_data = np.load(now_dir + "/new_trained.npz")
n_train = np.array(new_data['train'])
n_train_labels = np.array(new_data['train_labels'])

 #기존데이터와 합치기
train = np.concatenate([p_train, n_train], axis=0)
train_labels = np.concatenate([p_train_labels, n_train_labels], axis=0)

np.savez(now_dir + "/up_trained.npz", train=train, train_labels=train_labels) #정답 데이터 저장
print(f'데이터 저장 완료')