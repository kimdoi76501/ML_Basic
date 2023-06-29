import pandas as pd
import numpy as np
# from ../2.Perceptron/Perceptron import Perceptron
from perceptron import Perceptron
import pickle

def step1_get_data():
    # iris 데이터 파일에서 읽어오기
    df = pd.read_csv('3.IrisPerceptron\iris.data', header=None) # ./iris.data : 현재 디렉토리의 iris데이터라는 뜻 # . 찍고 오류나면 위치가 NEW폴더라는 뜻이라서 3.IrisPerceptron 추가해주거나 점 제거
    # print(df.head())
    # 꽃잎 데이터 추출
    X = df.iloc[:100, [2, 3]].values
    # print(X)
    # 꽃종류 (종속데이터)
    y = df.iloc[:100, 4].values
    y = np.where(y == 'Iris-setosa', 1, -1)
    # print(y)
    return X, y

def step2_learning():
    ppn = Perceptron(eta=0.1)
    X, y = step1_get_data()
    # 학습
    ppn.fit(X, y)
    print(ppn.erros_)
    print(ppn.w_)
    # 학습된 모델을 저장
    with open('3.IrisPerceptron\perceptron.iris', 'wb') as fp:
        pickle.dump(ppn, fp) # dump: 메모리 있는 모양 그대로 저장하기에 save가 아님
    print('학습완료')

def step3_using():
    # 저장된 모델 불러오기
    with open('3.IrisPerceptron\perceptron.iris', 'rb') as fp:
        ppn = pickle.load(fp)
    while True:
        a1 = input('꽃잎의 길이를 입력하세요: ')
        a2 = input('꽃잎의 너비를 입력하세요: ')
        X = np.array([float(a1), float(a2)])
        # print(X)
        result = ppn.predict(X)
        if result == 1:
            print('Iris-setosa')
        else:
            print('Iris-versicolor')

if __name__ == '__main__':
    # step1_get_data()
    # step2_learning()
    step3_using()