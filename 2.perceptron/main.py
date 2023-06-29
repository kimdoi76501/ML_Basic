import numpy as np
from perceptron import Perceptron # from perceptron.py에서 Perceptron class 호출
# from ..\1.Helloworld/perceptron import Perceptron # 1.Helloworld에 퍼셉트론 들어있을 경우
from time import time
import pickle

def step1_learning():
    # 학습과 테스트를 위해 사용할 데이터 정의
    X = np.array([[1,1], [1,0], [0,1], [0,0]])
    y = np.array([1, -1, -1, -1])
    # 퍼셉트론 객체 생성
    ppn = Perceptron(eta=0.1)
    # 학습
    s_time = time()
    ppn.fit(X, y)
    e_time = time()
    print('학습에 걸린 시간: ', (e_time - s_time))
    print('학습 중 오차가 난 수: ', ppn.erros_)
    # 학습된 모델 저장
    with open('perceptron.model', 'wb') as f: #2번 폴더에 저장하려면 2.Perceptron/perceptron.model
        pickle.dump(ppn, f)

def step2_using():
    # 저장된 모델 불러오기
    with open('perceptron.model', 'rb') as f:
        ppn = pickle.load(f) # 파일에서 읽어와서 메모리에 변수처럼 변환해서 올림 # 피클은 파이썬에서만 가능

    while True:
        a1 = input('첫번째 2진수를 입력하세요: ')
        a2 = input('두번째 2진수를 입력하세요: ')
        X = np.array([[int(a1), int(a2)]])
        result = ppn.predict(X)
        if result == 1:
            print('결과: 1')
        else:
            print('결과: 0')

if __name__ == '__main__':
    # p1 = Perceptron()
    # p1.test()
    # step1_learning()
    step2_using()