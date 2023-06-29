# import numpy as np

# class Perceptron:
#     # 생성자 함수 # 생성자 지정 안 하면 알아서 생성됨
#     def __init__(self):
#         print('Perceptron created') # 실행할 때마다 출력됨

# if __name__ == '__main__': #if__name__: 내가 나를 실행시킬 때
#     p1 = Perceptron()
#     print(type(p1))
# # p1: 퍼셉트론 타입


################################################


# import numpy as np

# class Perceptron:
#     # thresholds: 임계값, 계산된 예측값을 비교하는 값
#     # eta: 학습률
#     # n_iter: 학습 횟수
#     a = ''
#     def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
#         self.thresholds = thresholds
#         self.eta = eta
#         self.n_iter = n_iter

#     def test(self):
#         print(self.thresholds)

# if __name__ == '__main__':
#     p1 = Perceptron(thresholds=2)
#     print(p1.test())


################################################

# import numpy as np

# class Perceptron:
#     a = ''
#     def __init__(self, thresholds=0.0, eta=0.01, n_iter=10): 
#         self.thresholds = thresholds
#         self.eta = eta
#         self.n_iter = n_iter

#     # 학습 함수
#     # X : 입력 데이터, 독립변수, 특징, 퓨처, 설명변수
#     # y : 결과 데이터, 정답, 종속변수, 라벨, 클래스
#     def fit(self, X, y):
#         # 가중치를 담을 행렬 행성
#         self.w_ = np.zeros(1 + X.shape[1])
#         # 예측값과 실제값을 비교하여 다른 예측값의 개수를 담음
#         self.erros_ = []
#         # 지정된 학습 횟수만큼 반복
#         for _ in range(self.n_iter): # for _ : 루프만 돌 때
#             pass

#     def test(self):
#         print(self.thresholds)

# if __name__ == '__main__': # __붙으면 외부(class)에서 쓰지 마라
#     p1 = Perceptron(thresholds=2)
#     print(p1.test())

################################################

import numpy as np

class Perceptron:
    a = ''
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10): 
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # 가중치를 담을 행렬 행성
        self.w_ = np.zeros(1 + X.shape[1]) # 컬럼 개수보다 1 많음: 바이어스를 같이 넣기 위해서
        self.erros_ = []
        for _ in range(self.n_iter): #1,-1,-1,-1이 10개씩이라 총 40번 루프
            # 예측값과 실제값이 다른 개수를 담을 변수
            errors = 0
            # 입력 데이터와 결과 데이터를 묶어줌
            temp1 = zip(X, y) 
            for xi, target in temp1: # 알아서 인덱스 처리해서 하나씩 넘어옴
                # 예측
                a1 = self.predict(xi)
                # 예측값과 실제값이 다르면 가중치를 업데이트
                if target != a1:
                    update = self.eta * (target - a1)
                    # 가중치(기울기)
                    self.w_[1:] += update * xi
                    # 절편(바이어스)
                    self.w_[0] += update
                    errors += int(update != 0.0) #로그 찍으려고 #예측이랑은 상관없음
            self.erros_.append(errors)
            print(self.w_)

    def predict(self, X):
        # step function
        a2 = np.where(self.net_input(X) > self.thresholds, 1, -1) # ?는 스칼라 변수에만 가능 # X는 벡터라서 벡터만큼 for문 써야하는데 np.where하면 개수만큼 알아서 돌아감 # 스텝펑션 # net_input:ax+b
        # for i in X:
        #     if i > self.thresholds:
        #         a2 = 1
        #     else:
        #         a2 = -1 이만큼을 한 줄로 줄인 게 np.where
        return a2 # 예측된값

    def net_input(self, X):
        a3 = np.dot(X, self.w_[1:]) + self.w_[0] # dot : 행렬 곱 
        return a3

    def test(self):
        print(self.thresholds)

if __name__ == '__main__':
    p1 = Perceptron()
    print('test ok', p1.test())

