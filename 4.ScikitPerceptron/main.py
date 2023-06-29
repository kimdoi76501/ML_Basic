# 퍼셉트론 알고리즘을 이용해 꽃받침 길이와 너비, 꽃잎의 길이와 너비를 이용해 아이리스 품종을 구분할 수 있는 머신러닝 수행
# 퍼셉트론은 바이너리 결과를 가지므로 3개의 품종을 동시에 구분할 수 없다
# 따라서 setosa와 versicolor의 2개 품종 구분할 수 있게 학습
# 0629 df 선언한걸 함수안에


# 학습용 데이터
from sklearn import datasets
# 학습, 테스트 분리
from sklearn.model_selection import train_test_split
# 데이터 표준화
from sklearn.preprocessing import StandardScaler
# 퍼셉트론 알고리즘
from sklearn.linear_model import Perceptron
# 정확도 계산 함수
from sklearn.metrics import accuracy_score
# 파일 저장
import pickle
import numpy as np

names = None
def step1_get_data():
    # 데이터 가져오기
    iris = datasets.load_iris()
    print(iris)
    X = iris.data[:100, [2, 3]]  #iris['data']
    y = iris.target[:100]
    names = iris.target_names[:100]
    # print(X)
    # print(y)
    # print(names)
    return X, y

def step2_learning():
    X, y = step1_get_data()
    # 학습, 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    # 데이터 표준화 작업
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    # 학습
    ml = Perceptron(eta0=0.01, max_iter=10, random_state=0)
    ml.fit(X_train_std, y_train)

    # 테스트 데이터 정확도 확인
    X_test_std = sc.transform(X_test)
    y_pred = ml.predict(X_test_std)
    print('학습 정확도: ', accuracy_score(y_test, y_pred))
    with open('4.ScikitPerceptron/iris.ml', 'wb') as fp:
        pickle.dump(sc, fp)
        pickle.dump(ml, fp)
    # print('학습완료')

def step3_using():
    with open('4.ScikitPerceptron\iris.ml', 'rb') as fp:
        sc = pickle.load(fp)
        ml = pickle.load(fp)
    while True:
        a1 = input('꽃잎의 길이를 입력하세요: ')
        a2 = input('꽃잎의 너비를 입력하세요: ')
        X = np.array([[float(a1), float(a2)]])
        X_std = sc.transform(X)
        y = ml.predict(X_std)
        # print(y)
        if y[0] == 0:
            print('Iris-setosa')
        else:
            print('Iris-versicolor')

        
if __name__ == '__main__':
    step1_get_data()
    # step2_learning()
    # step3_using()