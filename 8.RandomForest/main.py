# 학습용 데이터
from sklearn import datasets
# 학습, 테스트 분리
from sklearn.model_selection import train_test_split
# 데이터 표준화
from sklearn.preprocessing import StandardScaler
# RandomForest 알고리즘
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# 파일 저장
import pickle
import numpy as np 

def step1_get_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target[:]
    return X,y
    
def step2_learning():
    X, y = step1_get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    # n_estimators: 포레스트 내의 나무 개수
    # n_jobs: 병렬처리에서 사용할 코어 수
    # criterion: 불순도 측정 방식(entropy, gini)
    # max_depth: 트리의 최대 깊이
    ml = RandomForestClassifier(n_estimators=10, n_jobs=3, criterion='entropy', max_depth=3, random_state=0)
    ml.fit(X_train_std, y_train)
    X_test_std = sc.transform(X_test)
    y_pred = ml.predict(X_test_std)
    print('학습 정확도: ', accuracy_score(y_test, y_pred))
    with open('8.RandomForest/model.pkl', 'wb') as fp:
        pickle.dump(sc, fp)
        pickle.dump(ml, fp)

def step3_using():
    with open('8.RandomForest/model.pkl', 'rb') as fp:
        sc = pickle.load(fp)
        ml = pickle.load(fp)
    X = [[1.4, 0.2], [1.5, 0.3], [1.6, 0.4], 
         [4.5, 1.5], [4.1, 1.0], [4.5, 1.6],
         [5.2, 2.0], [5.4, 2.3], [5.1, 1.8]]
    X_std = sc.transform(X)
    y_pred = ml.predict(X_std)
    # print(len(y_pred))
    for v in y_pred:
        if v == 0:
            print('Iris-setosa')
        elif v == 1:
            print('Iris-versicolor')
        else:
            print('Iris-virginica')

if __name__ == '__main__':
    # step1_get_data()
    step2_learning()
    # step3_using()
