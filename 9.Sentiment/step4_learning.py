import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline # 여러가지 작업을 할 때 묶어서 한번에 편하게 해주는
# 아래는 전체문장을 정형데이터로 바꿈
from sklearn.feature_extraction.text import CountVectorizer # 숫자로바꿔줌
from sklearn.feature_extraction.text import TfidfVectorizer # 역빈도분석
import pickle
from step3_word_tokenizer import tokenizer, tokenizer_porter, step3_word_tokenizer

def step4_learning(): #인터프리터 언어라서 이걸 위에 step3 아래 쓰면안됨. 안그러면 이 위에 import 한 sklearn 6개가 step4에서 작동이 안됨(코딩할 때 상호참조를 없도록 코딩해야 좋다)
    df = pd.read_csv('9.Sentiment/aclImdb/step2_movie_review.csv')
    # print(df.head())
    # 학습, 테스트 데이터 분리 (test split 써도 됨 같은 원리임)
    X_train = df.loc[:35000 -1, 'review'].values #index가 깔끔해서 loc사용 가능, 0부터 시작하기 때문에 -1 해준다.
    y_train = df.loc[:35000 -1, 'sentiment'].values
    X_test = df.loc[35000:, 'review'].values
    y_test = df.loc[35000:, 'sentiment'].values
    # print(type(X_train), X_train.shape, y_train.shape) # 반드시 단계 단계별로 확인해보는 습관을 첨부터 들이는게 아주 중요함.
    # print(type(X_test), X_train.shape, y_train.shape) # 이렇게 확인하는 코드를 안찍어버릇 하면 디버깅 할 때 logic이 어디에서 꼬인지 알기 어려움.
    # 단어장 만들어주는 객체 생성
    tfidf = TfidfVectorizer(lowercase=False, tokenizer=tokenizer_porter, token_pattern=None) # 우린 이미 소문자로 바꿔둬서 소문자 false 한다 안그럼 연산만 증가
    # ml 객체 생성
    logistic = LogisticRegression(C=10.0, penalty='l2', random_state=0) #penalty는 l1정규(거리를 절대로구함-아웃라이어에 덜 민감하나 기울기가 0일때 계산이 안됨) 또는 l2정규(거리를 제곱해서 구함-아웃라이어에 민감.그러나 어느 경우나 기울기를 구할 수 있어서 안정적임) 사용하면 됨. elasticent는 둘을 짬뽕함 아래쪽은 제곱하고 윗쪽은 직선 다만…성능 좋게 다 좋은게 아님. 연산이 많아지기때문. 따라서 데이터 상태나 구조를 알면 적절한걸 골라쓰고 모르면 연산많고 기능 좋은거 쓰셈
    # pipeline 객체 생성 (바로 fit 안쓰고 파이프라인 생성한다?? 무슨 뜻이람..ㅜ8.randomforest sc.fit 해서 ml.fit 했던것 참고..5-6줄짜리를 파이프라인 한 줄로 처리한 것임)
    pipeline = Pipeline([('tfidf', tfidf), ('logistic', logistic)]) # 파이프라인이 알아서 tfidf 실행후 logistic 을 순서대로 실행(묶어서 실행). tfidf 괄호안 실행 결과를 logistic 괄호에 넘겨서 실행
    # 학습
    stime = time.time()
    print('학습시작')
    pipeline.fit(X_train, y_train)
    print('학습완료')
    print('학습시간 : %.d' % (time.time() - stime))

    # 테스트
    y_pred = pipeline.predict(X_test)
    print('정확도 : %.2f' % accuracy_score(y_test, y_pred))

    # 모델 저장
    with open('9.Sentiment/model.pkl', 'wb') as fp:
        pickle.dump(pipeline, fp)
    print('저장완료')

if __name__ == '__main__':
    # step1_get_data()
    # step2_preprocessing() #전처리
    # step3_word_tokenizer()
    step4_learning()