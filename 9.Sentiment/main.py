import pandas as pd
import numpy as np
import os
import codecs
import re
import time

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
# stopwords 사전 다운로드
# nltk.download('stopwords')
# stop = stopwords.words('english')
# porter = PorterStemmer()



# 여기까지 한 전처리는 1차 전처리다. 2차 전처리는 실제 모델학습에 필요한 전처리를 하게 됨
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline # 여러가지 작업을 할 때 묶어서 한번에 편하게 해주는
# 아래는 전체문장을 정형데이터로 바꿈
from sklearn.feature_extraction.text import CountVectorizer # 숫자로바꿔줌
from sklearn.feature_extraction.text import TfidfVectorizer # 역빈도분석
import pickle

from step1_get_data import step1_get_data  # from 어디저장했나 import 무슨클래스,무슨함수 쓸지
from step2_preprocessing import step2_preprocessing
from step3_word_tokenizer import tokenizer, tokenizer_porter, step3_word_tokenizer
from step4_learning import step4_learning


def step5_using():
    with open('9.Sentiment/model.pkl', 'rb') as fp:
        Pipeline = pickle.load(fp) #불러와서 다시 복원? 이게 왜 복원이지
    while True:
        text = input('영문 리뷰를 입력하세요: ')
        y = Pipeline.predict([text])        # []는 차원 맞춰주려고 넣음
        rate = Pipeline.predict_proba([text]) * 100 # predict_proba 예측에 대한 확률값(예측 맞을 확률?)
        # print('y: ', y)
        # print('rate: ', rate)
        rate = np.max(rate)
        if y == 1:
            print('positive reivew')
        else:
            print('negative review')
        print('정확도: ', round(rate,))



if __name__ == '__main__':
    # step1_get_data()
    # step2_preprocessing() #전처리
    # step3_word_tokenizer()
    # step4_learning()
    step5_using()

 # (리팩토링: 파일로 빼서 임포트 해서 step1~4 함수 호출해서 써보기)