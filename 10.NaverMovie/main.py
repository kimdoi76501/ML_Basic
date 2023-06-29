import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import re
from m1 import step1_get_data
import numpy as np
from sklearn.model_selection import train_test_split

# 별점 처리 함수
def star_preprocessing(text):
    value = float(text)
    if value > 5: # 왓챠면 2.5 네이버면 5
        return '1'
    else:
        return '0'

def review_preprocessing(text):
    if text.startswith('관람객'):
        new_str = text[3:]
        return new_str
    else:
        return text

def step2_preprocessing():
    df = pd.read_csv('10.NaverMovie/naver_star.csv', encoding='utf-8')
    # df = df.replace('\n관람객', '\n')
    # 랜덤하게 섞기
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))
    # 전처리
    df['star'] = df['star'].apply(star_preprocessing)
    df['review'] = df['review'].apply(review_preprocessing)
    review_list = df['review'].tolist()
    star_list = df['star'].tolist()
    X_train, X_test, y_train, y_test = train_test_split(review_list, star_list, test_size=0.2, random_state=0)
    print(len(X_train), len(X_test), len(y_train), len(y_test))

    # 학습, 테스트 데이터 저장
    dic_train = {'review': X_train, 'star': y_train}
    df_train = pd.DataFrame(dic_train)
    dic_test = {'review': X_test, 'star': y_test}
    df_test = pd.DataFrame(dic_test)
    df_train.to_csv('naver_train.csv', encoding='utf-8-sig', index=False)
    df_test.to_csv('naver_test.csv', encoding='utf-8-sig', index=False)

from konlpy.tag import Okt
def step3_tokenizer():
    pass
        

if __name__ == '__main__':
    # step1_get_data()
    # step2_preprocessing()
    step3_tokenizer()
