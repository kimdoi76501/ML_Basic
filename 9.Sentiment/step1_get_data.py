import pandas as pd
import numpy as np
import os
import codecs
import re

def step1_get_data():
    # 데이터 파일 위치
    path = '9.Sentiment/aclImdb/' # 코드 맨 끝에 /가 있는지 확인하기
    # 긍정 부정값
    labels = {'pos': 1, 'neg': 0}
    df = pd.DataFrame()
    # 디렉토리 수 만큼 반복  (test, train 파일안에 pos와 neg 만큼 반복할것)
    for s in ('test', 'train'):
        for name in ('pos', 'neg'): # 이렇게 만들면 test 안의 pos, test안의 neg, train 안의 pos, train 안의  이런 식으로 for 문 돌아감
            subpath = '%s/%s' % (s, name)
            # print(subpath)
            # print(path + subpath) # 경로 확인
            # 현재 디렉토리의 파일 목록
            file_list = os.listdir(path + subpath)
            # print(file_list)  이제 실제파일 이름들 읽어서 데이터에 집어넣기
            for file in file_list:
                # print(path + subpath + '/' + file)  #print(path + subpath + '\\' + file) / 또는 \\써야함
                # print(os.path.join(path+subpath, file))  # 이렇게 쓰면 슬래시 /를 역슬래시 \\ 쓰든 무관함
                file_name = os.path.join(path+subpath, file)
                with codecs.open(file_name, 'r', 'utf-8') as fp:
                    txt = fp.read()
                    # print(labels[name], txt) #pos인지 neg인지도 알 수있게 출력함
                df = df.append([[txt, labels[name]]], ignore_index=True)
    # 3중 포문에 맞추기
    df.columns = ['review', 'sentiment']
    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index)) # index를 랜덤하게 섞어 재정렬
    # 저장
    df.to_csv('9.Sentiment/aclImdb/movie_review.csv', index=False) # 한 곳에 모아 csv로 저장(회사에서는 db나 hadoop 저장)


if __name__ == '__main__':
    step1_get_data()
