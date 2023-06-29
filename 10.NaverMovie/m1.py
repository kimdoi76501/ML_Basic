import requests
from bs4 import BeautifulSoup
from time import sleep
import pandas as pd
import re


def preprocesspr(text):
    return text

def is_float(n):
    try:
        float(n) # 숫자 있으면 리턴(?) 빈문자 또는 문자있으면 플롯함수 에러
    except ValueError:
        return 0
    else:
        return float(n)

def step1_get_data():
    # 영화 코드 목록 만들기
    site1 = 'https://pedia.watcha.com/ko-KR/?domain=movie'
    # root > div > div.css-1xm32e0 > section > div > section > div:nth-child(6) > div.css-5rbrg6 > div > div.css-9dnzub > div > div > ul > li:nth-child(1)
    # //*[@id="root"]/div/div[1]/section/div/section/div[6]/div[2]/div/div[1]/div/div/ul
    # css-a19lp3-VisualUl
    res1 = requests.get(site1)
    code_movie_list = list()
    name_movie_list = list()
    if res1.status_code == requests.codes.ok:
        bs1 = BeautifulSoup(res1.text, 'html.parser') #빨리 하려면 (실무에서 쓴다든지) lxml 사용
        ul = bs1.find_all(class_ = re.compile('-VisualUl'))[2]
        # print(ul)
        lis = ul.find_all('li')
        # li_a = lis[0].find('a')
        # print(li_a.get('href').split('/')[-1], li_a.get('title'))
        # print(ls_a)    
        for li in lis:
            # print(li.find('a').get('href').split('/')[-1])
            # print(li.find('a').get('title'))
            code_movie_list.append(li.find('a').get('href').split('/')[-1])
            name_movie_list.append(li.find('a').get('title'))
        
    # 영화 코드별 리뷰 가져오기
    temp = zip(code_movie_list, name_movie_list)
    df = pd.DataFrame()
    # 최초 저장 여부 상태값
    check_save = False # True면 두번째부터 저장함
    count = 0
    for code, name in temp:
        sleep(0.5) # 잠시 멈췄다 실행 # requests.get()있으면 이거 써주기 # 특히 셀레니움 써줄때도 # 아니면 차단빵 먹을수도
        site2 = 'https://pedia.watcha.com/ko-KR/contents/%s/comments' % code
        print(name)
        df = pd.DataFrame()
        res2 = requests.get(site2)
        if res2.status_code == requests.codes.ok: # HTTP 정상적이면 200 URL 없으면 400 서버에러면 500 등.. # OK되면 text 안의 페이지소스 값
            bs2 = BeautifulSoup(res2.text, 'html.parser') # 몇 번째 컬럼 불러와라 # 태그에 있는 거를 어떻게 해석할거냐
            # -CommentLists
            div1 = bs2.find(class_ = re.compile('-CommentLists')) # 하나밖에 없어서 find_all 말고 find
            # css-bawlbm
            # ul = div1.find('ul') # 목록에 대한 div도 있지만 부가정보도 있어서 11개 나옴
            div2s = div1.find_all(class_ = 'css-bawlbm') # class 명 지정해서 존재하는 8개만 출력
            # print(div2[0].get_text()) # tag 빼고 text만 출력
            for div2 in div2s:
                div3 = div2.find_all('div')
                # print('별점: ', div3[5].get_text())
                # print('리뷰: ', div3[6].get_text())
                # print('좋아요: ', div3[8].find('em').get_text())
                star = div3[5].get_text()
                review = div3[6].get_text()
                review = review.replace(',', '')
                # good = div3[8].find('em').get_text()
                df = df.append([[name, code, star, review]], ignore_index=True)
    # 저장 
    if check_save == False: # 첫 번째 저장
        df.columns = ['name', 'code', 'star', 'review']
        df.to_csv('movie_data.csv', encoding='utf-8', index=False)
        check_save = True
    else: # 두 번째 이후 저장
        df.to_csv('movie_data.csv', encoding='utf-8', index=False, mode = 'a', header=False) #모드:어펜드
    count += 1
    print('진행 중: ', count)


if __name__ == '__main__':
    step1_get_data()