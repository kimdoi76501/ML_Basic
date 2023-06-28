
def preprocessor(text):
    # HTML 태그 삭제
    text = re.sub('<[^>]*>', '', text) # < 아무문자 >
    # 이모티콘 삭제
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)|\^.?\^', text) # :-) ^.^ 같은 표정 삭제
    text = re.sub('[\W]+', ' ', text.lower() + ' '.join(emoticons).replace('-', '')) # 글자만, 공백, 소문자로+이모티콘변환
    return text

def step2_preprocessing(): #전처리
    # print(preprocessor('<br />hahahaha :-( ;-)ADFSDFdfdfdfd')) # 위에서 만든 함수로 html태그 잘 삭제가 되는지 확인해보기
    df = pd.read_csv('9.Sentiment/aclImdb/movie_review.csv')
    # print(df.shape)  # 유실된 데이터가있는지 5만건 맞나 확인하기
    df['review'] = df['review'].apply(preprocessor) #.apply하면 각 셀별로 적용 apply(preprocessor()) 판다스에게 함수 넘겨줄 때()쓰지 않음. 판다스에서 실행되어야하는데 ()쓰면 여기서 실행되지 판다스에 넘겨서 실행되지 않음. 괄호를 쓰면 apply(여기서) 실행한단 뜻
    df.to_csv('9.Sentiment/aclImdb/step2_movie_review.csv', index=False)


if __name__ == '__main__':
    step2_preprocessing()