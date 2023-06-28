from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords # stopwords는 불용어
import nltk

# stopwords 사전 다운로드 (따로 또 사용할것 같아서 def 안에 넣지않고 여기 def3위에 씀)
nltk.download('stopwords')
stop = stopwords.words('english')
porter = PorterStemmer()  # 개별 단어를 자연어 처리할 때 과거형, 현재형 등의 형태를 원형으로 동일하게 맞추는 작업을 한다. 

# 공백 기준 단어 분리
def tokenizer(text):
    return text.split()

# 단어의 원형 변환
def tokenizer_porter(text):
    word_list = tokenizer(text)
    #단어 원형
    word_list2 = [porter.stem(word) for word in word_list] #뒤부터 읽음 for wold list 개수만큼 루프 돌며 word 내벹음 어디다? stem(word)에 벹음. 그러면 porter.stem()에서 원형으로 바꿔줌
    return word_list2

# 공백 기준 단어 분리
def tokenizer(text):
    return text.split()

# 단어의 원형 변환
def tokenizer_porter(text):
    word_list = tokenizer(text)
    #단어 원형
    word_list2 = [porter.stem(word) for word in word_list] #뒤부터 읽음 for wold list 개수만큼 루프 돌며 word 내벹음 어디다? stem(word)에 벹음. 그러면 porter.stem()에서 원형으로 바꿔줌
    return word_list2

def step3_word_tokenizer():
    print(tokenizer_porter('Runners like running and thus they run')) # 이모티콘만 빼주고 .은 안떼주네ㅜ 같은 단어 run에 이제 숫자를 할당하여 숫자로 표현한다.
   

if __name__ == '__main__':
    step3_word_tokenizer()