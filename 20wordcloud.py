import matplotlib
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
  
font_name = font_manager.FontProperties(fname='C:/Windows/fonts/malgun.ttf').get_name()
matplotlib.rc('font', family=font_name)


# ModuleNotFoundError: No module named 'wordcloud'
# pip install w소문ordcloud


msg='''  
cherry 제리톰 cherry 토요일 leurto  adslfj 가을 cherry 토요일  sldj
일요일 sld 제리 jfldf 월요일 sky winter summer  adslfj
일요일 803 ldfjlqwewtry upu fdgjld
cherry 일요일 sld cherry jfldf 토요일 leurto  adslfj 
weoruwoeru  weoripti cherry
qazx 제리토 가을 koetiet 9734 234 토요일 톰 제리톰 adslfj cherry 월요일
bc  토요일 하늘 lsdjlfjp 월요일 eirp cherry weoripti
'''

  
print(msg)
file = open('./data/kakao.txt', 'r', encoding='utf-8')   
rfile=file.read()
print(rfile)
print() #복붙

spwords = set(STOPWORDS)
mywc = WordCloud(max_font_size=350, stopwords=spwords, 
               font_path='C:/Windows/fonts/malgun.ttf',
               background_color='black', width=800, height=800)

mywc.generate(msg)
# mywc.generate(rfile) #파일

plt.figure(figsize=(12,8))
plt.imshow(mywc)
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()
print()