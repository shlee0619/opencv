import matplotlib
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image 
  
font_name = font_manager.FontProperties(fname='C:/Windows/fonts/malgun.ttf').get_name()
rc('font', family=font_name)


file = open('./data/alice.txt', 'r', encoding='utf-8')   
rfile=file.read()
print(rfile)
print() #복붙

spwords = set(STOPWORDS)
spwords.update(['그','저', '니'])
spwords.add('said')
spwords.update(['hahaha','hohoho'])

mywc = WordCloud(max_font_size=350, stopwords=spwords, 
               font_path='C:/Windows/fonts/malgun.ttf',
               background_color='black', width=800, height=800)


mywc.generate(rfile) #파일

plt.figure(figsize=(12,8))
plt.imshow(mywc)
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()
print()

'''
이미지처리는 구글이미지 
import numpy as np
from PIL import Image 
'''

image_file = './data/alice.png'
img_file = Image.open(image_file)
alice_mask = np.array(img_file)
mywc = WordCloud(max_font_size=350, stopwords=spwords, 
               mask= alice_mask  ,
               font_path='C:/Windows/fonts/malgun.ttf',
               background_color='black', width=800, height=800)


mywc.generate(rfile) #파일
plt.figure(figsize=(12,8))
plt.imshow(mywc)
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()
print()
print()



