#/user/bin/python
#author:wangkun
#time : 2020/4/26
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
from nltk import FreqDist
import seaborn as sns
import pandas as pd
import jieba
import matplotlib


rawtexts = []
texts=[]
def loadRaw(file):
    with open(file,encoding='gbk') as txt:
        while True:
            lines=txt.readline()
            if (lines.replace('\n', '') != ''):
                rawtexts.append(lines)
            if not lines:
                break
                pass

def filter():
    for i in range(len(rawtexts)):
        if (len(rawtexts[i])>10):
            texts.append(rawtexts[i])

for root, dirs, files in os.walk('/home/wangkun/sentence_extract/dataset/'):
    for i in range(len(files)):
        loadRaw(os.path.join(root,files[i]))
print(len(rawtexts))
filter()
print(len(texts))

def plot_textlen():
    dict={}
    len1=[]
    for text in texts:
        len1.append(len(text))

    plt.hist(len1, bins=100, histtype="stepfilled", alpha=.8)
    plt.title('The hist of passage length')
    plt.xlabel('passage length')
    plt.ylabel('个数')
    plt.savefig('passagelength')


def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = jieba.cut(all_words)
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(fdist.keys()), 'count': list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n=terms)
    plt.figure(figsize=(20, 5))
    ax = sns.barplot(data=d, x="word", y="count")
    ax.set(ylabel='Count')
    plt.show()
plot_textlen()

#freq_words(texts)
