#!/usr/bin/python
# -*- coding: <utf-8> -*-
#python中存放着两种字符串：1.文本；2.字节码。
#python3中文本字符串被命名为str,字节字符串类型命名为bytes
#实例化一个字符串会得到一个str对象，如果得到bytes，则在str前面加上b,或者encode
#str-->encode()-->bytes,,,bytes-->decode-->str

from gensim import corpora, models
import jieba.posseg as jp, jieba
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
import string

def preprocess(file):
    teacher_text=dict()
    with open(file,encoding='gbk') as txt:
        while True:
            lines=txt.readline()
            if(lines.replace('\n','')!=''):
                lines=lines.replace('\n','')
                if lines[0].isdigit() and lines[1].isdigit():
                    teacher_text.setdefault(lines[:2], []).append(lines[2:].replace("：",'').replace(":",""))
                elif lines[0].isdigit():
                    teacher_text.setdefault(lines[0],[]).append(lines[1:].replace("：",'').replace(":",""))
            if not lines:
                break
                pass
    texts=[]
    for key in teacher_text:

        for item in teacher_text[key]:
            print('#' * 100)
            print(item)
            texts.append(item)
    str="".join(texts)
    return str

def getFilelist(path):
    filelist=[]
    files=os.listdir(path)
    for f in files :
        if(f[0] == '.') :
            pass
        else :
            filelist.append(f)
    return filelist,path

def fenci(filename, path):
    # 保存分词结果的目录
    sFilePath = 'segfile'
    if not os.path.exists(sFilePath):
        os.mkdir(sFilePath)
    # 读取文档
    filepath = path +'/'+ filename
    file_list_str = preprocess(filepath)


    flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
    stopwords = []
    with open('stopwords.txt', encoding='utf-8') as txt:
        for line in txt:
            stopwords.append(line.strip())

    seg_list = [w.word for w in jp.cut(file_list_str) if w.flag in flags and w.word not in stopwords]
    # 将分词后的结果用空格隔开，保存至本地。比如"我来到北京清华大学"，分词结果写入为："我 来到 北京 清华大学"
    f = open(sFilePath + "/" + filename + "-seg.txt", "w+")
    f.write(' '.join(seg_list))
    f.close()


def Tfidf(filelist):
    path = 'segfile'
    corpus = []  # 存取100份文档的分词结果
    for ff in filelist:
        fname = path + '/'+ff+'-seg.txt'
        f = open(fname, 'r+')
        content = f.read()
        f.close()
        corpus.append(content)

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    word = vectorizer.get_feature_names()  # 所有文本的关键字
    weight = tfidf.toarray()  # 对应的tfidf矩阵

    sFilePath = 'tfidffile'
    if not os.path.exists(sFilePath):
        os.mkdir(sFilePath)

    # 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
    for i in range(len(weight)):
        print (u"--------Writing all the tf-idf in the", i, u" file into ", sFilePath + '/' + str(i).zfill(5) + '.txt', "--------")
        f = open(sFilePath + '/' + str(i).zfill(5) + '.txt', 'w+')
        for j in range(len(word)):
            f.write(word[j] + "    " + str(weight[i][j]) + "\n")
        f.close()


if __name__ == "__main__" :
    (allfile,path) = getFilelist('dataset')
    for ff in allfile:
        print('process ',ff)
        print_('/home/wangkun/sentence_extract/dataset/'+ff)
    # for ff in allfile :
    #     print ("Using jieba on "+ff)
    #     fenci(ff,path)
    # Tfidf(allfile)
# # 分词过滤条件
# #jieba.add_word('四强', 9, 'n')
# flags = ('n', 'nr', 'ns', 'nt', 'eng', 'v', 'd')  # 词性
# stopwords=[]
#
# with open ('stopwords.txt',encoding='utf-8') as txt:
#     for line in txt:
#         stopwords.append(line.strip())
# print(stopwords)
# stopwords = ('没', '就', '知道', '是', '才', '听听', '坦言', '全面', '越来越', '评价', '放弃', '人')  # 停词
# #分词
# words_ls = []
# for text in texts:
#     words = [w.word for w in jp.cut(text) if w.flag in flags and w.word not in stopwords]
#     words_ls.append(words)
# print(words_ls)
# # 构造词典
# dictionary = corpora.Dictionary(words_ls)
# # 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
# corpus = [dictionary.doc2bow(words) for words in words_ls]
# # lda模型，num_topics设置主题的个数
# lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
# # 打印所有主题，每个主题显示5个词
# for topic in lda.print_topics(num_words=5):
#     print(topic)
# # 主题推断
# print(lda.inference(corpus))
