# -*- coding=utf-8 -*-
import pickle
import numpy as np
import pandas as pd

# decode URL, HTML, HEX, Unicode
def decode(data):
    import urllib
    import html
    data = urllib.parse.unquote(data)
    data = html.unescape(data)
    return data

# generate words segmenting
def GeneSeg(data):
    import re
    import nltk
    data = data.lower()
    data, _ = re.subn(r'\d+', "0", data) # numbers -> 0
    data, _ = re.subn(r'(http|https)://[a-zA-Z0-9\.@/!&#\?]+', 
        "http://u", data) # urls -> http://u
    seg = r'''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    token = nltk.regexp_tokenize(data, seg)
    return token

# dictionary
def dictionary_set(data):
    from collections import Counter
    
    dictionary_size=3000
    words = [] # [token1, token2...]
    for line in data:
        words += line
    # build a dictionary with most common words
    count = [['WORD', -1]]
    counter = Counter(words)
    count.extend(counter.most_common(dictionary_size-1))
    dictionary = [word if num>1 else 0 for word,num in count]
    # transfer segment data to word in dictionary or 'WORD'
    datas = []
    for line in data:
        for i in range(len(line)):
            if line[i] not in dictionary:
                line[i] = 'WORD'
                count[0][1] += 1
        datas.append(line)

    return datas

# word2vec
def word2vec(datas):
    from gensim.models.word2vec import Word2Vec

    embedding_size=256
    skip_window=5
    num_sampled=64
    num_iter=5
    model=Word2Vec(datas,size=embedding_size,
        window=skip_window,negative=num_sampled,iter=num_iter)
    embeddings=model.wv
    return embeddings    

# extract word embeddings and dump into file
# return labels
def word_embed(filename,data,isxss):
    import time

    pk_file = filename[:-4]+'_text.pickle'
    start = time.time()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = decode(line)
            token = GeneSeg(line)
            data.append(token)
            y.append(isxss)
    datas = dictionary_set(data)
    embeddings = word2vec(datas)
    end = time.time()
    print('Finish preprocess to word embeddings in %4f.' %(end-start))

    with open(pk_file, 'wb') as pkf:
        pickle.dump(embeddings, pkf)
    print('model.wv has been dumped into ', pk_file)

    return y

if __name__ == "__main__":
    x = [] # script extract from file
    y = [] # lables
    good_example = 'xss_detector\\data\\good_sample.csv'
    xss_example = 'xss_detector\\data\\xss_sample.csv'
    f2l(good_example,x,0)
    f2l(xss_example,x,1)