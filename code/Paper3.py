# advaned semi-supervisor algorithm
# feature extraction
# 
import numpy as np
import pandas as pd

evil_word = ['alert', 'script', 'onerror', 'confirm', 'img', 'onload', 'eval', 'prompt', 'src', 'href', 'javascript', 'window', 'fromcharcode', 'document', 'onmouseover', 'cookie', 'domain', 'onfocus', 'expression', 'iframe', 'onclick', 'location', 'this', 'createelement', 'String.fromCharCode', 'search', 'div', '<script', 'var', 'http', '.js']
evil_char = ["'", '"', '<', '>', '\\', ',', '+', '&', '%', '/', '?', '!', ';', '#', '=', '[', ']', '$', '(', ')', '^', '*', '-', '@', '_', ':', '{', '}', '~', '.', '|', ' ']
re_unique = ['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '\\', '|', '(', ')']

feature_names = ['URL_length', 'parameter_maxlen', 'digit_percentage', 'letter_percentage', 'parameter_number', 'XSS_count', 'evil_char_count']
word_num = len(evil_word)
char_num = len(evil_char)
for i in range(word_num):
    char = 'char_'+str(i)
    vars()[char] = 0
    feature_names.append(char)
for i in range(char_num):
    word = 'word_'+str(i)
    vars()[word] = 0
    feature_names.append(word)
feature_names.append('label')

def featureExtract(feature_file, df):
    import urllib
    from urllib import parse
    import html
    import re
    
    features = [] # write features row by row
    datas = df.values
    for data in datas:
        input = data[0]
        input = input.rstrip()
        input = urllib.parse.unquote(input)
        input = html.unescape(input) #decode
        
        label = data[1]
        feature = []

        # URL_length
        URL_length = len(input)

        # parameter_maxlen
        params = parse.parse_qs(input)
        values = [params[k] for k in params]
        parameter_maxlen = 0
        for v in values:
            l = len(v[0])
            if l > parameter_maxlen:
                parameter_maxlen = l

        # digit_percentage, letter_percentage
        countd = counta = 0
        for c in input:
            if c.isdigit():
                countd = countd + 1
            if c.isalpha():
                counta = counta + 1
        digit_percentage = '%.6f' %(countd/URL_length)
        letter_percentage = '%.6f' %(counta/URL_length)

        # parameter_number
        parameter_number = len(params)

        # XSS_count and word0~31
        num = 0
        XSS_count = 0
        for word in evil_word:
            result = re.findall(word, input)
            if result:
                vars()['word_'+str(num)] = len(result)
                num = num + 1
                XSS_count = XSS_count + 1
        
        # evil_char_count and char0~30
        num = 0
        evil_char_count = 0
        for char in evil_char:
            if char in re_unique:
                pattern = '[\\'+char+']'
            else:
                pattern = '['+char+']'
            result = re.findall(pattern, input)
            if result:
                vars()['char_'+str(num)] = len(result)
                num = num + 1
                evil_char_count = evil_char_count + 1

        for f in feature_names:
            feature.append(eval(f))

        features.append(feature)
    
    # write features into file 
    df_csv = pd.DataFrame(data=features, columns=feature_names)
    df_csv.to_csv(feature_file, index=None)
        
def readDatas(file):
    df = pd.read_csv(file, header=None)
    return df
    
if __name__ == '__main__':
    file = 'data\\test.csv'
    # file = 'data\\train.csv'
    feature_file = file.split('.')[0]+'Feature.csv'
    new_feature_file = feature_file.split('.')[0]+'_useful.csv'

    # df = readDatas(file)
    # featureExtract(feature_file, df)

    useful = [6,27,7,4,45,25,24,44,40,26,23,20,21,22,19]
    df_feature = pd.read_csv(feature_file,header=0)
    df_feature = df_feature.round({'digit_percentage':6,'letter_percentage':6})

    useful_feature_names = []
    for i in useful:
        useful_feature_names.append(feature_names[i-1])
    useful_feature_names.append('label')
    print('useful features:', useful_feature_names)

    drop_feature_names = list(set(feature_names)-set(useful_feature_names))

    df_feature = df_feature.drop(drop_feature_names,axis=1)
    df_feature.to_csv(new_feature_file,index=None)
