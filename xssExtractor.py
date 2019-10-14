# -*- coding:utf-8 -*-
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
# from sklearn import datasets
# from sklearn import svm
# from sklearn.externals import  joblib
# from sklearn.metrics import classification_report
from sklearn import metrics

import tflearn
import pickle
from os.path import exists

def get_len(url):
    return len(url)

def get_url_count(url):
    if re.search('(http://)|(https://)', url, re.IGNORECASE) :
        return 1
    else:
        return 0

def get_evil_char(url):
    return len(re.findall("[<>,\'\"/\\&]", url, re.IGNORECASE))

#Total count of URLs with maximum number of obfuscated characters
# def get_obfuscated_char(url):
#     obfuscatedCharCount = 0
#     special = re.split(r'%', url)
#     if (len(special)-1) > (len(url)/3):
#         obfuscatedCharCount += 1
#     return obfuscatedCharCount

def get_evil_word(url):
    XSS = ["iframe","response_write","prompt","javascript","document.cookie","alert","script=)(%3c","%3e","%20","onerror","onload","eval","src=","prompt"]
    XSS = '(' + ')|('.join(XSS) + ')'
    return len(re.findall( XSS, url, re.IGNORECASE))

def get_feature(url):
    return [get_len(url),get_evil_char(url),get_evil_word(url)]

def do_metrics(y_test,y_pred):
    # TP+TN/(TP+TN+FP+NP)
    print ("metrics.accuracy_score:", metrics.accuracy_score(y_test, y_pred))
    # [[TP, FN], [FP, TN]]
    print ("metrics.confusion_matrix:", metrics.confusion_matrix(y_test, y_pred))
    # TP/(TP+FP)
    print ("metrics.precision_score:", metrics.precision_score(y_test, y_pred))
    # TP/(TP+FN)
    print ("metrics.recall_score:", metrics.recall_score(y_test, y_pred))
    print ("metrics.f1_score:", metrics.f1_score(y_test,y_pred))

def etl(filename,data,y,isxss):
    xss = [1, 0] if isxss else [0, 1]
    pk_file = filename[:-3]+'pickle'
    if(exists(pk_file)):
        with open(pk_file, 'rb') as pkf:
            pkdata = pickle.load(pkf)
            length = len(pkdata)
            data.extend(pkdata)
            y.extend([xss for i in range(length)])
    else:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                f1=get_len(line)
                f2=get_url_count(line)
                f3=get_evil_char(line)
                f4=get_evil_word(line)
                data.append([f1,f2,f3,f4])
                y.append(isxss)
        with open(pk_file, 'wb') as pkf:
            pickle.dump(data, pkf)
        
    return data

def normalization(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x -= mean
    x /= std

def do_DNN(X, Y, testX, testY):
    input_layer = tflearn.input_data(shape=[None, 4])
    dense1 = tflearn.fully_connected(input_layer, 32, activation='tanh',
        regularizer='L2', weight_decay=0.0001)
    dropout1 = tflearn.dropout(dense1, 0.6)
    dense2 = tflearn.fully_connected(dropout1, 32, activation='tanh',
        regularizer='L2', weight_decay=0.0001)
    dropout2 = tflearn.dropout(dense2, 0.6)
    softmax = tflearn.fully_connected(dropout2, 2, activation='softmax')

    net = tflearn.regression(softmax, optimizer='adam', loss='binary_crossentropy')
    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='log')
    model.fit(X, Y, n_epoch=2, validation_set=(testX, testY),
              show_metric=True, run_id="xss_dense_model")
              
    predY = model.predict(testX)
    do_metrics(testY, predY)
    print(model.evaluate(testX, testY))

if __name__ == "__main__":
    x = [] # feature data vectors
    y = [] # labels
    # Read data from pickle
    etl('data/good_example.csv',x,y,1)
    etl('data/xss_example.csv',x,y,0)

    # X, Y, testX, testY = mnist.load_data(one_hot=True)
    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=int)
    y.shape = -1,2

    X, testX, Y, testY = train_test_split(x,y, test_size=0.3, random_state=0)

    normalization(X)
    normalization(testX)

    do_DNN(X, Y, testX, testY)