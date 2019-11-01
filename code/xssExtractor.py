# -*- coding:utf-8 -*-
import re
import numpy as np
import pickle
import time
from os.path import exists
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.utils import to_categorical

# decode URL, HTML, HEX, Unicode
def decode(data):
    import urllib
    import html
    data = data.rstrip()
    data = urllib.parse.unquote(data)
    data = html.unescape(data)
    return data

def get_len(url):
    return len(url)

def get_url_count(url):
    if re.search('(http)|(https)', url, re.IGNORECASE) :
        return 1
    else:
        return 0

def get_evil_char(url):
    return len(re.findall("[`~!@#$%^*()_-+=<>:\"{}| ,./;\'\\[]·]", url, re.IGNORECASE))

def get_evil_word(url):
    XSS = ["window","location","this","iframe","response_write","prompt","javascript",
    "document","cookie","alert","script","onerror","onload","eval","createelement",
    "string","search","cmd","div","img","<script","href","src","var","prompt",".js"]
    XSS = '(' + ')|('.join(XSS) + ')'
    return len(re.findall( XSS, url, re.IGNORECASE))

def get_feature(url):
    return [get_len(url),get_evil_char(url),get_evil_word(url)]

def do_metrics(y_test,y_pred):
    from sklearn import metrics

    # TP+TN/(TP+TN+FP+NP)
    print ("metrics.accuracy_score:", metrics.accuracy_score(y_test, y_pred))
    # [[TP, FN], [FP, TN]]
   # print ("metrics.confusion_matrix:", metrics.confusion_matrix(y_test, y_pred))
    # TP/(TP+FP)
    print ("metrics.precision_score:", metrics.precision_score(y_test, y_pred))
    # TP/(TP+FN)
    print ("metrics.recall_score:", metrics.recall_score(y_test, y_pred))
    print ("metrics.f1_score:", metrics.f1_score(y_test,y_pred))

def etl(filename,data,y,isxss):
    # xss = [1, 0] if isxss else [0, 1]
    pk_file = filename[:-3]+'pickle'
    if(exists(pk_file)):
        with open(pk_file, 'rb') as pkf:
            pkdata = pickle.load(pkf)
            length = len(pkdata)
            data.extend(pkdata)
            y.extend([isxss for i in range(length)])
    else:
        with open(filename, encoding="utf-8") as f:
            for line in f:
                line = decode(line)
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

def mlp(X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics 

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    start = time.time()
    hidden = [128,128]
    model = MLPClassifier(batch_size=256, hidden_layer_sizes=hidden, 
    activation='logistic', solver='adam', random_state=0, max_iter=500)
    model.fit(X_train, y_train, )
    end = time.time()
    print('finish mlp training in %.4f second.' %(end-start))

    y_pre = model.predict(X_test)
    do_metrics(y_test,y_pre)
    
def do_DNN(X, Y, testX, testY):
    import tflearn

    NUMBER_OF_CLASSES = len(Y[0])
    _, NUMBER_OF_FEATURES = X.shape

    net = tflearn.input_data(shape=[None, NUMBER_OF_FEATURES])
    net = tflearn.fully_connected(net, 32, activation='relu', 
    regularizer='L2', weight_decay=0.001)
    net = tflearn.dropout(net, 0.6)
    net = tflearn.fully_connected(net, 32, activation='relu',
    regularizer='L2', weight_decay=0.001)   
    net = tflearn.dropout(net, 0.6)
    net = tflearn.fully_connected(net, NUMBER_OF_CLASSES, activation='sigmoid')
    net = tflearn.regression(net, learning_rate=0.005,loss='categorical_crossentropy')

    # net = tflearn.input_data(shape=[None, NUMBER_OF_FEATURES])
    # net = tflearn.fully_connected(net, 32, activation='tanh',
    #     regularizer='L2', weight_decay=0.0001)
    # net = tflearn.dropout(net, 0.6)
    # net = tflearn.fully_connected(net, 32, activation='tanh',
    #     regularizer='L2', weight_decay=0.0001)
    # net = tflearn.dropout(net, 0.6)
    # net = tflearn.fully_connected(net, NUMBER_OF_CLASSES, activation='softmax')
    # net = tflearn.regression(net, optimizer='adam', loss='binary_crossentropy')
   
    model = tflearn.DNN(net, tensorboard_verbose=0, tensorboard_dir='log')
    
    # Training
    model.fit(X, Y, n_epoch=2, validation_set=(testX, testY),
              show_metric=True, batch_size=64, run_id="xss_normal_model")
              
    # do_metrics(testY, predY)
    acc = model.evaluate(testX, testY)
    print("Test accuracy : ", acc)

if __name__ == "__main__":
    x = [] # feature data vectors
    y = [] # labels
    # Read data from pickle
    etl('data/good_example.csv',x,y,0)
    etl('data/xss_example.csv',x,y,1)

    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=int)
    y.shape=-1,1
    #y = to_categorical(y, num_classes=2)


    X, testX, Y, testY = train_test_split(x,y, test_size=0.3, random_state=0)

    # normalization(X)
    # normalization(testX)

    mlp(X, Y, testX, testY)
    # do_DNN(X, Y, testX, testY)