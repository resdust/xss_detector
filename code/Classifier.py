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
from pre import pre_features

def mlp(X_train, y_train, X_test, y_test):
    from sklearn.neural_network import MLPClassifier
    from sklearn import metrics 
    import time

    start = time.time()
    hidden = [128,128]
    model = MLPClassifier(batch_size=256, hidden_layer_sizes=hidden, 
    activation='logistic', solver='adam', random_state=0, max_iter=500)
    model.fit(X_train, y_train, )
    end = time.time()
    print('finish mlp training in %.4f second.' %(end-start))

    y_pre = model.predict(X_test)
    pre_features.do_metrics(y_test,y_pre)
    
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

def RF(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib

    start = time.time()
    clf = RandomForestClassifier(n_estimators=100, max_depth=None,
        min_samples_split=20, random_state=0)
    clf.fit( x_train, y_train)
    y_pre = clf.predict(x_test)
    end = time.time()
    print('Random Forest finished in %.4f' %(end-start))

    pre_features.do_metrics(y_test, y_pre)
    joblib.dump(clf, 'model/RF.pkl')

if __name__ == "__main__":
    x = [] # feature data vectors
    y = [] # labels
    # Read data from pickle
    pre_features.extract_feature('data/good_example.csv',x,y,0)
    pre_features.extract_feature('data/xss_example.csv',x,y,1)
    
    x = np.asarray(x,dtype=float)
    y = np.asarray(y,dtype=int)
    y.shape=-1,1
    #y = to_categorical(y, num_classes=2)
    y = y.ravel()

    X, testX, Y, testY = train_test_split(x,y, test_size=0.3, random_state=0)

    # normalization(X)
    # normalization(testX)

<<<<<<< HEAD:code/Classifier.py
    RF(X, Y, testX, testY)
    # mlp(X, Y, testX, testY)
    # do_DNN(X, Y, testX, testY)
=======
    mlp(X, Y, testX, testY)
    # do_DNN(X, Y, testX, testY)
>>>>>>> 6395847e9fbcf164606c547df188c72b60242144:code/xssExtractor.py
