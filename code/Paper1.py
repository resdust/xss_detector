import re
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
   
f = {}
f['punctuation'] = "&%/+'!;#=,-<>@_:~ \""
f['respecial'] = [r'\\',r'\^',r'\[',r'\]',r'\(',r'\)',r'\{',r'\}',
r'\?',r'\*',r'\$',r'\.',r'\|']
f['cominations'] = ['><','\'\"><','[]','==','&#']
f['objects'] = ['Document','window','iframe','location','This']
f['events'] = ['createelement','String.fromCharCode','Search']
f['tags'] = ['DIV','IMG','<script']
f['attributes'] = ['SRC','Href','Cookie']
f['reserve'] = ['Var']
f['functions'] = ['eval()']
f['protocol'] = ['HTTP']
f['externel_file'] = ['.js']

def plot(x, data, label):
    plt.plot([i+1 for i in range(5)], data['test_precision_macro'], c='red', label='test precision')
    plt.plot([i+1 for i in range(5)], data['test_recall_macro'], c='orange', label='test recall')
    plt.plot([i+1 for i in range(5)], data['train_precision_macro'], c='blue', label='train precision')
    plt.plot([i+1 for i in range(x)], data['train_recall_macro'], c='green', label='train recall')
    plt.xlabel('Time')
    plt.ylabel('Score')
    plt.title(label)
    plt.legend(loc='best')

    # plt.show()
    plt.savefig('log/Shuffle Paper1 '+label+'.png',bbox_inches='tight')
    plt.cla()

def process(filename, x, y, isxss):
    with open(filename, encoding='utf-8') as file:
        for line in file:
            vector = []
            for k in f:
                for rule in f[k]:
                    if(re.findall('['+rule+']', line, re.IGNORECASE) == []):
                        vector.append(0)
                    else:
                        vector.append(1)
            x.append(vector)
            y.append(isxss)
    
def SVMLinear(x, y):
    from sklearn import svm

    start = time.time()
    clf = svm.SVC(kernel='linear', C=7)
    # scoring = ['precision_macro', 'recall_macro']
    # scores = cross_validate(clf, x, y, return_train_score=True, 
    # n_jobs=4, scoring = scoring, verbose=3, cv=5)
    scores = split_train(clf, x, y)
    log_file = 'log\\SVMLinear1.json'
    log(log_file, scores)
    end = time.time()
    print('Finish SVM linear validation in %.4fs.' %(end-start))

def SVMPoly(x, y):
    from sklearn import svm

    start = time.time()
    clf = svm.SVC(kernel='poly', gamma='auto', coef0=0.10)
    # scores = cross_validate(clf, x, y, return_train_score=True, 
    # n_jobs=4, scoring = scoring, verbose=3, cv=5)
    scores = split_train(clf, x, y)
    log_file = 'log\\SVMPoly1.json'
    log(log_file, scores)
    end = time.time()
    print('Finish SVM poly validation in %.4fs.' %(end-start))

def knn(x, y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=1)
    scoring = ['precision_macro', 'recall_macro']
    # scores = cross_validate(clf, x, y, return_train_score=True, 
    # n_jobs=4, scoring = scoring, verbose=3, cv=5)
    scores = split_train(clf, x, y)
    log_file = 'log\\knn1.json'
    log(log_file, scores)
    end = time.time()
    print('Finish knn validation in %.4fs.' %(end-start))
    
def RF(x, y):
    from sklearn.ensemble import RandomForestClassifier

    start = time.time()
    clf = RandomForestClassifier(n_estimators=40, max_depth=None,
        min_samples_split=20, random_state=0)
    # scores = cross_validation(clf, x, y)
    scores = split_train(clf, x, y)
    log_file = 'log\\RF1.json'
    log(log_file, scores)
    
    end = time.time()
    print('Random Forest finished in %.4f' %(end-start))

def cross_validation(clf,x,y):
    scoring = ['accuracy', 'f1', 'precision', 'recall']
    scores = cross_validate(clf, x, y, return_train_score=True, 
    n_jobs=4, scoring = scoring, verbose=3, cv=5)
    sorted(scores.keys())
    data = {}
    for key in data.keys():
        data[key] = list(data[key])
    return data

def do_metrics(y_test,y_pred):
    from sklearn import metrics
    # [[TN, FP], [FN, TP]]
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = confusion_matrix[0][0], confusion_matrix[0][1], confusion_matrix[1][0], confusion_matrix[1][1]

    print('[[TN, FP], [FN, TP]]\n',confusion_matrix)
    scores = {}
    # TP+TN/(TP+TN+FP+NP)
    scores['accuracy_score'] =  metrics.accuracy_score(y_test, y_pred)
    # TP/(TP+FP)    
    scores['precision_score'] =  metrics.precision_score(y_test, y_pred)
    # TP/(TP+FN)    
    scores['recall_score'] = metrics.recall_score(y_test, y_pred)
    scores['false_alarm_rate'] = '%.6f' %((FP)/(TN+FP))
    scores['missing_rate'] = '%.6f' %(1-(TP)/(FN+TP))
    # 2* (Precision*Recall)/(precision+recall)
    scores['f1_score'] = metrics.f1_score(y_test,y_pred)

    return scores

def split_train(clf, x, y):
    labeled = 0.2
    unlabeled = 1- labeled
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=unlabeled, random_state=0)

    test_file = r'data\testFeature_useful.csv'
    x_test = y_test = np.array([])
    with open(test_file, 'r', encoding='utf-8') as f:
        datas = f.readlines()
        datas.pop(0)
        for data in datas:
            data = eval('['+data+']')
            x_test = np.insert(x_test,0,data[:-1],axis = 0)
            y_test = np.insert(y_test,0,data[-1],axis = 0)
            x_test.shape=(-1,len(data[:-1]))

    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return do_metrics(y_test, y_pred)

def log(file, data):
    import json    
    # plot(5, data, 'Random forest 5-fold validation')  

    for key in data:
        print(key + ': ', data[key])

    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

x = []
y = []
good = 'data\\good_example.csv'
evil = 'data\\xss_example.csv'
feature_file = 'data\\trainFeature_useful.csv'

start = time.time()
# process(good,x,y,0)
# process(evil,x,y,1)
# process = time.time()
# print('Finish Processing in %.4fs.' %(process-start))

import numpy as np
import pandas as pd
df = pd.read_csv(feature_file)

for data in df.values:
    y.append(data[-1])
    x.append(data[:-1])

# from sklearn.utils import shuffle
# x,y = shuffle(x,y)
# print("Data set has been shuffled.")

SVMLinear(x, y)
SVMPoly(x, y)
knn(x, y)
RF(x, y)
