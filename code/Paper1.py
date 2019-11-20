import re
import numpy as np
import time
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
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, x, y, return_train_score=True, 
    n_jobs=4, scoring = scoring, verbose=3, cv=5)
    sorted(scores.keys())
    for key in scores.keys():
        print(key + ': ' , scores[key])
    end = time.time()
    plot(5, scores, 'SVM Linear 5-fold validation')
    print('Finish SVM linear 5-fold validation in %.4fs.' %(end-start))

def SVMPoly(x, y):
    from sklearn import svm

    start = time.time()
    clf = svm.SVC(kernel='poly', gamma='auto', coef0=0.10)
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, x, y, return_train_score=True, 
    n_jobs=4, scoring = scoring, verbose=3, cv=5)
    sorted(scores.keys())
    for key in scores.keys():
        print(key + ': ' , scores[key])
    end = time.time()
    plot(5, scores, 'SVM Polynomial 5-fold validation')
    print('Finish SVM poly 5-fold validation in %.4fs.' %(end-start))

def knn(x, y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=1)
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, x, y, return_train_score=True, 
    n_jobs=4, scoring = scoring, verbose=3, cv=5)
    sorted(scores.keys())
    for key in scores.keys():
        print(key + ': ' , scores[key])
    end = time.time()
    plot(5, scores, 'K-Nearest Neighber 5-fold validation')
    print('Finish knn 5-fold validation in %.4fs.' %(end-start))
    
def RF(x, y):
    from sklearn.ensemble import RandomForestClassifier

    start = time.time()
    clf = RandomForestClassifier(n_estimators=40, max_depth=None,
        min_samples_split=20, random_state=0)
    scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(clf, x, y, return_train_score=True, 
    n_jobs=4, scoring = scoring, verbose=3, cv=5)
    sorted(scores.keys())
    for key in scores.keys():
        print(key + ': ' , scores[key])
    end = time.time()
    plot(5, scores, 'Random forest 5-fold validation')
    print('Random Forest 5-fold finished in %.4f' %(end-start))

x = []
y = []
good = 'data\\good_example.csv'
evil = 'data\\xss_example.csv'

start = time.time()
process(good,x,y,0)
process(evil,x,y,1)
process = time.time()
print('Finish Processing in %.4fs.' %(process-start))

from sklearn.utils import shuffle
x,y = shuffle(x,y)
print("Data set has been shuffled.")

SVMLinear(x, y)
SVMPoly(x, y)
# knn(x, y)
# RF(x, y)
