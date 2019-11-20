import re

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
    # return len(re.findall("[`~!@#$%^*(_)-+=<>:\"{}| ,./;\'\\[]·]", url, re.IGNORECASE))
    return len(re.findall("[<>,\'\"/]", url,re.IGNORECASE))

def get_evil_word(url):
    XSS = ["window","location","this","iframe","response_write","prompt","javascript",
    "document","cookie","alert","script","onerror","onload","eval","createelement",
    "string","search","cmd","div","img","<script","href","src","var","prompt",".js"]
    # XSS = ["response_write","prompt","javascript",
    # "cookie","alert","script","onerror","onload","eval","createelement",
    # "cmd","<script","href","src",".js"]
    XSS = '(' + ')|('.join(XSS) + ')'
    return len(re.findall( XSS, url, re.IGNORECASE))

def get_feature(url):
    return [get_len(url),get_evil_char(url),get_evil_word(url)]

def do_metrics(y_test,y_pred):
    from sklearn import metrics
    
    # [[TP, FN], [FP, TN]]
    print ("metrics.confusion_matrix:  [[TP, FN], [FP, TN]] \n", metrics.confusion_matrix(y_test, y_pred))
    # TP+TN/(TP+TN+FP+NP)
    print ("metrics.accuracy_score:", metrics.accuracy_score(y_test, y_pred))
    # TP/(TP+FP)
    print ("metrics.precision_score:", metrics.precision_score(y_test, y_pred))
    # TP/(TP+FN)
    print ("metrics.recall_score:", metrics.recall_score(y_test, y_pred))
    print ("metrics.f1_score:", metrics.f1_score(y_test,y_pred))

# extract features to list and dump into file
# return labels
def extract_feature(filename,data,y,isxss):
    import pickle
    import time

    pk_file = filename[:-3]+'pickle'
    start = time.time() 
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = decode(line)
            f1=get_len(line)
            f2=get_url_count(line)
            f3=get_evil_char(line)
            f4=get_evil_word(line)
            data.append([f1,f2,f3,f4])
            y.append(isxss)
    end = time.time()
    print('Finish feature extraction in %.4fs.' %(end-start))

    with open(pk_file, 'wb') as pkf:
        pickle.dump(data, pkf)
        
    return y

def normalization(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x -= mean
    x /= std

if __name__ == "__main__":

    x = [] # feature data vectors
    y = [] # labels
    
    extract_feature('data/good_example.csv',x,y,0)
    extract_feature('data/xss_example.csv',x,y,1)
    