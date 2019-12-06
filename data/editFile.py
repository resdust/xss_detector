import csv
import random

def writeNormal(infile1, infile2, outfile, length):
    with open(infile1, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
    with open(infile2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
        lines2 = lines2[:int(length/2)]
    resultList = random.sample(range(0, len(lines)), length - int(length/2))
    with open(outfile, 'w', encoding='utf-8') as f3:
        for i in resultList:
            f3.write(lines[i])
        f3.writelines(lines2)
        

def addEvil(src_file, dest_file):
    with open(src_file, 'r', encoding='utf-8') as f1:
        lines = f1.readlines()
    with open(dest_file, 'a', encoding='utf-8') as f2:
        f2.writelines(lines)

def dropRepeat(file, newfile):
    with open(file, 'r', encoding='utf-8',newline='') as f:
        lines = f.readlines()
        datas = {}
        for line in lines:
            if line not in datas:
                datas[line]=1
        
    with open(newfile,'w',encoding='utf-8',newline='') as f:
        for data in datas:
            f.write(data)

def extractPayload(file, newfile):
    import re 
    payloads = {}
    with open(file, 'r', encoding='utf-8',newline='') as f:
        lines = f.readlines()
        datas = lines[31000:]
    for data in datas:
        reg_get = '\?(.*?)$'
        reg_result = re.findall(reg_get, data)
        if(reg_result):
            data=reg_result[0]
            if data not in payloads:
                payloads[data]=1
            data = data+'\n'
        with open(newfile,'w',encoding='utf-8',newline='') as f:
            payload = [p for p in payloads.keys()]
            f.writelines(payload)

def extractXssLog(log, newfile):
    import re 

    payloads = {}
    with open(log, 'r', encoding='utf-8',newline='') as f:
        lines = f.readlines()
    for line in lines:
        reg = 'Payload: (.*?)$'
        reg_result = re.findall(reg, line)
        if(reg_result):
            data=reg_result[0]+'\n'
            if data not in payloads:
                payloads[data]=1
    
    with open(newfile,'w',encoding='utf-8',newline='') as f:
        payload = [p for p in payloads.keys()]
        f.writelines(payload)

def reWrite(old, new):
    import csv

    with open(old, 'r', encoding='utf-8') as f:
        datas = csv.reader(f, delimiter='\n')
        with open(new, 'w', encoding='utf-8', newline='') as f1:
            writer = csv.writer(f1)
            for data in datas:
                writer.writerow(data)

def mergeFiles(xss, normal, out):
    import csv

    with open(xss, 'r', encoding='utf-8') as f1:
        data1 = csv.reader(f1)
        with open(normal, 'r', encoding='utf-8') as f2:
            data2 = csv.reader(f2)
            with open(out, 'w', encoding='utf-8', newline='') as f3:
                writer = csv.writer(f3)
                for data in data1:
                    new = [data[0],'1']
                    writer.writerow(new)
                for data in data2:
                    new = [data[0],'0']
                    writer.writerow(new)

def splitFile(input):
    pass
normal_csv_file = "./normal_data.csv"
good_file = "good-xss-200000.txt"
goodfile = "good_example.csv"

xss_csv_file = "xss_data.csv"
xss_file = "xss-200000.txt"
xssfile = "xss_example.csv"

xss1 = r"data\xss_data_3k_xsstrik.csv"
xss2 = r"data\xss_data_28k_xssed.csv"

normal1 = r"data\normal_data_39k_bupt.csv"
normal2 = r"data\normal_data_162k_dl.csv"

test = r"data\test.csv"
train = r"data\train.csv"
labeled_20 = r"data\labeled_20.csv"
labeled_40 = r"data\labeled_40.csv"
unlabeled_20 = r"data\unlabeled_20.csv"
unlabeled_40 = r"data\unlabeled_40.csv"

# length = writeEvil(xss_csv_file, xss_file, xssfile)
# writeNormal(normal_csv_file, good_file, goodfile, length)
# addEvil('normal_data.csv', 'normal_data2.csv')
# dropRepeat('normal_examples.csv','normal_data2.csv')
# extractPayload('xss.csv','payload.csv')
# extractXssLog('xsstrike-good.log','xss_data2.csv')
# reWrite(r'data\xss_data_3k_xsstrike.csv', r'data\xss_data_3k_xsstrik2.csv')
mergeFiles(xss1, normal1,test)
mergeFiles(xss2, normal2,train)

splitFile(train)