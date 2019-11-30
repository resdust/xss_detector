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

normal_csv_file = "./normal_data.csv"
good_file = "good-xss-200000.txt"
goodfile = "good_example.csv"

xss_csv_file = "xss_data.csv"
xss_file = "xss-200000.txt"
xssfile = "xss_example.csv"

# length = writeEvil(xss_csv_file, xss_file, xssfile)
# writeNormal(normal_csv_file, good_file, goodfile, length)
# addEvil('normal_data.csv', 'normal_data2.csv')
dropRepeat('normal_examples.csv','normal_data2.csv')
# extractPayload('xss.csv','payload.csv')
