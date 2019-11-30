#!/usr/bin/python
from scapy.all import rdpcap
import scapy_http.http as http
import csv
import re
def analyseTEXT(logfile):
	import re
	reg = "GET*HTTP/1.1"
	strs = logfile.split('_')
	testfile = "test_"+strs[1]+strs[2]
	with open(logfile, 'r') as f:
		i = 0
		lines = f.readlines()
		new_line = ''
		for line in lines:
			#print(line)
			line = str(line.strip())
			res = re.findall("(?<=GET).*?(?=HTTP)", line)
			if(res):
				new_line = res[0].strip()
				print("line", i, new_line)
			else:
				pass
			i = i+1

def analysePCAP(file):
	a = rdpcap(file)
	# sessions = a.sessions()
	payloads = {}
	count = 0
	# for session in sessions:
	for packet in a:
		# data = ''
		# if packet.haslayer(http.TCP):
			if (packet.haslayer(http.HTTPRequest)):
				# http_header = packet[http.HTTPRequest].fields
				data = packet['TCP'].payload
				method = data.Method.decode("utf-8")
				# print("http_header:", http_header)
				if(method == 'GET'):
					payload = data.Path.decode("utf-8")
					reg_get = '\?(.*?)$'
					reg_result = re.search(reg_get, payload)
					if(reg_result):
						payload=reg_result.group(1)+'\n'
						if(payload not in payloads):
							payloads[payload] = 1
					else:
						continue
				else:
					continue
				# elif(method == 'POST'):
				# 	reg_type=r'Content-Type: (.*?)\\r'
				# 	data = str(data.original)
				# 	res = re.findall(reg_type, data)
				# 	# type = data.fields['Content-Type'].decode("utf-8")
				# 	if res:
				# 		type = res[0]
				# 		if type=="application/x-www-form-urlencoded":
				# 			reg_data=r'\\n(.*?)$'								
				# 		if type == "application/json":
				# 			reg_data=r'({.*?})$'
				# 		res = re.findall(reg_data, data)
				# 		if res:
				# 			payload = res[0]
				# 		else:
				# 			continue
				# 	else:
				# 		continue
				print(method + ", payload: "+payload)
				print("==============",count,"================")
				count = count + 1

	with open('data/normal_data2.csv','w',encoding='utf-8') as f:
		datas = [p for p in payloads]
		f.writelines(datas)

# analyseTEXT("data/tcpdump_2019916_023213.txt")

datas = []
logfile="../EvilData/http2.pcap"
print ("extracting payload from libpcap file: " + logfile )
analysePCAP(logfile)
# dpktPCAP(logfile)
print ("quit")
