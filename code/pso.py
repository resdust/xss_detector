# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import random
from math import floor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from xssExtractor import etl

input_dim = 4
output_dim = 1
batchsize = 128
alpha = 0.01

#----------------------PSO参数设置---------------------------------  
class PSO():
    def __init__(self,pN,max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1= 0.6
        self.r2= 0.3
        self.pN = pN                                #粒子数量
        self.max_iter = max_iter                    #迭代次数  
        self.X = np.zeros(self.pN).astype(int)      #所有粒子的位置和速度
        self.V = np.zeros(self.pN).astype(int)
        self.pbest = np.zeros(self.pN).astype(int)  #个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros(1).astype(int)
        self.p_fit = np.zeros(self.pN)              #每个个体的历史最佳适应值  
        self.fit = 0                                #全局最佳适应值

# ----------------------初始化内层种群---------------------------------
    def inner_init(self,pN,dim,max_iter):
        self.ipN = pN                #粒子数量  
        self.idim = dim              #搜索维度  
        self.imax_iter = max_iter    #迭代次数

        #输入层到隐藏层的权重矩阵
        self.iX = np.zeros((self.ipN, input_dim, self.idim))       #所有粒子的位置和速度
        self.iV = np.zeros((self.ipN, input_dim, self.idim))
        self.ipbest = np.zeros((self.ipN, input_dim, self.idim))   #个体经历的最佳位置和全局最佳位置
        self.igbest = np.zeros((1, input_dim, self.idim))

        #隐藏层到输出层的权重矩阵
        self.iX2 = np.zeros((self.ipN, self.idim, output_dim))
        self.iV2 = np.zeros((self.ipN, self.idim, output_dim))
        self.ipbest2 = np.zeros((self.ipN, self.idim, output_dim))
        self.igbest2 = np.zeros((1, self.idim, output_dim))

        self.ip_fit = np.zeros((self.ipN, self.ipN))               #每个个体的历史最佳适应值
        self.ifit = 0                                              #全局最佳适应值

        for i in range(self.ipN):
            for j in range(input_dim):
                for k in range(self.idim):
                    self.iX[i][j][k] = random.uniform(0,1)
                    self.iV[i][j][k] = random.uniform(0,1)
            self.ipbest[i] = self.iX[i]

        self.igbest = self.iX[0]

        for i in range(self.ipN):
            for j in range(self.idim):
                for k in range(output_dim):
                    self.iX2[i][j][k] = random.uniform(0,1)
                    self.iV2[i][j][k] = random.uniform(0,1)
            self.ipbest2[i] = self.iX2[i]

        self.igbest2 = self.iX2[0]

        for i in range(self.ipN):
            for j in range(self.ipN):
                self.ip_fit[i][j] = 0

#---------------------目标函数-----------------------------
    def function(self,  sess, M, w_h, w_o):
        # 定义神经网络模型
        def model(X, w_h, w_o):
            h = tf.matmul(X, w_h)
            return tf.matmul(h, w_o)

        X = tf.placeholder(tf.float64, [None, 4])
        Y = tf.placeholder(tf.float64, [None, 1])

        w_h = tf.Variable(w_h)  # 输入层到隐藏层的权重矩阵
        w_o = tf.Variable(w_o)  #隐藏层到输出层的权重矩阵

        model = model(X, w_h, w_o)                 
        loss = tf.reduce_mean((model-Y)**2)  # 计算均方误差

        tf.initializers.global_variables().run(session=sess)
		
        sum = 0
        length = len(trY)
        for start, end in zip(range(0, length, batchsize), range(batchsize, length + 1, batchsize)):
            sum += sess.run(loss, feed_dict={X: trX[start:end], Y: trY[start:end]})
        if not length%batchsize:    #论文没有对分批处理的算法做介绍，这里对所有批的均方误差取平均值了
            E = sum/(length/batchsize)
        else:
            E = sum/(1+floor(length/batchsize))
        return 1/(E+alpha*M)

#---------------------初始化外层种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            self.X[i] = random.randint(1,100)
            self.V[i] = random.randint(1,100)
            self.pbest[i] = self.X[i]
            self.p_fit[i] = 0
        self.fit = 0
        self.gbest = self.X[0]

#----------------------更新粒子位置----------------------------------
    def iterator(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
        sess = tf.Session(config=config)

        fitness = []
        ifitness = 0
        for t in range(self.max_iter):
            for i in range(self.pN):         #更新gbest/pbest
                self.inner_init(10,self.X[i],50)
                for it in range(self.imax_iter):
                    for ii in range(self.ipN):
                        with open("schedule.txt","w") as f:
                            f.write("目前进度：外层第%d轮第%d个粒子，内层第%d轮第%d个粒子\n"%(t,i,it,ii))
                            f.close()
                        for ii2 in range(self.ipN):
                            temp = self.function(sess, self.X[i],self.iX[ii],self.iX2[ii2])
                            if(temp > self.ip_fit[ii][ii2]):      #更新个体最优
                                self.ip_fit[ii][ii2] = temp
                                self.ipbest[ii] = self.iX[ii]
                                self.ipbest2[ii2] = self.iX2[ii2]
                                if(self.ip_fit[ii][ii2] > self.ifit):  #更新全局最优
                                    self.igbest = self.iX[ii]
                                    self.igbest2 = self.iX2[ii2]
                                    self.ifit = self.ip_fit[ii][ii2]
                        for ii2 in range(self.ipN):
                            self.iV2[ii2] = self.w * self.iV2[ii2] + self.c1 * self.r1 * (self.ipbest2[ii2] - self.iX2[ii2]) + \
                                          self.c2 * self.r2 * (self.igbest2 - self.iX2[ii2])
                            self.iX2[ii2] = self.iX2[ii2] + self.iV2[ii2]
                    for ii in range(self.ipN):
                        self.iV[ii] = self.w*self.iV[ii] + self.c1*self.r1*(self.ipbest[ii] - self.iX[ii]) + \
                            self.c2*self.r2*(self.igbest - self.iX[ii])
                        self.iX[ii] = self.iX[ii] + self.iV[ii]
                    ifitness = max(self.ifit,ifitness)
                    #print(it,ifitness)
                if(ifitness > self.p_fit[i]):
                    self.p_fit[i] = ifitness
                    self.pbest = self.X[i]
                    if(self.p_fit[i] > self.fit):
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
                with open ("pso.out","a") as f:
                    f.write("第%d轮第%d个粒子对应适应度函数值为%lf "%(t,i,self.fit))  #输出最优值
                    f.write(str(self.gbest)+' igbest='+str(self.igbest)+' igbest2='+str(self.igbest2)+'\n')
                    f.close()
            with open("pso.out", "a") as f:
                f.write(str(t)+" "+str(self.fit)+" "+str(self.gbest)+" "+str(self.igbest)+" "+str(self.igbest2)+'\n')
                f.close()
            for i in range(self.pN):
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i]) + \
                            self.c2*self.r2*(self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
        return fitness

#----------------------程序执行-----------------------
if __name__ == "__main__":
    x = []  # feature data vectors
    y = []  # labels
	
    # Read data from pickle
    etl('data/good_example.csv', x, y, 1)
    etl('data/xss_example.csv', x, y, 0)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    y.shape=-1,1

    trX, teX, trY, teY = train_test_split(x, y, test_size=0.3, random_state=0)

    my_pso = PSO(pN=10, max_iter=50)
    my_pso.init_Population()
    fitness = my_pso.iterator()

    plt.title("pso-bp")
    plt.plot(50, fitness)
    plt.xlabel('epoch')
    plt.ylabel('fitness')
    # 可视化图
    plt.show()
