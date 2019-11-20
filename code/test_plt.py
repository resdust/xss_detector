import matplotlib.pyplot as plt
import numpy as np
# test_precision_macro = [0.8238113,  0.69850414, 0.68124291, 0.79054854, 0.99849548]
# train_precision_macro = [0.79773786, 0.82118044, 0.82110249, 0.8073124,  0.77621944]
# test_recall_macro = [0.72888413, 0.50091205, 0.50083274, 0.64077246, 0.99849302]
# train_recall_macro = [0.66092,    0.72201844, 0.72183008, 0.6870923,  0.59738084]

# plt.plot([i+1 for i in range(5)], test_precision_macro, c='red', label='test precision')
# plt.plot([i+1 for i in range(5)], test_recall_macro, c='orange', label='test recall')
# plt.plot([i+1 for i in range(5)], train_precision_macro, c='blue', label='train precision')
# plt.plot([i+1 for i in range(5)], train_recall_macro, c='green', label='train recall')
# plt.xlabel('Time')
# plt.ylabel('Score')
# plt.title('k-NN scores with n_neigjbors = 1')
# plt.legend(loc='best')

# plt.show()
n_dots=200
#下面两行就是在生成数据集
X=np.linspace(0,1,n_dots)#从0到1之间生成两百个数。
y=np.sqrt(X)+0.2*np.random.rand(n_dots)-0.1;
#下面两行就是n_sample * n_feature的矩阵，将其特征固定为1，其中-1的意思就是全部
X=X.reshape(-1,1)
y=y.reshape(-1,1)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def polynomial_model(degree=1):#degrees代表的是多项式阶数
    polynomial_features=PolynomialFeatures(degree=degree,include_bias=False)#模型生成，没有偏差
    linear_regression=LinearRegression()#线性回归算法
    pipeline=Pipeline([("polynomial_features",polynomial_features),("linear_regression",linear_regression)])
    #流水线生产多项式模型，之后使用线性回归拟合数据
    return pipeline


def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,train_sizes=np.linspace(0.1,1.0,5)):
    plt.title(title)#图像标题
    if ylim is not None:#y轴限制不为空时
        plt.ylim(*ylim)
    plt.xlabel("Training examples")#两个标题
    plt.ylabel("Score")
    train_sizes,train_scores,test_scores=learning_curve(estimator,X,y,cv=cv,n_jobs=n_jobs,train_sizes=train_sizes)#获取训练集大小，训练得分集合，测试得分集合
    train_scores_mean=np.mean(train_scores,axis=1)#将训练得分集合按行的到平均值
    train_scores_std=np.std(train_scores,axis=1)#计算训练矩阵的标准方差
    test_scores_mean=np.mean(test_scores,axis=1)
    test_scores_std=np.std(test_scores,axis=1)
    plt.grid()#背景设置为网格线
    
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='r')
    # plt.fill_between()函数会把模型准确性的平均值的上下方差的空间里用颜色填充。
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='g')
    plt.plot(train_sizes,train_scores_mean,'o-',color='r',label='Training score')
    # 然后用plt.plot()函数画出模型准确性的平均值
    plt.plot(train_sizes,test_scores_mean,'o-',color='g',label='Cross_validation score')
    plt.legend(loc='best')#显示图例
    return plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#交叉验证类进行十次迭代，测试集占0.2，其余的都是训练集
titles = ['Learning Curves(Under Fitting)', 'Learning Curves', 'Learning Curves(Over Fitting)']
degrees = [1, 3, 10]#多项式的阶数
plt.figure(figsize=(18, 4), dpi=200)#设置画布大小，dpi是每英寸的像素点数
for i in range(len(degrees)):#循环三次
    plt.subplot(1, 3, i + 1)#下属三张画布，对应编号为i+1
    plot_learning_curve(polynomial_model(degrees[i]), titles[i], X, y, ylim=(0.75, 1.01), cv=cv)#开始绘制曲线

plt.show()#显示
