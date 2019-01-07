import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases_v2 import *
from planar_utils import plot_decision_boundary,sigmoid,load_extra_datasets,load_planar_dataset

np.random.seed(1)

def load_planar_dataset():
    np.random.seed(1)
    m = 400 # 样本数量
    N = int(m/2) # 每个类别的样本量
    D = 2 # 维度数
    X = np.zeros((m,D)) # 初始化X
    Y = np.zeros((m,1), dtype='uint8') # 初始化Y
    a = 4 # 花儿的最大长度

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y

X,Y = load_planar_dataset()
Y = Y.ravel()
#plt.scatter(X[0,:],X[1,:],c=Y,s=40,cmap = plt.cm.Spectral)
#plt.title("original data")

# try lr to classify this sample
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)

#plot_decision_boundary(lambda x:clf.predict(x),X,Y)
#plt.title("logistic regression")
yhat = clf.predict(X.T)
print("precision of lr: "+str(100 - (np.sum(np.abs(yhat - Y))/400))+"%")

# use one hidden layer NN
class NNsimple():
    def __init__(self):
        self.w1 = np.random.randn(4,self.get_nx(x)[0])  #（4，2）
        self.b1 = np.random.randn(4,1)
        self.w2 = np.random.randn(self.get_ny(y)[0],4) #（1，4）
        self.b2 = np.random.randn(1,1)

    def get_nx(self,x):
        return np.shape(x) # (2,400) 2

    def get_ny(self,y):
        return np.shape(y) # (1,400) 1

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def forwardPp(self,x):
        z1 = np.dot(self.w1,x)+self.b1 # 4 400
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2,a1)+self.b2 # 1 400
        a2 = self.sigmoid(z2)
        return z1,a1,z2,a2

    def backPp(self,x,y):
        m = self.get_nx(x)[1]
        dz2 = self.forwardPp(x)[3] - y #1 400
        dw2 = 1.0/m*np.dot(dz2,self.forwardPp(x)[1].T) # 1 4
        db2 = 1.0/m*np.sum(dz2,axis=1,keepdims=True) # 1 1

        g_z1prime = 1.0-np.power(self.forwardPp(x)[1],2)
        dz1 = np.dot(self.w2.T,dz2)*g_z1prime # 4 400
        dw1 = 1.0/m*np.dot(dz1,x.T) # 4 2
        db1 = 1.0/m*np.sum(dz1,axis = 1,keepdims= True) # 4 1

        return dw1,db1,dw2,db2

    def update(self,alpha,x,y):
        self.w1 = self.w1 - alpha*self.backPp(x,y)[0]
        self.b1 = self.b1 - alpha*self.backPp(x,y)[1]
        self.w2 = self.w2 - alpha*self.backPp(x,y)[2]
        self.b2 = self.b2 - alpha*self.backPp(x,y)[3]
        return self.w1,self.b1,self.w2,self.b2

    def cost(self,x,y):
        logpros = np.multiply(np.log(self.forwardPp(x)[3]),y)
        return -np.sum(logpros)

    def predict(self,x):
        y = self.forwardPp(x)[3]
        y = np.where(y >0.5,1,0)
        return y.flatten()

    def train(self,x,y,alpha,epoch):
        for i in range(epoch):
            self.forwardPp(x)
            self.backPp(x,y)
            self.update(alpha,x,y)
        print(self.cost(x,y))


x,y = load_planar_dataset()
nn = NNsimple()
nn.train(X,y,1.2,10000)
yhat = nn.predict(x)
y = y.flatten()
print("precision of nn:"+str(100-(np.sum(np.abs(yhat-y)))/4)+"%")

plot_decision_boundary(lambda x:nn.predict(x.T),x,y)
plt.show()




