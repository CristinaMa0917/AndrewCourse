import matplotlib.pyplot as plt
import numpy as np
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


class deepNN():
    def __init__(self):
        self.l = len(self.get_para(para))
        self.ld = np.r_[self.get_x(x)[0],self.get_para(para)]
        m = self.get_x(x)[1]
        print(self.ld)
        self.w = []
        self.b = []
        self.a = []
        self.z = []
        self.dw = []
        self.db = []
        self.da = []
        self.dz = []
        for i in range(self.l):
            self.a.append(np.random.randn(self.ld[i+1],m))
            self.z.append(np.random.randn(self.ld[i+1],m))
            self.w.append(np.random.randn(self.ld[i+1],self.ld[i]))
            self.b.append(np.random.randn(self.ld[i+1],1))

            self.da.append(np.random.randn(self.ld[i+1],m))
            self.dz.append(np.random.randn(self.ld[i+1],m))
            self.dw.append(np.random.randn(self.ld[i+1],self.ld[i]))
            self.db.append(np.random.randn(self.ld[i+1],1))

        #print(str(self.w)+"===========/n"+str(self.b)+ "init finished")

    def get_para(self, para):
        return para

    def get_x(self, x):
        return np.shape(x)

    def forPpRelu(self,i,x): # i= 0..l-1 a_former 0..l-1 a[0] =
        if i >0 :
            self.z[i] = np.dot(self.w[i],self.a[i-1])+self.b[i] # z_later  1..l  a[0] = x
        else:
            self.z[i] = np.dot(self.w[i],x) + self.b[i]
        self.a[i] = np.maximum(self.z[i],0) # 1..l a[l] = y

    def forPpSig(self,i):
        self.z[i] = np.dot(self.w[i],self.a[i-1])+self.b[i] # z_later  1..l  a[0] = x
        self.a[i] = 1.0/(1+np.exp(-self.z[i])) # 1..l a[l] = y

    def backPpRelu(self,i):
        gz = np.where(self.z[i]>0,1,0)
        m = self.get_x(x)[1]
        self.dz[i] = self.da[i]*gz
        if i >0:
            self.dw[i] = 1.0/m*np.dot(self.dz[i],self.a[i-1].T)
        else:
            self.dw[i] = 1.0/m*np.dot(self.dz[i],x.T)
        self.db[i] = 1/m*np.sum(self.dz[i],axis=1,keepdims=True)
        if i>0 :
            self.da[i-1] = np.dot(self.w[i].T,self.dz[i])

    def backPpSig(self,i):
        gz = 1.0/(1+np.exp(-self.z[i]))
        gz = gz*(1-gz)
        m = self.get_x(x)[1]
        self.dz[i] = self.da[i]*gz
        if i>0:
            self.dw[i] = 1.0/m*np.dot(self.dz[i],self.a[i-1].T)
        else:
            self.dw[i] = 1.0/m*np.dot(self.dz[i], x.T)
        self.db[i] = 1.0/m*np.sum(self.dz[i],axis=1,keepdims=True)
        if i > 0 :
            self.da[i-1] = np.dot(self.w[i].T,self.dz[i])

    def update(self,lr):
        for i in range(self.l):
            self.w[i] = self.w[i]-lr*self.dw[i] #  i+1,i
            self.b[i] = self.b[i]-lr*self.db[i] # i+1

    def train(self,x,y,epoch,lr):
        m = self.get_x(x)[1]
        for k in range(epoch):
            # forward
            for i in range(self.l-1): # 0,1
                self.forPpRelu(i,x)
            self.forPpSig(self.l-1) # 2

            # backward
            self.dz[self.l-1] = self.a[self.l-1] - y # 确定最后一层是sigmoid
            self.dw[self.l-1] = 1.0/m*np.dot(self.dz[self.l-1],self.a[self.l-2].T)
            self.db[self.l-1] = 1.0/m*np.sum(self.dz[self.l-1],axis = 1,keepdims=True)
            self.da[self.l-2] = np.dot(self.w[self.l-1].T,self.dz[self.l-1])
            for i in range(self.l-2,-1,-1): # 1 0
                self.backPpRelu(i)

            #print(self.w)
            self.update(lr)

    def predict(self,x):
        for i in range(self.l - 1):  # 0,1
            self.forPpRelu(i,x)
        self.forPpSig(self.l - 1)# 2
        return np.where(self.a[self.l-1]>0.5,1,0)

para = [103,50,25,5,2,1]

x,y = load_planar_dataset()
y = y.ravel()
print(x.shape)
print(len(y))
'''
para = [4,5,1]
dnn = deepNN()
dnn.train(x,y,10000,0.8)
yhat = dnn.predict(x)
print(yhat.shape)

print("precision of nn:"+str(100-(np.sum(np.abs(yhat-y)))/4)+"%")
plot_decision_boundary(lambda x:dnn.predict(x.T),x,y)
plt.show()
'''
