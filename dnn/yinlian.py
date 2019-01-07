import pandas as pd
import scipy as sp
import numpy as np
#import statsmodel as ss
import matplotlib.pyplot as plt
# import syspy
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import seaborn as sns

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

raw_data = pd.read_csv('/Users/luma/Downloads/银联商务/sample/model_sample.csv')
totalN = 11017
columnsN =  201

y_total = raw_data['y']
x_total = raw_data.drop('y',axis =1)

iori = x_total.columns.ravel()
idiv = [iori[1:20],iori[21:41],iori[41:131],iori[131:147],iori[147:188],iori[188:200]]
na = x_total.isna()
na_num = na[na == True].sum()
lowna = na_num[na_num<totalN*0.3]# 105
zerona = na_num[na_num == 0] # 40
selectedIndex = lowna.index.ravel()[1:]


x_select = x_total[selectedIndex]
x_train, x_test, y_train, y_test = train_test_split(x_select, y_total, test_size=0.33, random_state=42)

para = [50,25,5,2,1]
x = x_train.fillna(0).T
y = y_train.tolist()
dnn = deepNN()
dnn.train(x,y,1000,0.8)
y_hat = dnn.predict(x_test.fillna(0).T)
y_label = np.array(y_test)
print("precision of nn:"+str((1-(np.sum(np.abs(y_hat[0]-y_label)))/len(y_label))*100)+"%")