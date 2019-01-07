# Package imports
from model_package import *
from lr import logistic_regression
from nn_model import nn_model

np.random.seed(1) # set a seed so that the results are consistent
X, Y = load_planar_dataset()        #加载数据
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral) # 可视化数据
# plt.show()

### 数据处理
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size

# 逻辑回归模型
# logistic_regression(X,Y,isplot=True)

# 神经网络模型
# 测试不同隐层神经元个数
# hidden_layer_sizes = [4]
hidden_layer_sizes = [1, 2, 3, 4, 5, 20]
nn_model(X,Y,hidden_layer_sizes,num_iterations= 10000,print_cost= False)